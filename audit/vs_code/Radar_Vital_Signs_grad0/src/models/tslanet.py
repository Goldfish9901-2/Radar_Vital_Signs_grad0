"""TSLANet-style backbone for radar heart-rate regression.

Ported from the official repo (emadeldeen24/TSLANet, Forecasting/TSLANet_Forecasting.py).
Supports two ``channel_mode`` settings as a deliberate experimental factor (see the
integration review doc, factor F1):

* ``"joint"``   — channels are mixed inside the patch projection (``Linear(C*P, emb)``),
                  matching the other backbones in this repo. This is the Phase-1 path.
* ``"official"``— channel-independent: each channel is processed as its own batch element
                  (``Linear(P, emb)``), mirroring the official TSLANet forward exactly, then
                  the per-channel features are mean-aggregated to a single HR scalar.

Deliberate, documented deviations from the official implementation are listed in
``TSLANET_DEVIATIONS`` below so they are not mistaken for canonical TSLANet behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Tiny local reimplementations to avoid new dependencies (timm/einops).
# ---------------------------------------------------------------------------
def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> Tensor:
    """Truncated normal initialization (copied from timm)."""
    with torch.no_grad():
        def _norm_cdf(x: Tensor) -> Tensor:
            return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

        l = _norm_cdf(torch.tensor(a))
        u = _norm_cdf(torch.tensor(b))
        # Map the truncated-normal CDF interval (l, u) onto (-1, 1) so erfinv_ is
        # well-defined. (timm's trunc_normal_ uses 2*l - 1, 2*u - 1.)
        tensor.uniform_(2 * l.item() - 1, 2 * u.item() - 1)
        tensor.erfinv_()
        tensor.mul_(std * torch.sqrt(torch.tensor(2.0)))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """Stochastic depth (copied from timm)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            random = random.div_(keep)
        return x * random


# ---------------------------------------------------------------------------
# Atomic modules (ported; base class LightningModule -> nn.Module;
# global ``args`` replaced by constructor arguments).
# ---------------------------------------------------------------------------
class AdaptiveSpectralBlock(nn.Module):
    """Frequency-domain adaptive filter (ASB)."""

    def __init__(self, dim: int, adaptive_filter: bool = True) -> None:
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=0.02)
        trunc_normal_(self.complex_weight, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft: Tensor) -> Tensor:
        B = x_fft.shape[0]
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in: Tensor) -> Tensor:
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted = x_weighted + x_masked * weight_high

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm="ortho")
        x = x.to(dtype)
        return x.view(B, N, C)


class InteractiveConvBlock(nn.Module):
    """Interactive convolution block (ICB)."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class TSLANetLayer(nn.Module):
    """One encoder block: norm -> ASB -> norm -> ICB -> drop-path residual."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_asb: bool = True,
        use_icb: bool = True,
        adaptive_filter: bool = True,
    ) -> None:
        super().__init__()
        self.use_asb = use_asb
        self.use_icb = use_icb
        self.adaptive_filter = adaptive_filter
        self.norm1 = norm_layer(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter) if use_asb else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = InteractiveConvBlock(dim, mlp_hidden_dim, drop) if use_icb else None

    def forward(self, x: Tensor) -> Tensor:
        if self.use_asb and self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TSLANetConfig:
    time_channels: int = 7
    freq_channels: int = 7
    emb_dim: int = 64
    depth: int = 3
    dropout: float = 0.5          # official default; higher than this repo's other models
    patch_len: int = 16           # repo default (PatchTST-style); official uses 64
    patch_stride: int = 8         # repo default; official uses 32
    use_asb: bool = True
    use_icb: bool = True
    adaptive_filter: bool = True
    normalize: bool = True         # internal RevIN-style instance norm
    use_frequency_domain: bool = False  # Phase 1: time-only branch
    channel_mode: str = "joint"    # Phase 2 adds "official" (channel-independent)


# ---------------------------------------------------------------------------
# Branch (encoder) + top-level model
# ---------------------------------------------------------------------------
class TSLANetBranch(nn.Module):
    """Single-branch encoder: patch-embed -> stacked TSLANetLayer -> pool."""

    def __init__(self, in_channels: int, cfg: TSLANetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.input_layer = nn.Linear(in_channels * cfg.patch_len, cfg.emb_dim)
        dpr = [x.item() for x in torch.linspace(0, cfg.dropout, cfg.depth)]
        self.blocks = nn.ModuleList(
            [
                TSLANetLayer(
                    cfg.emb_dim,
                    mlp_ratio=3.0,
                    drop=cfg.dropout,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    use_asb=cfg.use_asb,
                    use_icb=cfg.use_icb,
                    adaptive_filter=cfg.adaptive_filter,
                )
                for i in range(cfg.depth)
            ]
        )
        self.feature_dim = cfg.emb_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) < self.patch_len:
            x = F.pad(x, (0, self.patch_len - x.size(-1)), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        num_patches = patches.shape[2]
        tokens = patches.permute(0, 2, 1, 3).reshape(x.size(0), num_patches, -1)
        tokens = self.input_layer(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens.mean(dim=1)


class TSLANetHeartRateModel(nn.Module):
    """TSLANet backbone for single-value HR regression.

    Inputs:
        x_time: (B, C, T)
        x_freq: (B, C, F)  # optional, only used when use_frequency_domain=True
    Output:
        scalar HR prediction, shape (B,)
    """

    def __init__(self, cfg: TSLANetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TSLANetConfig()
        if self.cfg.channel_mode not in ("joint", "official"):
            raise ValueError(f"Unknown channel_mode: {self.cfg.channel_mode!r}")
        # In official (channel-independent) mode each channel is unfolded separately,
        # so the patch embedding sees a single channel per token -> in_channels = 1.
        time_in = 1 if self.cfg.channel_mode == "official" else self.cfg.time_channels
        self.time_branch = TSLANetBranch(time_in, self.cfg)
        self.freq_branch = (
            TSLANetBranch(self.cfg.freq_channels, self.cfg)
            if self.cfg.use_frequency_domain
            else None
        )
        fusion_dim = self.cfg.emb_dim * (2 if self.cfg.use_frequency_domain else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.cfg.emb_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.emb_dim, 1),
        )

    def _norm(self, x: Tensor) -> Tensor:
        if not self.cfg.normalize:
            return x
        means = x.mean(-1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, -1, keepdim=True, unbiased=False) + 1e-5).detach()
        return (x - means) / stdev

    def forward(self, x_time: Tensor, x_freq: Tensor | None = None) -> Tensor:
        if self.cfg.channel_mode == "official":
            B, C, T = x_time.shape
            # Per-channel RevIN-style normalization, then treat each channel as an
            # independent sample: (B, C, T) -> (B*C, 1, T). This exactly mirrors the
            # official TSLANet forward (rearrange b l m -> b m l, unfold, reshape
            # (b m) n p). After the encoder we mean-pool channels to one HR scalar.
            x_norm = self._norm(x_time).reshape(B * C, 1, T)
            token = self.time_branch(x_norm).reshape(B, C, -1).mean(dim=1)
        else:
            token = self.time_branch(self._norm(x_time))
        features = [token]
        if self.freq_branch is not None:
            if x_freq is None:
                raise ValueError("x_freq is required when use_frequency_domain=True")
            features.append(self.freq_branch(self._norm(x_freq)))
        return self.head(torch.cat(features, dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Documented deviations from the official implementation (for reviewers).
# ---------------------------------------------------------------------------
TSLANET_DEVIATIONS = [
    "channel_mode: 'joint' mixes channels inside the patch projection (Linear(C*P, emb)); 'official' "
    "replicates the official channel-independent forward exactly (Linear(P, emb) per channel, channels "
    "flattened into the batch) and then mean-aggregates the per-channel features to a single HR scalar. "
    "The two modes are the F1 experimental factor (tslanet_joint vs tslanet_official).",
    "Official channel aggregation: the official forecasting head produces one series per channel and "
    "de-normalizes each; for single-value HR we instead mean-pool the per-channel encoder features to "
    "one (B, emb) feature, then regress. This aggregation step is a design choice documented here, not "
    "part of the canonical TSLANet forward.",
    "Patch size: default patch_len=16 / patch_stride=8 (repo PatchTST-style) instead of official 64 / 32.",
    "Head: regression head mean-pools patch tokens then Linear->1 (scalar). Official forecasting head "
    "flattens all tokens and projects to pred_len.",
    "No output de-normalization: the official RevIN denorm is dropped because the target is a single HR "
    "scalar, not a reconstructed series. Only input normalization is kept (controlled by cfg.normalize).",
    "Self-supervised masked pretraining is not used; trained supervised from scratch (consistent with "
    "other backbones in this repo).",
    "LightningModule base classes and the global `args` object from the official code were removed; "
    "sub-modules are plain nn.Module with flags passed via constructors.",
    "einops.rearrange replaced by explicit unfold/reshape; timm DropPath/trunc_normal_ reimplemented "
    "locally to avoid new dependencies (trunc_normal_ verified to match timm's 2*l-1 / 2*u-1 uniform "
    "bounds — a mismatch here produced NaN weights, see fix).",
    "freq branch (use_frequency_domain) defaults to False; feeding a spectrum into ASB is "
    "semantically questionable and its handling is a deferred ablation (R8), NOT wired in this phase.",
]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
