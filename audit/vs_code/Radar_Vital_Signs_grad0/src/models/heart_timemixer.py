"""PyTorch HeartTimeMixer for radar heart-rate regression.

This module adapts the TimeMixer PDM/FMM ideas to the exported
Radar_Vital_Signs training windows:

    x_time: (channels=7, window=256)
    x_freq: (channels=7, rfft_bins=129)

The model is intentionally PyTorch-native because the project container ships
with torch 2.6 + CUDA and no TensorFlow runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DecompMethod = Literal["moving_avg", "dft"]
DownsampleMethod = Literal["avg", "max"]


@dataclass
class HeartTimeMixerConfig:
    time_seq_len: int = 256
    freq_seq_len: int = 129
    enc_in: int = 7
    d_model: int = 64
    d_ff: int = 128
    e_layers: int = 2
    dropout: float = 0.15
    moving_avg: int = 25
    top_k: int = 5
    decomp_method: DecompMethod = "moving_avg"
    down_sampling_window: int = 2
    down_sampling_layers: int = 3
    down_sampling_method: DownsampleMethod = "avg"
    channel_independence: bool = True
    use_frequency_domain: bool = True


class MovingAverageDecomp(nn.Module):
    """Moving-average series decomposition used by TimeMixer."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: B, T, D
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        trend = F.avg_pool1d(
            F.pad(x.transpose(1, 2), (pad_left, pad_right), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)
        season = x - trend
        return season, trend


class DFTSeriesDecomp(nn.Module):
    """DFT decomposition that keeps the strongest temporal frequencies."""

    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = int(top_k)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: B, T, D. Keep top-k frequencies per sample/channel.
        xf = torch.fft.rfft(x, dim=1)
        magnitude = xf.abs()
        if magnitude.size(1) > 0:
            magnitude[:, 0, :] = 0
        k = min(self.top_k, magnitude.size(1))
        if k <= 0:
            season = torch.zeros_like(x)
        else:
            threshold = torch.topk(magnitude, k=k, dim=1).values[:, -1:, :]
            xf = torch.where(magnitude >= threshold, xf, torch.zeros_like(xf))
            season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        trend = x - season
        return season, trend


class ScaleNormalize(nn.Module):
    """Lightweight reversible instance normalization for one scale."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        x = (x - mean) / std
        if self.weight is not None and self.bias is not None:
            x = x * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        return x


class DataEmbedding(nn.Module):
    """Value embedding without positional embedding, matching TimeMixer style."""

    def __init__(self, in_features: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(x))


class MultiScaleProcessor(nn.Module):
    """Create the TimeMixer multi-resolution input pyramid."""

    def __init__(
        self,
        down_sampling_window: int,
        down_sampling_layers: int,
        method: DownsampleMethod = "avg",
    ) -> None:
        super().__init__()
        self.down_sampling_window = int(down_sampling_window)
        self.down_sampling_layers = int(down_sampling_layers)
        self.method = method

    def forward(self, x: Tensor) -> List[Tensor]:
        # x: B, T, C
        out = [x]
        x_c = x.transpose(1, 2)
        for _ in range(self.down_sampling_layers):
            if self.method == "avg":
                x_c = F.avg_pool1d(x_c, kernel_size=self.down_sampling_window)
            elif self.method == "max":
                x_c = F.max_pool1d(x_c, kernel_size=self.down_sampling_window)
            else:
                raise ValueError(f"Unsupported down_sampling_method: {self.method}")
            out.append(x_c.transpose(1, 2))
        return out


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal components."""

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(down_sampling_layers):
            in_len = seq_len // (down_sampling_window**i)
            out_len = seq_len // (down_sampling_window ** (i + 1))
            self.layers.append(nn.Sequential(nn.Linear(in_len, out_len), nn.GELU(), nn.Linear(out_len, out_len)))

    def forward(self, season_list: List[Tensor]) -> List[Tensor]:
        # Input list tensors are B, D, T. Output tensors are B, T, D.
        out_high = season_list[0]
        out_low = season_list[1]
        out = [out_high.transpose(1, 2)]
        for i in range(len(season_list) - 1):
            out_low = out_low + self.layers[i](out_high)
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out.append(out_high.transpose(1, 2))
        return out


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend components."""

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in reversed(range(down_sampling_layers)):
            in_len = seq_len // (down_sampling_window ** (i + 1))
            out_len = seq_len // (down_sampling_window**i)
            self.layers.append(nn.Sequential(nn.Linear(in_len, out_len), nn.GELU(), nn.Linear(out_len, out_len)))

    def forward(self, trend_list: List[Tensor]) -> List[Tensor]:
        trend_rev = list(reversed(trend_list))
        out_low = trend_rev[0]
        out_high = trend_rev[1]
        out = [out_low.transpose(1, 2)]
        for i in range(len(trend_rev) - 1):
            out_high = out_high + self.layers[i](out_low)
            out_low = out_high
            if i + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[i + 2]
            out.append(out_low.transpose(1, 2))
        return list(reversed(out))


class PastDecomposableMixing(nn.Module):
    """TimeMixer PDM block adapted for fixed-length radar features."""

    def __init__(self, cfg: HeartTimeMixerConfig, seq_len: int) -> None:
        super().__init__()
        self.channel_independence = cfg.channel_independence
        self.decomp = (
            MovingAverageDecomp(cfg.moving_avg)
            if cfg.decomp_method == "moving_avg"
            else DFTSeriesDecomp(cfg.top_k)
        )
        self.cross = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
        self.season_mixing = MultiScaleSeasonMixing(seq_len, cfg.down_sampling_window, cfg.down_sampling_layers)
        self.trend_mixing = MultiScaleTrendMixing(seq_len, cfg.down_sampling_window, cfg.down_sampling_layers)
        self.out_cross = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x_list: List[Tensor]) -> List[Tensor]:
        lengths = [x.size(1) for x in x_list]
        season_list: List[Tensor] = []
        trend_list: List[Tensor] = []
        for x in x_list:
            season, trend = self.decomp(x)
            if not self.channel_independence:
                season = self.cross(season)
                trend = self.cross(trend)
            season_list.append(season.transpose(1, 2))
            trend_list.append(trend.transpose(1, 2))

        mixed_season = self.season_mixing(season_list)
        mixed_trend = self.trend_mixing(trend_list)

        out: List[Tensor] = []
        for residual, season, trend, length in zip(x_list, mixed_season, mixed_trend, lengths):
            y = season + trend
            y = residual + self.dropout(self.out_cross(y))
            out.append(self.norm(y[:, :length, :]))
        return out


class DomainTimeMixer(nn.Module):
    """One TimeMixer branch for either time-domain or frequency-domain features."""

    def __init__(self, cfg: HeartTimeMixerConfig, seq_len: int, name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.name = name
        self.processor = MultiScaleProcessor(cfg.down_sampling_window, cfg.down_sampling_layers, cfg.down_sampling_method)
        embed_in = 1 if cfg.channel_independence else cfg.enc_in
        self.embedding = DataEmbedding(embed_in, cfg.d_model, cfg.dropout)
        self.normalizers = nn.ModuleList([ScaleNormalize(cfg.enc_in) for _ in range(cfg.down_sampling_layers + 1)])
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(cfg, seq_len) for _ in range(cfg.e_layers)])
        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(seq_len // (cfg.down_sampling_window**i), 1)
                for i in range(cfg.down_sampling_layers + 1)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(cfg.down_sampling_layers + 1))
        self.channel_logits = nn.Parameter(torch.zeros(cfg.enc_in))
        self.projection = nn.Linear(cfg.d_model, 1)
        self.feature_proj = nn.Linear(cfg.d_model * (cfg.down_sampling_layers + 1), cfg.d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Public input shape: B, C, T. TimeMixer internal shape: B, T, C.
        x = x.transpose(1, 2).contiguous()
        bsz, _, channels = x.shape
        x_scales = self.processor(x)

        enc_out: List[Tensor] = []
        for idx, scale in enumerate(x_scales):
            scale = self.normalizers[idx](scale)
            if self.cfg.channel_independence:
                _, steps, n_channels = scale.shape
                scale = scale.transpose(1, 2).reshape(bsz * n_channels, steps, 1)
            enc_out.append(self.embedding(scale))

        for block in self.pdm_blocks:
            enc_out = block(enc_out)

        scale_preds: List[Tensor] = []
        pooled_features: List[Tensor] = []
        channel_weights = torch.softmax(self.channel_logits[:channels], dim=0)
        for idx, enc in enumerate(enc_out):
            pred = self.predict_layers[idx](enc.transpose(1, 2)).transpose(1, 2)
            pred = self.projection(pred).squeeze(-1).squeeze(-1)
            feat = enc.mean(dim=1)
            if self.cfg.channel_independence:
                pred = pred.view(bsz, channels)
                pred = (pred * channel_weights.view(1, channels)).sum(dim=1)
                feat = feat.view(bsz, channels, -1).mean(dim=1)
            scale_preds.append(pred)
            pooled_features.append(feat)

        scale_weights = torch.softmax(self.scale_logits[: len(scale_preds)], dim=0)
        domain_pred = torch.stack(scale_preds, dim=1).mul(scale_weights.view(1, -1)).sum(dim=1)
        domain_feature = self.feature_proj(torch.cat(pooled_features, dim=-1))
        return domain_pred, domain_feature


class HeartTimeMixer(nn.Module):
    """Two-domain TimeMixer regressor for heart-rate BPM prediction."""

    def __init__(self, cfg: HeartTimeMixerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HeartTimeMixerConfig()
        self.time_branch = DomainTimeMixer(self.cfg, self.cfg.time_seq_len, "time")
        self.freq_branch = (
            DomainTimeMixer(self.cfg, self.cfg.freq_seq_len, "freq") if self.cfg.use_frequency_domain else None
        )
        fusion_in = self.cfg.d_model * (2 if self.cfg.use_frequency_domain else 1)
        self.fusion_gate = nn.Sequential(nn.Linear(fusion_in, self.cfg.d_model), nn.GELU(), nn.Linear(self.cfg.d_model, 2))
        self.residual_head = nn.Sequential(
            nn.Linear(fusion_in, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, 1),
        )

    def forward(self, x_time: Tensor, x_freq: Tensor | None = None) -> Tensor:
        time_pred, time_feat = self.time_branch(x_time)
        if self.freq_branch is None:
            return time_pred
        if x_freq is None:
            raise ValueError("x_freq is required when use_frequency_domain=True")
        freq_pred, freq_feat = self.freq_branch(x_freq)
        features = torch.cat([time_feat, freq_feat], dim=-1)
        weights = torch.softmax(self.fusion_gate(features), dim=-1)
        base = weights[:, 0] * time_pred + weights[:, 1] * freq_pred
        residual = self.residual_head(features).squeeze(-1)
        return base + residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
