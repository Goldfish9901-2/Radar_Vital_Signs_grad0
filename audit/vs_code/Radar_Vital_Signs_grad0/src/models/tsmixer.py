"""TSMixer (all-MLP) baseline for radar heart-rate regression.

Completes the MLP-only family in the benchmark (Ekambaram et al., 2023).
No attention, no convolution, no recurrence: a stack of TSMixer blocks that
mix across the *time* axis and across the *feature* (radar-channel) axis with
plain MLPs. This is deliberately simple and cheap to maintain — the point is a
strong, low-complexity baseline, not a novel architecture.

Input contract (shared with every backbone in this repo):
    x_time: (B, 7, 256)
    x_freq: (B, 7, 129)   # optional, used when use_frequency_domain=True
Output:
    normalized HR scalar, shape (B,)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TSMixerConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    use_frequency_domain: bool = True


class TSMixerBlock(nn.Module):
    """One TSMixer block: T-mixing then F-mixing, each pre-norm + residual.

    Input / output: ``(B, L, D)`` where L is the time length of this modality
    (256 for time, 129 for freq) and D is the feature dim (= d_model).
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(d_model)
        self.t_mlp = nn.Linear(seq_len, seq_len)          # mixes across time, per channel
        self.norm_f = nn.LayerNorm(d_model)
        self.f_mlp = nn.Linear(d_model, d_model)          # mixes across channels, per timestep
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.norm_t(x).transpose(1, 2)                # (B, D, L)
        h = self.t_mlp(h)                                 # (B, D, L)
        h = F.gelu(h)
        h = self.drop(h).transpose(1, 2)                  # (B, L, D)
        x = x + h

        h = self.norm_f(x)                                # (B, L, D)
        h = self.f_mlp(h)                                 # (B, L, D)
        h = F.gelu(h)
        h = self.drop(h)
        x = x + h
        return x


class TSMixerBranch(nn.Module):
    """Process one modality ``(B, C, L)``.

    The ``C`` radar bins are linearly projected to ``d_model`` (a feature-mix on
    the raw channels), then a stack of :class:`TSMixerBlock` mixes time and
    features; the sequence is pooled to a fixed feature vector. This mirrors how
    TCN/Mamba treat the 7 channels as feature channels, keeping the benchmark
    comparison fair.
    """

    def __init__(self, in_channels: int, seq_len: int, cfg: TSMixerConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, cfg.d_model)
        # T-mixing is a per-channel MLP over the time axis, so its weights are
        # tied to ``seq_len``: TSMixer is fixed-length (like the published model)
        # and is not variable-length like TCN/Mamba. The benchmark always feeds
        # 256 (time) / 129 (freq), so this is fine.
        self.blocks = nn.ModuleList(
            [TSMixerBlock(cfg.d_model, seq_len, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = cfg.d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = x.transpose(1, 2)                             # (B, L, C)
        x = self.input_proj(x)                           # (B, L, d_model)
        for blk in self.blocks:
            x = blk(x)                                   # (B, L, d_model)
        x = x.transpose(1, 2)                            # (B, d_model, L)
        return torch.cat([self.avg(x).squeeze(-1), self.max(x).squeeze(-1)], dim=-1)  # (B, 2*d_model)


class TSMixerHeartRateModel(nn.Module):
    """Dual-branch TSMixer regressor (time + optional freq).

    Inputs:
        x_time: (B, 7, 256)
        x_freq: (B, 7, 129)
    Output:
        normalized HR scalar, shape (B,)
    """

    def __init__(self, cfg: TSMixerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TSMixerConfig()
        self.time_branch = TSMixerBranch(self.cfg.time_channels, 256, self.cfg)
        self.freq_branch = (
            TSMixerBranch(self.cfg.freq_channels, 129, self.cfg)
            if self.cfg.use_frequency_domain else None
        )
        fusion_dim = self.time_branch.feature_dim * (2 if self.cfg.use_frequency_domain else 1)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, self.cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model * 2, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, 1),
        )

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor | None = None) -> torch.Tensor:
        features = [self.time_branch(x_time)]
        if self.freq_branch is not None:
            if x_freq is None:
                raise ValueError("x_freq is required when use_frequency_domain=True")
            features.append(self.freq_branch(x_freq))
        return self.head(torch.cat(features, dim=-1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
