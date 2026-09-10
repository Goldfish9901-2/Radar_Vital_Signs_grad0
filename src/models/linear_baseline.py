"""DLinear / NLinear linear baselines for radar heart-rate regression.

Implements the two strong, minimal linear baselines from Zeng et al. (2023),
"Are Transformers Effective for Time Series Forecasting?". They are deliberately
*not* another Transformer — they are cheap, transparent floors for the
benchmark, and they expose whether the heavier backbones beat a linear map.

  * **DLinear**: decomposes each radar-channel series into a trend (moving
    average) and a seasonal (residual) component, then applies a separate
    linear map over time to each, and concatenates the two per-channel scalars.
  * **NLinear**: subtracts the last time step from the whole series (a simple
    level normalization), applies a single linear map over time, then adds the
    last step back. Robust to distribution shift between train/test.

Input contract (shared with every backbone in this repo):
    x_time: (B, 7, 256)
    x_freq: (B, 7, 129)   # optional, used when use_frequency_domain=True
Output:
    normalized HR scalar, shape (B,)

Note (like TSMixer): the time-axis linear weights are tied to the sequence
length (256 / 129), so these models are fixed-length. The benchmark always
feeds fixed-length tensors, so this is fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LinearBaselineConfig:
    variant: str = "dlinear"   # "dlinear" or "nlinear"
    time_channels: int = 7
    freq_channels: int = 7
    moving_avg: int = 25       # trend window for DLinear (odd-ish; floor-div used)
    dropout: float = 0.1
    use_frequency_domain: bool = True


class LinearBaselineBranch(nn.Module):
    """Process one modality ``(B, C, L)`` with a linear map over time.

    ``variant`` selects DLinear (trend+seasonal) or NLinear (level-norm).
    """

    def __init__(self, in_channels: int, seq_len: int, cfg: LinearBaselineConfig) -> None:
        super().__init__()
        self.variant = cfg.variant
        self.seq_len = seq_len
        if cfg.variant == "dlinear":
            self.decomp = nn.AvgPool1d(cfg.moving_avg, stride=1, padding=cfg.moving_avg // 2)
            self.trend_linear = nn.Linear(seq_len, 1, bias=False)
            self.seasonal_linear = nn.Linear(seq_len, 1, bias=False)
            self.feature_dim = in_channels * 2
        elif cfg.variant == "nlinear":
            self.linear = nn.Linear(seq_len, 1, bias=False)
            self.feature_dim = in_channels
        else:
            raise ValueError(f"Unknown linear baseline variant: {cfg.variant}")
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, l = x.shape
        flat = x.reshape(b * c, l)                       # (B*C, L)
        if self.variant == "dlinear":
            trend = self.decomp(flat.unsqueeze(1)).squeeze(1).reshape(b, c, l)  # (B, C, L)
            seasonal = x - trend
            tf = self.trend_linear(trend.reshape(b * c, l)).reshape(b, c)       # (B, C)
            sf = self.seasonal_linear(seasonal.reshape(b * c, l)).reshape(b, c)  # (B, C)
            feat = torch.cat([tf, sf], dim=-1)            # (B, 2C)
        else:  # nlinear
            last = x[..., -1:]                            # (B, C, 1)
            xn = (x - last).reshape(b * c, l)             # level-normalize
            o = self.linear(xn).reshape(b, c)             # (B, C)
            o = o + last.squeeze(-1)                      # add level back
            feat = o                                      # (B, C)
        return self.drop(feat)


class _LinearBaselineHeartRateModel(nn.Module):
    """Shared regressor for DLinear / NLinear (dual time + optional freq branch).

    Inputs:
        x_time: (B, 7, 256)
        x_freq: (B, 7, 129)
    Output:
        normalized HR scalar, shape (B,)
    """

    def __init__(self, cfg: LinearBaselineConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LinearBaselineConfig()
        self.time_branch = LinearBaselineBranch(self.cfg.time_channels, 256, self.cfg)
        self.freq_branch = (
            LinearBaselineBranch(self.cfg.freq_channels, 129, self.cfg)
            if self.cfg.use_frequency_domain else None
        )
        fusion_dim = self.time_branch.feature_dim * (2 if self.cfg.use_frequency_domain else 1)
        # Self-contained head (no d_model needed): dim grows from fused features.
        mid = max(fusion_dim * 2, 32)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, mid),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(mid, fusion_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor | None = None) -> torch.Tensor:
        features = [self.time_branch(x_time)]
        if self.freq_branch is not None:
            if x_freq is None:
                raise ValueError("x_freq is required when use_frequency_domain=True")
            features.append(self.freq_branch(x_freq))
        return self.head(torch.cat(features, dim=-1)).squeeze(-1)


class DLinearHeartRateModel(_LinearBaselineHeartRateModel):
    """DLinear variant (trend + seasonal linear)."""

    def __init__(self, cfg: LinearBaselineConfig | None = None) -> None:
        c = cfg or LinearBaselineConfig()
        if c.variant != "dlinear":
            c = LinearBaselineConfig(
                variant="dlinear", time_channels=c.time_channels, freq_channels=c.freq_channels,
                moving_avg=c.moving_avg, dropout=c.dropout,
                use_frequency_domain=c.use_frequency_domain,
            )
        super().__init__(c)


class NLinearHeartRateModel(_LinearBaselineHeartRateModel):
    """NLinear variant (level-normalized linear)."""

    def __init__(self, cfg: LinearBaselineConfig | None = None) -> None:
        c = cfg or LinearBaselineConfig()
        if c.variant != "nlinear":
            c = LinearBaselineConfig(
                variant="nlinear", time_channels=c.time_channels, freq_channels=c.freq_channels,
                moving_avg=c.moving_avg, dropout=c.dropout,
                use_frequency_domain=c.use_frequency_domain,
            )
        super().__init__(c)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
