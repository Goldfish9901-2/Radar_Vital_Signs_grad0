"""TimesNet-style baseline for radar heart-rate regression.

This lightweight implementation folds the sequence by several candidate
periods and applies 2D convolutions to model intra-period and inter-period
variation, matching the TimesNet idea without adding external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TimesNetConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    d_ff: int = 128
    num_blocks: int = 3
    dropout: float = 0.15
    use_frequency_domain: bool = True
    periods: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64)


class TimesBlock(nn.Module):
    def __init__(self, channels: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, d_ff, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_ff, d_ff, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_ff, channels, kernel_size=1),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))


class Period2DBranch(nn.Module):
    def __init__(self, in_channels: int, cfg: TimesNetConfig) -> None:
        super().__init__()
        self.periods = tuple(int(p) for p in cfg.periods)
        self.input_proj = nn.Conv1d(in_channels, cfg.d_model, kernel_size=1)
        self.blocks = nn.ModuleList(
            [nn.Sequential(*[TimesBlock(cfg.d_model, cfg.d_ff, cfg.dropout) for _ in range(cfg.num_blocks)]) for _ in self.periods]
        )
        self.period_score = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.feature_dim = cfg.d_model

    def _fold(self, x: Tensor, period: int) -> Tensor:
        batch, channels, length = x.shape
        pad = (period - (length % period)) % period
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")
        return x.reshape(batch, channels, -1, period)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        features = []
        for period, block in zip(self.periods, self.blocks):
            folded = self._fold(x, period)
            encoded = block(folded)
            features.append(encoded.mean(dim=(-1, -2)))
        period_features = torch.stack(features, dim=1)
        weights = torch.softmax(self.period_score(period_features).squeeze(-1), dim=-1)
        pooled = torch.sum(period_features * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled)


class TimesNetHeartRateModel(nn.Module):
    def __init__(self, cfg: TimesNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TimesNetConfig()
        self.time_branch = Period2DBranch(self.cfg.time_channels, self.cfg)
        self.freq_branch = (
            Period2DBranch(self.cfg.freq_channels, self.cfg)
            if self.cfg.use_frequency_domain
            else None
        )
        fusion_dim = self.cfg.d_model * (2 if self.cfg.use_frequency_domain else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.cfg.d_ff),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_ff, 1),
        )

    def forward(self, x_time: Tensor, x_freq: Tensor | None = None) -> Tensor:
        features = [self.time_branch(x_time)]
        if self.freq_branch is not None:
            if x_freq is None:
                raise ValueError("x_freq is required when use_frequency_domain=True")
            features.append(self.freq_branch(x_freq))
        return self.head(torch.cat(features, dim=-1)).squeeze(-1)
