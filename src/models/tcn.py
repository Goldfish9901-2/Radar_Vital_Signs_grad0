"""TCN baseline for radar heart-rate regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass
class TCNConfig:
    time_channels: int = 7
    freq_channels: int = 7
    hidden_channels: int = 48
    num_blocks: int = 4
    kernel_size: int = 7
    dropout: float = 0.2
    use_frequency_domain: bool = True


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.downsample(x)


class TCNBranch(nn.Module):
    def __init__(self, in_channels: int, cfg: TCNConfig) -> None:
        super().__init__()
        blocks = []
        channels = in_channels
        for idx in range(cfg.num_blocks):
            blocks.append(
                TemporalBlock(
                    in_channels=channels,
                    out_channels=cfg.hidden_channels,
                    kernel_size=cfg.kernel_size,
                    dilation=2**idx,
                    dropout=cfg.dropout,
                )
            )
            channels = cfg.hidden_channels
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = cfg.hidden_channels * 2

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return torch.cat([self.pool(y).squeeze(-1), self.max_pool(y).squeeze(-1)], dim=-1)


class TCNHeartRateModel(nn.Module):
    """Dual-branch TCN regressor.

    Inputs:
        x_time: B, 7, 256
        x_freq: B, 7, 129
    Output:
        normalized HR scalar, shape B
    """

    def __init__(self, cfg: TCNConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TCNConfig()
        self.time_branch = TCNBranch(self.cfg.time_channels, self.cfg)
        self.freq_branch = TCNBranch(self.cfg.freq_channels, self.cfg) if self.cfg.use_frequency_domain else None
        fusion_dim = self.time_branch.feature_dim * (2 if self.cfg.use_frequency_domain else 1)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, self.cfg.hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_channels * 2, self.cfg.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_channels, 1),
        )

    def forward(self, x_time: Tensor, x_freq: Tensor | None = None) -> Tensor:
        features = [self.time_branch(x_time)]
        if self.freq_branch is not None:
            if x_freq is None:
                raise ValueError("x_freq is required when use_frequency_domain=True")
            features.append(self.freq_branch(x_freq))
        return self.head(torch.cat(features, dim=-1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
