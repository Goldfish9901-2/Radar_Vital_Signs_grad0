"""Mamba (State-Space) baseline for radar heart-rate regression.

Dependency-free selective-scan SSM following the canonical minimal Mamba
formulation (Gu & Dao, 2023), so it stays portable on the 4 GB laptop GPU
without the CUDA ``mamba-ssm`` / ``causal-conv1d`` kernels. The block math is
the standard minimal version; if the optimized kernel is later installed it
can replace :class:`MambaBlock` internals without changing the model interface.

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
class MambaConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    num_layers: int = 2
    d_state: int = 16
    dt_rank: int = 16
    expand: int = 2
    dropout: float = 0.2
    use_frequency_domain: bool = True


class MambaBlock(nn.Module):
    """Single selective-scan SSM block. Input / output: ``(B, L, D)``."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner * 2, self.d_inner * 2, d_conv,
            padding=d_conv - 1, groups=self.d_inner * 2, bias=False,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=False)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        b, l, _ = x.shape
        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        xz = xz.transpose(1, 2)                    # (B, 2*d_inner, L)
        xz = self.conv1d(xz)[..., :l]              # depthwise causal conv, trim
        xz = xz.transpose(1, 2)                    # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        dbl = self.x_proj(x)                       # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)                      # (B, L, d_inner)
        dt = F.softplus(dt)
        A = -torch.exp(self.A_log)                 # (d_inner, d_state)
        D = self.D                                 # (d_inner,)

        deltaA = torch.exp(dt[:, :, :, None] * A)  # (B, L, d_inner, d_state)
        deltaBx = deltaA * (x[:, :, :, None] * B[:, :, None, :])  # (B, L, d_inner, d_state)

        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaBx[:, i]
            ys.append((h * C[:, i, None, :]).sum(-1))  # (B, d_inner)
        y = torch.stack(ys, dim=1)                 # (B, L, d_inner)
        y = y + x * D
        y = self.out_proj(y * F.silu(z))           # (B, L, d_model)
        return y


class MambaBranch(nn.Module):
    """Process one modality ``(B, C, L)``.

    The ``C`` radar bins are projected to ``d_model`` (mixing channels), then a
    stack of :class:`MambaBlock` scans over time; the sequence is pooled to a
    fixed feature vector. This mirrors how TCN/CycleFormer treat the 7 channels
    as feature channels, keeping the benchmark comparison fair.
    """

    def __init__(self, in_channels: int, cfg: MambaConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, cfg.d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(cfg.d_model, cfg.d_state, cfg.dt_rank, expand=cfg.expand)
             for _ in range(cfg.num_layers)]
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = cfg.d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = x.transpose(1, 2)                      # (B, L, C)
        x = self.input_proj(x)                     # (B, L, d_model)
        for blk in self.blocks:
            x = blk(x)                             # (B, L, d_model)
        x = x.transpose(1, 2)                      # (B, d_model, L)
        return torch.cat([self.avg(x).squeeze(-1), self.max(x).squeeze(-1)], dim=-1)  # (B, 2*d_model)


class MambaHeartRateModel(nn.Module):
    """Dual-branch Mamba regressor (time + optional freq), mirroring TCN layout.

    Inputs:
        x_time: (B, 7, 256)
        x_freq: (B, 7, 129)
    Output:
        normalized HR scalar, shape (B,)
    """

    def __init__(self, cfg: MambaConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MambaConfig()
        self.time_branch = MambaBranch(self.cfg.time_channels, self.cfg)
        self.freq_branch = (
            MambaBranch(self.cfg.freq_channels, self.cfg)
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
