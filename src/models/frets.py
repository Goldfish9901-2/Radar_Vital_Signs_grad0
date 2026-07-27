"""FreTS (Frequency-domain MLP) backbone for radar HR regression.

From-scratch implementation of the frequency-domain MLP from Zhou et al. (2022),
"FiLM: Frequency improved Legendre Memory Model for Long-term Time Series
Forecasting" / FreTS: each channel's signal is transformed to the frequency
domain (FFT), its amplitude and phase are each passed through a 2-layer MLP over
the frequency axis, then inverse-FFT reconstructs a refined time-domain signal
that is pooled to a fixed feature vector. This keeps the model permutation-
equivariant over time and explicitly models inter-frequency dependencies.

Dependency-free (pure torch.fft), so it runs on the 4 GB laptop GPU.

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


@dataclass
class FreTSConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    use_frequency_domain: bool = True
    seq_len: int = 256
    freq_len: int = 129


class FreqMLP(nn.Module):
    """2-layer MLP applied over the frequency axis of a (B, C, F) tensor."""

    def __init__(self, freq_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, freq_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FreTSBranch(nn.Module):
    """Frequency-domain MLP over one modality (B, C, L)."""

    def __init__(self, in_channels: int, cfg: FreTSConfig, seq_len: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.freq_dim = seq_len // 2 + 1
        self.amp_mlp = FreqMLP(self.freq_dim, cfg.d_model)
        self.phase_mlp = FreqMLP(self.freq_dim, cfg.d_model)
        self.input_proj = nn.Linear(in_channels, cfg.d_model)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = cfg.d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        xf = torch.fft.rfft(x, dim=-1)                       # (B, C, F), F = L//2 + 1
        amp = xf.abs()
        phase = xf.angle()
        amp2 = self.amp_mlp(amp)
        phase2 = self.phase_mlp(phase)
        recon = torch.complex(
            amp2 * torch.cos(phase2),
            amp2 * torch.sin(phase2),
        )
        x_rec = torch.fft.irfft(recon, n=x.shape[-1], dim=-1)  # (B, C, L)
        x_rec = x_rec.transpose(1, 2)                         # (B, L, C)
        x_rec = self.input_proj(x_rec)                       # (B, L, d_model)
        x_rec = x_rec.transpose(1, 2)                        # (B, d_model, L)
        return torch.cat([self.avg(x_rec).squeeze(-1), self.max(x_rec).squeeze(-1)], dim=-1)  # (B, 2*d_model)


class FreTSHeartRateModel(nn.Module):
    """Dual-branch FreTS regressor (time + optional freq)."""

    def __init__(self, cfg: FreTSConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FreTSConfig()
        self.time_branch = FreTSBranch(self.cfg.time_channels, self.cfg, self.cfg.seq_len)
        self.freq_branch = (
            FreTSBranch(self.cfg.freq_channels, self.cfg, self.cfg.freq_len)
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
