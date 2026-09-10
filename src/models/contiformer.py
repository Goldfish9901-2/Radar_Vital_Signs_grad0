"""ContiFormer (continuous-time Transformer) baseline for radar HR regression.

Completes the *continuous-time* family in the benchmark. Unlike a standard
discrete-position Transformer, ContiFormer operates directly on continuous time
stamps:

  * each token carries a continuous time value ``t`` (a regular grid here, since
    the radar frames are uniformly sampled, but the math is general for
    irregular sampling);
  * the token is augmented with a continuous-time positional embedding
    (sinusoid evaluated at ``t``, generalizing the discrete positional encoding);
  * the self-attention score gets a *relative-time bias* ``b(t_i - t_j)``
    produced by a small MLP, so attention is time-aware rather than purely
    order-aware.

This is the distinguishing "continuous-time modeling" capability. With a fixed
sampling grid it reduces to a time-biased Transformer, but the machinery
supports variable / irregular time gaps without changing the interface.

Input contract (shared with every backbone in this repo):
    x_time: (B, 7, 256)
    x_freq: (B, 7, 129)   # optional, used when use_frequency_domain=True
Output:
    normalized HR scalar, shape (B,)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContiFormerConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    use_frequency_domain: bool = True
    time_dt: float = 1.0   # sampling interval of the time branch (regular grid)
    freq_dt: float = 1.0   # sampling interval of the freq branch (regular grid)


class ContinuousTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding evaluated at *continuous* time values.

    Standard Transformer PE fixes ``t`` to integer positions; here ``t`` is any
    real-valued time, so the same module works for regular or irregular series.
    """

    def __init__(self, d_model: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(max_period) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (..., L) real-valued time stamps
        t = t.unsqueeze(-1)                              # (..., L, 1)
        phase = t * self.div_term                       # (..., L, d_model/2)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (..., L, d_model)
        if emb.shape[-1] != self.d_model:               # odd d_model: pad one col
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return emb


class ContinuousTimeAttention(nn.Module):
    """Multi-head self-attention with a learned relative-time bias."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # relative-time bias: f(t_i - t_j) -> per-head scalar
        self.bias_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, nhead),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)  t: (B, L) continuous time stamps
        b, l, _ = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, nhead, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                # each (B, nhead, L, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nhead, L, L)
        dt = t[:, :, None] - t[:, None, :]              # (B, L, L) relative time
        bias = self.bias_net(dt.unsqueeze(-1))          # (B, L, L, nhead)
        bias = bias.permute(0, 3, 1, 2)                 # (B, nhead, L, L)
        scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                     # (B, nhead, L, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)  # (B, L, D)
        return self.proj(out)


class ContiFormerBlock(nn.Module):
    """Pre-norm Transformer block with continuous-time attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ContinuousTimeAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), t))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ContiFormerBranch(nn.Module):
    """Process one modality ``(B, C, L)`` with continuous-time attention."""

    def __init__(self, in_channels: int, seq_len: int, dt: float, cfg: ContiFormerConfig) -> None:
        super().__init__()
        # seq_len is accepted for signature uniformity with the other branches
        # but NOT baked in: the time grid is generated from the actual input
        # length at forward time, so ContiFormer stays length-flexible.
        self.dt = dt
        self.input_proj = nn.Linear(in_channels, cfg.d_model)
        self.time_embed = ContinuousTimeEmbedding(cfg.d_model)
        self.blocks = nn.ModuleList(
            [ContiFormerBlock(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
             for _ in range(cfg.num_layers)]
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = cfg.d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, _, l = x.shape
        x = x.transpose(1, 2)                            # (B, L, C)
        x = self.input_proj(x)                          # (B, L, d_model)
        t = torch.arange(l, dtype=torch.float32, device=x.device) * self.dt  # (L,)
        t = t.expand(b, -1)                             # (B, L)
        x = x + self.time_embed(t)                      # continuous-time PE
        for blk in self.blocks:
            x = blk(x, t)                               # (B, L, d_model)
        x = x.transpose(1, 2)                           # (B, d_model, L)
        return torch.cat([self.avg(x).squeeze(-1), self.max(x).squeeze(-1)], dim=-1)  # (B, 2*d_model)


class ContiFormerHeartRateModel(nn.Module):
    """Dual-branch ContiFormer regressor (time + optional freq).

    Inputs:
        x_time: (B, 7, 256)
        x_freq: (B, 7, 129)
    Output:
        normalized HR scalar, shape (B,)
    """

    def __init__(self, cfg: ContiFormerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ContiFormerConfig()
        self.time_branch = ContiFormerBranch(self.cfg.time_channels, 256, self.cfg.time_dt, self.cfg)
        self.freq_branch = (
            ContiFormerBranch(self.cfg.freq_channels, 129, self.cfg.freq_dt, self.cfg)
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
