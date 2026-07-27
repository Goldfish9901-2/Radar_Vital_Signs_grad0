"""xLSTM (Extended Long Short-Term Memory) backbone for radar HR regression.

From-scratch, dependency-free implementation of the xLSTM building blocks
(Beck et al., 2024, "xLSTM: Extended Long Short-Term Memory") so it runs on the
4 GB laptop GPU without the CUDA ``xlstm`` package:

  * :class:`sLSTMCell`  - scalar LSTM with exponential gating + scalar stabilizer
                         state + (per-unit) memory / normalizer states.
  * :class:`mLSTMCell`  - matrix memory LSTM with the covariance update rule and
                         normalized query / key (the main storage mechanism).
  * :class:`XLSTMLayer` - Pre-LayerNorm recurrent cell + gated MLP residual.
  * :class:`XLSTMBlock` - stack of layers whose cell type is chosen per layer from
                         ``block_types`` (cycled), so a single config can mix
                         sLSTM (recall) and mLSTM (storage).
  * :class:`XLSTMBranch` / :class:`XLSTMHeartRateModel` - dual time+freq regressor
    mirroring the Mamba / TCN layout.

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
class XLSTMConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    num_layers: int = 2
    # Per-layer cell type, cycled. "m" = mLSTM (matrix memory), "s" = sLSTM.
    block_types: tuple = ("m", "s")
    d_ff: int = 128
    dropout: float = 0.2
    use_frequency_domain: bool = True


class sLSTMCell(nn.Module):
    """sLSTM cell with exponential gating (input: (B, L, D) -> (B, L, D))."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # gate projection: z (cell input, unbounded), i/f/o gates (sigmoid).
        self.proj = nn.Linear(d_model, 4 * d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        gates = self.proj(x)  # (B, L, 4D)
        z, i, f, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        m = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        c = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        n = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        out = []
        for t in range(l):
            zt, it, ft, ot = z[:, t], i[:, t], f[:, t], o[:, t]
            # scalar stabilizer state: elementwise max of candidate terms
            m_new = torch.maximum(ft * m, it * zt)
            i_hat = torch.exp(it * zt - m_new)
            f_hat = torch.exp(ft * m - m_new)
            c = f_hat * c + i_hat * zt
            n = f_hat * n + i_hat
            h = ot * (c / (n + 1e-5))
            out.append(h)
            m = m_new
        return torch.stack(out, dim=1)


class mLSTMCell(nn.Module):
    """mLSTM cell with matrix memory (input: (B, L, D) -> (B, L, D)).

    q and k are L2-normalized per step (a lightweight, dependency-free stand-in
    for the learned q/k norm of the reference implementation) so the covariance
    update stays bounded without the optimized CUDA kernel.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, 4 * d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        gates = self.proj(x)  # (B, L, 4D)
        q, k, v, f = gates.chunk(4, dim=-1)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-5)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
        f = torch.sigmoid(f)

        C = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        n = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        out = []
        for t in range(l):
            qt, kt, vt, ft = q[:, t], k[:, t], v[:, t], f[:, t]
            C = ft.unsqueeze(-1) * C + vt.unsqueeze(-1) * kt.unsqueeze(-2)  # (B, D, D)
            n = ft * n + kt
            h = torch.bmm(C, qt.unsqueeze(-1)).squeeze(-1)          # (B, D)
            h = h / ((n * qt).sum(dim=-1, keepdim=True) + 1e-5)
            out.append(h)
        return torch.stack(out, dim=1)


class XLSTMLayer(nn.Module):
    """Pre-LayerNorm recurrent cell followed by a gated MLP residual."""

    def __init__(self, cell: nn.Module, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cell = cell
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.cell(self.norm1(x))
        h = h + self.mlp(self.norm2(h))
        return h


class XLSTMBlock(nn.Module):
    """Stack of xLSTM layers; cell type per layer from ``block_types`` (cycled)."""

    def __init__(self, d_model: int, d_ff: int, block_types: tuple, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            kind = block_types[i % len(block_types)]
            cell = sLSTMCell(d_model) if kind == "s" else mLSTMCell(d_model)
            self.layers.append(XLSTMLayer(cell, d_model, d_ff, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class XLSTMBranch(nn.Module):
    """Process one modality (B, C, L): project channels -> d_model, run the xLSTM
    stack, pool (avg+max) to a fixed feature vector. Mirrors MambaBranch."""

    def __init__(self, in_channels: int, cfg: XLSTMConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, cfg.d_model)
        self.block = XLSTMBlock(cfg.d_model, cfg.d_ff, cfg.block_types, cfg.num_layers, cfg.dropout)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.feature_dim = cfg.d_model * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                      # (B, L, C)
        x = self.input_proj(x)                     # (B, L, d_model)
        x = self.block(x)                          # (B, L, d_model)
        x = x.transpose(1, 2)                      # (B, d_model, L)
        return torch.cat([self.avg(x).squeeze(-1), self.max(x).squeeze(-1)], dim=-1)  # (B, 2*d_model)


class XLSTMHeartRateModel(nn.Module):
    """Dual-branch xLSTM regressor (time + optional freq)."""

    def __init__(self, cfg: XLSTMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or XLSTMConfig()
        self.time_branch = XLSTMBranch(self.cfg.time_channels, self.cfg)
        self.freq_branch = (
            XLSTMBranch(self.cfg.freq_channels, self.cfg)
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
