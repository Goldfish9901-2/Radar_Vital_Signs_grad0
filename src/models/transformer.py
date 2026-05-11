"""Small Transformer baseline for radar heart-rate regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    use_frequency_domain: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBranch(nn.Module):
    def __init__(self, in_channels: int, cfg: TransformerConfig, max_len: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=max_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.feature_dim = cfg.d_model

    def forward(self, x: Tensor) -> Tensor:
        # B, C, T -> B, T, C
        x = x.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        return self.norm(x[:, 0])


class TransformerHeartRateModel(nn.Module):
    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        self.time_branch = TransformerBranch(self.cfg.time_channels, self.cfg, max_len=256)
        self.freq_branch = (
            TransformerBranch(self.cfg.freq_channels, self.cfg, max_len=129)
            if self.cfg.use_frequency_domain
            else None
        )
        fusion_dim = self.cfg.d_model * (2 if self.cfg.use_frequency_domain else 1)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, self.cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.dim_feedforward, 1),
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
