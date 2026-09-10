"""PatchTST-style baseline for radar heart-rate regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class PatchTSTConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.15
    patch_len: int = 16
    patch_stride: int = 8
    use_frequency_domain: bool = True


class PatchTSTBranch(nn.Module):
    def __init__(self, in_channels: int, cfg: PatchTSTConfig, max_patches: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.patch_proj = nn.Linear(in_channels * cfg.patch_len, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches + 1, cfg.d_model))
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
        # B, C, T -> B, N, C*patch_len
        if x.size(-1) < self.cfg.patch_len:
            x = F.pad(x, (0, self.cfg.patch_len - x.size(-1)), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.patch_stride)
        num_patches = patches.size(2)
        patches = patches.permute(0, 2, 1, 3).reshape(x.size(0), num_patches, -1)
        tokens = self.patch_proj(patches)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        pos = self.pos_embedding[:, : tokens.size(1), :]
        encoded = self.encoder(tokens + pos)
        return self.norm(encoded[:, 0])


class PatchTSTHeartRateModel(nn.Module):
    def __init__(self, cfg: PatchTSTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PatchTSTConfig()
        self.time_branch = PatchTSTBranch(self.cfg.time_channels, self.cfg)
        self.freq_branch = (
            PatchTSTBranch(self.cfg.freq_channels, self.cfg)
            if self.cfg.use_frequency_domain
            else None
        )
        fusion_dim = self.cfg.d_model * (2 if self.cfg.use_frequency_domain else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
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
