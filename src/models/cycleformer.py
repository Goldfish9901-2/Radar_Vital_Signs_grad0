"""CycleFormer for radar heart-rate regression.

The model builds tokens from candidate physiological periods instead of fixed
time patches. Each candidate period folds the window into repeated cycles,
encodes the intra-cycle shape, then lets a Transformer compare cycle tokens
across candidate heart/respiration periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.tcn import TCNBranch, TCNConfig


@dataclass
class CycleFormerConfig:
    time_channels: int = 7
    freq_channels: int = 7
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.15
    sampling_rate_hz: float = 20.0
    use_frequency_domain: bool = True
    use_local_branch: bool = True
    local_hidden_channels: int = 48
    local_num_blocks: int = 4
    local_kernel_size: int = 7
    heart_periods: tuple[int, ...] = (8, 10, 12, 14, 16, 20, 24, 30)
    respiration_periods: tuple[int, ...] = (40, 50, 64, 80, 100, 128)


class PeriodCycleEncoder(nn.Module):
    """Encode one folded candidate period into a single token."""

    def __init__(self, in_channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.shape_encoder = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cycle_pool = nn.AdaptiveAvgPool1d(1)
        self.token_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, period: int) -> Tensor:
        # x: B, C, T. Pad T so every token represents complete cycles.
        batch, channels, length = x.shape
        pad = (period - (length % period)) % period
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")
        cycles = x.reshape(batch, channels, -1, period).transpose(1, 2)
        cycles = cycles.reshape(batch * cycles.size(1), channels, period)
        encoded = self.shape_encoder(cycles)
        encoded = self.cycle_pool(encoded).squeeze(-1)
        encoded = encoded.reshape(batch, -1, encoded.size(-1)).mean(dim=1)
        return self.token_norm(encoded)


class CycleTokenBranch(nn.Module):
    """Convert a sequence into candidate-period tokens and model them jointly."""

    def __init__(self, in_channels: int, cfg: CycleFormerConfig) -> None:
        super().__init__()
        self.periods = tuple(int(p) for p in (*cfg.heart_periods, *cfg.respiration_periods))
        self.period_encoder = PeriodCycleEncoder(in_channels, cfg.d_model, cfg.dropout)
        self.period_embedding = nn.Parameter(torch.zeros(1, len(self.periods), cfg.d_model))
        self.scale_embedding = nn.Parameter(torch.zeros(1, len(self.periods), cfg.d_model))
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
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.feature_dim = cfg.d_model
        self._init_scale_embedding(cfg)

    def _init_scale_embedding(self, cfg: CycleFormerConfig) -> None:
        with torch.no_grad():
            periods = torch.tensor(self.periods, dtype=torch.float32)
            bpm = 60.0 * float(cfg.sampling_rate_hz) / periods
            is_heart = ((bpm >= 40.0) & (bpm <= 180.0)).float()
            # A compact prior: first half of the embedding marks heart-like
            # candidates, second half marks respiration-like candidates.
            midpoint = self.scale_embedding.size(-1) // 2
            self.scale_embedding[:, :, :midpoint] = is_heart.view(1, -1, 1)
            self.scale_embedding[:, :, midpoint:] = (1.0 - is_heart).view(1, -1, 1)

    def forward(self, x: Tensor) -> Tensor:
        tokens = [self.period_encoder(x, period) for period in self.periods]
        token_tensor = torch.stack(tokens, dim=1)
        token_tensor = token_tensor + self.period_embedding + self.scale_embedding
        encoded = self.encoder(token_tensor)
        weights = torch.softmax(self.attn_pool(encoded).squeeze(-1), dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled)


class CycleFormerHeartRateModel(nn.Module):
    """Cycle-token Transformer with local micro-motion branches."""

    def __init__(self, cfg: CycleFormerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CycleFormerConfig()
        self.time_branch = CycleTokenBranch(self.cfg.time_channels, self.cfg)
        local_cfg = TCNConfig(
            time_channels=self.cfg.time_channels,
            freq_channels=self.cfg.freq_channels,
            hidden_channels=self.cfg.local_hidden_channels,
            num_blocks=self.cfg.local_num_blocks,
            kernel_size=self.cfg.local_kernel_size,
            dropout=self.cfg.dropout,
            use_frequency_domain=self.cfg.use_frequency_domain,
        )
        self.time_local_branch = (
            TCNBranch(self.cfg.time_channels, local_cfg)
            if self.cfg.use_local_branch
            else None
        )
        self.freq_branch = (
            CycleTokenBranch(self.cfg.freq_channels, self.cfg)
            if self.cfg.use_frequency_domain
            else None
        )
        self.freq_local_branch = (
            TCNBranch(self.cfg.freq_channels, local_cfg)
            if self.cfg.use_frequency_domain and self.cfg.use_local_branch
            else None
        )
        cycle_domain_dim = self.cfg.d_model
        local_domain_dim = self.cfg.local_hidden_channels * 2
        local_fusion_dim = local_domain_dim * (2 if self.cfg.use_frequency_domain else 1)
        self.local_head = (
            nn.Sequential(
                nn.Linear(local_fusion_dim, self.cfg.local_hidden_channels * 2),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.local_hidden_channels * 2, self.cfg.local_hidden_channels),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.local_hidden_channels, 1),
            )
            if self.cfg.use_local_branch
            else None
        )
        domain_dim = cycle_domain_dim
        if self.cfg.use_local_branch:
            domain_dim += local_domain_dim
        fusion_dim = domain_dim * (2 if self.cfg.use_frequency_domain else 1)
        self.cycle_correction_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.cfg.dim_feedforward * 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.dim_feedforward * 2, self.cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.dim_feedforward, 1),
        )
        self.correction_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x_time: Tensor, x_freq: Tensor | None = None) -> Tensor:
        cycle_features = [self.time_branch(x_time)]
        local_features = []
        if self.time_local_branch is not None:
            local_time = self.time_local_branch(x_time)
            cycle_features.append(local_time)
            local_features.append(local_time)
        if self.freq_branch is not None:
            if x_freq is None:
                raise ValueError("x_freq is required when use_frequency_domain=True")
            cycle_features.append(self.freq_branch(x_freq))
            if self.freq_local_branch is not None:
                local_freq = self.freq_local_branch(x_freq)
                cycle_features.append(local_freq)
                local_features.append(local_freq)

        correction = self.cycle_correction_head(torch.cat(cycle_features, dim=-1)).squeeze(-1)
        if self.local_head is None:
            return correction
        local_pred = self.local_head(torch.cat(local_features, dim=-1)).squeeze(-1)
        return local_pred + torch.sigmoid(self.correction_logit) * correction
