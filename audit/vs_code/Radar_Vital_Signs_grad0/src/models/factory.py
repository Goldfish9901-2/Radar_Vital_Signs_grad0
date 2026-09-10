"""Model factory for heart-rate regression backbones.

Backbone choice is a comparison dimension, not part of the proposed signal
processing method. Centralizing construction keeps train/evaluate/adapt scripts
consistent when adding or changing a model.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

from torch import nn

from src.models import (
    ContiFormerConfig,
    ContiFormerHeartRateModel,
    CycleFormerConfig,
    CycleFormerHeartRateModel,
    HeartTimeMixer,
    HeartTimeMixerConfig,
    LinearBaselineConfig,
    MambaConfig,
    MambaHeartRateModel,
    PatchTSTConfig,
    PatchTSTHeartRateModel,
    TCNConfig,
    TCNHeartRateModel,
    TimesNetConfig,
    TimesNetHeartRateModel,
    TransformerConfig,
    TSMixerConfig,
    TSMixerHeartRateModel,
    TransformerHeartRateModel,
    TSLANetConfig,
    TSLANetHeartRateModel,
    XLSTMConfig,
    XLSTMHeartRateModel,
    FreTSConfig,
    FreTSHeartRateModel,
    DLinearHeartRateModel,
    NLinearHeartRateModel,
)

MODEL_CHOICES = (
    "contiformer",
    "cycleformer",
    "dlinear",
    "heart_timemixer",
    "mamba",
    "nlinear",
    "patchtst",
    "tcn",
    "timesnet",
    "transformer",
    "tslanet",
    "tsmixer",
    "xlstm",
    "frets",
)
ModelConfig = (
    ContiFormerConfig
    | CycleFormerConfig
    | HeartTimeMixerConfig
    | LinearBaselineConfig
    | MambaConfig
    | PatchTSTConfig
    | TCNConfig
    | TimesNetConfig
    |     TransformerConfig
    | TSLANetConfig
    | TSMixerConfig
    | XLSTMConfig
    | FreTSConfig
)


def _apply_channels(cfg: ModelConfig, args: Any) -> ModelConfig:
    """Override a config's input channel counts from CLI / export-derived args.

    Every backbone reads ``time_channels``/``freq_channels`` to size its input
    projection, but those defaults are 7 (matching the ``proposed`` 7-mode VMD
    representation). When a representation produces a different channel count
    (e.g. ``raw_logmag`` -> 1, ``raw_real_imag`` -> 2), the model must be built
    to match, otherwise the tensor shapes mismatch at the first linear layer.
    """
    if hasattr(cfg, "time_channels"):
        cfg.time_channels = args.time_channels
    if hasattr(cfg, "freq_channels"):
        cfg.freq_channels = args.freq_channels
    return cfg


def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """Instantiate a model from a serialized config dictionary."""
    if model_name == "contiformer":
        return ContiFormerHeartRateModel(ContiFormerConfig(**config))
    if model_name == "cycleformer":
        return CycleFormerHeartRateModel(CycleFormerConfig(**config))
    if model_name == "dlinear":
        return DLinearHeartRateModel(LinearBaselineConfig(**config))
    if model_name == "nlinear":
        return NLinearHeartRateModel(LinearBaselineConfig(**config))
    if model_name == "heart_timemixer":
        return HeartTimeMixer(HeartTimeMixerConfig(**config))
    if model_name == "patchtst":
        return PatchTSTHeartRateModel(PatchTSTConfig(**config))
    if model_name == "tcn":
        return TCNHeartRateModel(TCNConfig(**config))
    if model_name == "timesnet":
        return TimesNetHeartRateModel(TimesNetConfig(**config))
    if model_name == "transformer":
        return TransformerHeartRateModel(TransformerConfig(**config))
    if model_name == "tslanet":
        return TSLANetHeartRateModel(TSLANetConfig(**config))
    if model_name == "tsmixer":
        return TSMixerHeartRateModel(TSMixerConfig(**config))
    if model_name == "mamba":
        return MambaHeartRateModel(MambaConfig(**config))
    if model_name == "xlstm":
        return XLSTMHeartRateModel(XLSTMConfig(**config))
    if model_name == "frets":
        return FreTSHeartRateModel(FreTSConfig(**config))
    raise ValueError(f"Unsupported model: {model_name}")


def create_model_and_config(args: Any) -> Tuple[nn.Module, ModelConfig]:
    """Build a model and config from training CLI arguments."""
    if args.model == "contiformer":
        cfg = ContiFormerConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return ContiFormerHeartRateModel(cfg), cfg
    if args.model == "dlinear":
        cfg = LinearBaselineConfig(
            variant="dlinear",
            moving_avg=getattr(args, "moving_avg", 25),
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return DLinearHeartRateModel(cfg), cfg
    if args.model == "nlinear":
        cfg = LinearBaselineConfig(
            variant="nlinear",
            moving_avg=getattr(args, "moving_avg", 25),
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return NLinearHeartRateModel(cfg), cfg
    if args.model == "cycleformer":
        heart_periods = (
            args.heart_periods
            if args.heart_periods is not None
            else CycleFormerConfig.heart_periods
        )
        respiration_periods = (
            args.respiration_periods
            if args.respiration_periods is not None
            else CycleFormerConfig.respiration_periods
        )
        cfg = CycleFormerConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
            local_hidden_channels=args.hidden_channels,
            local_num_blocks=args.num_blocks,
            local_kernel_size=args.kernel_size,
            heart_periods=tuple(heart_periods),
            respiration_periods=tuple(respiration_periods),
        )
        cfg = _apply_channels(cfg, args)
        return CycleFormerHeartRateModel(cfg), cfg
    if args.model == "heart_timemixer":
        cfg = HeartTimeMixerConfig(
            d_model=args.d_model,
            d_ff=args.d_ff,
            e_layers=args.e_layers,
            dropout=args.dropout,
            decomp_method=args.decomp_method,
            top_k=args.top_k,
            moving_avg=args.moving_avg,
            down_sampling_layers=args.down_sampling_layers,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return HeartTimeMixer(cfg), cfg
    if args.model == "tcn":
        cfg = TCNConfig(
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return TCNHeartRateModel(cfg), cfg
    if args.model == "patchtst":
        cfg = PatchTSTConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return PatchTSTHeartRateModel(cfg), cfg
    if args.model == "timesnet":
        cfg = TimesNetConfig(
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return TimesNetHeartRateModel(cfg), cfg
    if args.model == "transformer":
        cfg = TransformerConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return TransformerHeartRateModel(cfg), cfg
    if args.model == "tslanet":
        # Phase 2a: TSLANet hyperparameters are now CLI-overridable. The frequency
        # branch is intentionally NOT enabled here (R8 is deferred); we keep
        # use_frequency_domain=False regardless of --time-only so we never silently
        # exercise the semantically-questionable spectrum-through-ASB path.
        cfg = TSLANetConfig(
            emb_dim=args.emb_dim,
            depth=args.tslanet_depth,
            patch_len=args.tslanet_patch_len,
            patch_stride=args.tslanet_patch_stride,
            dropout=args.tslanet_dropout,
            use_asb=args.use_asb,
            use_icb=args.use_icb,
            adaptive_filter=args.adaptive_filter,
            normalize=args.normalize,
            channel_mode=args.channel_mode,
            use_frequency_domain=False,
        )
        cfg = _apply_channels(cfg, args)
        return TSLANetHeartRateModel(cfg), cfg
    if args.model == "mamba":
        # Pure-PyTorch selective-scan SSM. Reuses the shared CLI args (d_model,
        # num_layers, dropout, --time-only) so train_model.py needs no change.
        cfg = MambaConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return MambaHeartRateModel(cfg), cfg
    if args.model == "tsmixer":
        # All-MLP mixers. Reuses the shared CLI args (d_model, num_layers,
        # dropout, --time-only) so train_model.py needs no change.
        cfg = TSMixerConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return TSMixerHeartRateModel(cfg), cfg
    if args.model == "xlstm":
        # xLSTM (sLSTM + mLSTM). Reuses the shared CLI args (d_model, num_layers,
        # d_ff, dropout, --time-only) so train_model.py needs no change.
        cfg = XLSTMConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return XLSTMHeartRateModel(cfg), cfg
    if args.model == "frets":
        # FreTS (frequency-domain MLP). Reuses the shared CLI args (d_model,
        # num_layers, dropout, --time-only) so train_model.py needs no change.
        cfg = FreTSConfig(
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        cfg = _apply_channels(cfg, args)
        return FreTSHeartRateModel(cfg), cfg
    raise ValueError(f"Unsupported model: {args.model}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for reporting experiment metadata."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def config_to_dict(config: ModelConfig) -> Dict[str, Any]:
    """Serialize a model config dataclass."""
    return asdict(config)
