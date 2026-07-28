"""Feature representation assembly for training windows.

This module is the single routing point for converting a complex RDA window into
model-ready tensors. The proposed path is intentionally explicit:
EDACM phase representation -> HR-AdaVMD -> time/frequency features. Simpler
representations are retained here as baselines for comparison experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.features.base import RadarRepresentation
from src.features.edacm import (
    select_target_bin_indices,
    target_edacm_signal,
    zscore_1d,
)
from src.features.hr_adavmd import (
    DEFAULT_VMD_ALPHA,
    DEFAULT_VMD_K,
    DEFAULT_VMD_MAX_ITER,
    DEFAULT_VMD_TOL,
    hr_adavmd_decompose,
    hr_adavmd_initial_omega,
    vmd_decompose,
)

DEFAULT_SAMPLING_RATE_HZ = 20.0
DEFAULT_RDA_REPRESENTATION = "log_magnitude"
# Bumped when the representation generator logic changes in a way that affects
# what is written to disk. Recorded in build_config.json / meta for exact
# reproducibility of every experiment.
REPRESENTATION_GENERATOR_VERSION = "1.0"
METHOD_PIPELINE = (
    "radar_rda_features",
    "edacm_phase_representation",
    "hr_adavmd_decomposition",
    "time_frequency_feature_construction",
    "cycleformer_period_token_regression",
    "pseudo_label_temporal_domain_adaptation",
    "heart_rate_regression",
)
INNOVATION_MODULES = (
    "EDACM target phase representation",
    "HR-AdaVMD radar micro-motion decomposition",
    "CycleFormer physiological period-token modeling",
    "Pseudo-label temporal adaptation for cross-dataset heart-rate estimation",
)
DOMAIN_ADAPTATION_METHOD = "Pseudo-label adaptation + temporal confidence weighting"

def frequency_features(
    modes: np.ndarray,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    time_modes = np.asarray(modes, dtype=np.float32)
    if time_modes.ndim == 1:
        time_modes = time_modes[None, :]
    n = int(time_modes.shape[-1])
    if n == 0:
        return (
            np.empty((*time_modes.shape[:-1], 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {
                "frequency_feature": "hann_rfft_log_magnitude",
                "sampling_rate_hz": float(sampling_rate_hz),
                "frequency_bins": 0,
                "frequency_resolution_hz": None,
            },
        )

    window = np.hanning(n).astype(np.float32)
    spectrum = np.fft.rfft(time_modes * window[None, :], axis=-1)
    x_freq = np.log1p(np.abs(spectrum)).astype(np.float32)
    x_freq = np.stack([zscore_1d(row) for row in x_freq], axis=0).astype(np.float32)
    freq_hz = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate_hz)).astype(np.float32)
    meta = {
        "frequency_feature": "hann_rfft_log_magnitude",
        "sampling_rate_hz": float(sampling_rate_hz),
        "frequency_bins": int(freq_hz.size),
        "frequency_resolution_hz": float(freq_hz[1] - freq_hz[0]) if freq_hz.size > 1 else None,
        "frequency_range_hz": [float(freq_hz[0]), float(freq_hz[-1])] if freq_hz.size else [],
    }
    return x_freq, freq_hz, meta


def rda_log_magnitude(radar: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(radar)).astype(np.float32)


def _raw_target_bin_signal(radar: np.ndarray) -> np.ndarray:
    """Raw complex (L,) signal at the SAME single target bin EDACM selects.

    Using EDACM's target-bin criterion (vital-sign stability score) but the raw
    signal keeps the raw-vs-EDACM comparison fair: the only difference is the
    signal processing (raw magnitude/complex vs EDACM phase + VMD), not *which*
    radar bin is read.
    """
    selected_bins, _weights, _meta = select_target_bin_indices(radar, top_bins=1)
    d, a, r = int(selected_bins[0][0]), int(selected_bins[0][1]), int(selected_bins[0][2])
    return radar[:, d, a, r].astype(np.complex64)


def _wrap(
    representation_name: str,
    x_time: Optional[np.ndarray] = None,
    x_freq: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    x_rda: Optional[np.ndarray] = None,
    freq_hz: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> RadarRepresentation:
    """Build a :class:`RadarRepresentation` with a fixed-schema ``meta``.

    Always records the channel/length geometry so the training pipeline can size
    models from the data instead of a name->shape registry.
    """
    meta = dict(meta or {})
    meta["sample_rate"] = float(DEFAULT_SAMPLING_RATE_HZ)
    meta["representation"] = representation_name
    meta["generator_version"] = REPRESENTATION_GENERATOR_VERSION
    if x_time is not None:
        meta["time_channels"] = int(x_time.shape[0])
        meta["window_size"] = int(x_time.shape[1])
    if x_freq is not None:
        meta["freq_channels"] = int(x_freq.shape[0])
        meta["freq_length"] = int(x_freq.shape[1])
    return RadarRepresentation(
        x_time=x_time,
        x_freq=x_freq,
        representation_name=representation_name,
        representation_version=REPRESENTATION_GENERATOR_VERSION,
        meta=meta,
        x=x,
        x_rda=x_rda,
        freq_hz=freq_hz,
    )


def _time_freq_bundle(
    x_time: np.ndarray,
    representation_alias: str,
    base_meta: Dict[str, Any],
) -> RadarRepresentation:
    """Build the standard (x_time, x_freq) bundle for a time-domain tensor.

    Every ablation representation emits BOTH x_time (C, L) and x_freq (C, 129)
    so the dual time+freq backbones consume them uniformly — this is what makes
    the representation factorial comparable. x_freq is the Hann-windowed, log
    RFFT of x_time with per-channel z-scoring (see :func:`frequency_features`).
    """
    x_time = np.asarray(x_time, dtype=np.float32)
    x_freq, freq_hz, freq_meta = frequency_features(x_time)
    meta = dict(base_meta)
    meta.update(freq_meta)
    return _wrap(
        representation_alias,
        x_time=x_time,
        x_freq=x_freq,
        x=x_time,
        freq_hz=freq_hz,
        meta=meta,
    )


def radar_to_feature_bundle(
    radar: np.ndarray,
    representation: str,
) -> RadarRepresentation:
    if representation == "real_imag":
        x = np.stack([np.real(radar), np.imag(radar)], axis=0).astype(np.float32)
        return _wrap("real_imag", x=x, meta={})
    elif representation == "magnitude":
        x = np.abs(radar).astype(np.float32)
        return _wrap("magnitude", x=x, meta={})
    elif representation == "log_magnitude":
        x = np.log1p(np.abs(radar)).astype(np.float32)
        return _wrap("log_magnitude", x=x, meta={})
    elif representation == "target_edacm":
        phase, phase_meta = target_edacm_signal(radar)
        x = phase[None, :].astype(np.float32)
        return _wrap("target_edacm", x=x, meta=phase_meta)
    elif representation in {"proposed", "target_edacm_vmd"}:
        phase, phase_meta = target_edacm_signal(radar)
        x_time, hr_adavmd_meta = hr_adavmd_decompose(phase, k=DEFAULT_VMD_K)
        x_freq, freq_hz, freq_meta = frequency_features(x_time)
        x_rda = rda_log_magnitude(radar)
        phase_meta.update(
            {
                "method_pipeline": list(METHOD_PIPELINE),
                "innovation_modules": list(INNOVATION_MODULES),
                "domain_adaptation_method": DOMAIN_ADAPTATION_METHOD,
                "representation_method": "EDACM phase representation + HR-AdaVMD decomposition + FFT spectrum for CycleFormer",
                "representation_alias": representation,
                "vmd_k": DEFAULT_VMD_K,
                "vmd_alpha": DEFAULT_VMD_ALPHA,
                "vmd_max_iter": DEFAULT_VMD_MAX_ITER,
                "vmd_tol": DEFAULT_VMD_TOL,
                "vmd_output_shape": list(x_time.shape),
                **hr_adavmd_meta,
                "feature_domains": ["time", "frequency"],
                "x_time_shape": list(x_time.shape),
                "x_freq_shape": list(x_freq.shape),
                "x_rda_shape": list(x_rda.shape),
                "x_rda_representation": DEFAULT_RDA_REPRESENTATION,
                **freq_meta,
            }
        )
        return _wrap(
            representation,
            x_time=x_time.astype(np.float32),
            x_freq=x_freq.astype(np.float32),
            x=x_time.astype(np.float32),
            x_rda=x_rda.astype(np.float32),
            freq_hz=freq_hz.astype(np.float32),
            meta=phase_meta,
        )
    # --- Representation ablation branches (all emit x_time + x_freq) ---
    elif representation == "edacm_only":
        # EDACM phase, NO VMD. Isolates the contribution of the micro-motion
        # decomposition relative to "proposed".
        phase, phase_meta = target_edacm_signal(radar)
        x_time = phase[None, :].astype(np.float32)  # (1, L)
        return _time_freq_bundle(x_time, "edacm_only", phase_meta)
    elif representation == "raw_logmag":
        # Raw RDA log-magnitude at the EDACM-selected target bin, NO EDACM, NO VMD.
        # Lowest-effort baseline; isolates the value of the EDACM phase pipeline.
        sig = _raw_target_bin_signal(radar)
        x_time = rda_log_magnitude(sig)[None, :].astype(np.float32)  # (1, L)
        return _time_freq_bundle(x_time, "raw_logmag", {})
    elif representation == "raw_real_imag":
        # Raw RDA complex (real+imag) at the EDACM-selected target bin, NO EDACM, NO VMD.
        sig = _raw_target_bin_signal(radar)
        x_time = np.stack([np.real(sig), np.imag(sig)], axis=0).astype(np.float32)  # (2, L)
        return _time_freq_bundle(x_time, "raw_real_imag", {})
    elif representation == "edacm_vmd_fixed":
        # EDACM + plain VMD with EQUAL mode weights (no HR-aware adaptive
        # weighting). Isolates the value of the adaptive weighting in "proposed".
        phase, phase_meta = target_edacm_signal(radar)
        init_omega = hr_adavmd_initial_omega(k=DEFAULT_VMD_K)
        modes = vmd_decompose(signal=phase, k=DEFAULT_VMD_K, init_omega=init_omega)
        x_time = modes.astype(np.float32)  # (K, L)
        return _time_freq_bundle(x_time, "edacm_vmd_fixed", phase_meta)
    else:
        raise ValueError(f"Unsupported representation: {representation}")


# Representations used for the input-representation ablation (factorial vs backbone).
# "proposed" is the control (the full method); the others ablate one component.
REPRESENTATION_ABLATION_CHOICES = (
    "proposed",
    "edacm_only",
    "raw_logmag",
    "raw_real_imag",
    "edacm_vmd_fixed",
)


# (time_channels, freq_channels) produced by each representation. Backbones are
# built to match these so the dual time+freq contract stays consistent and the
# representation factorial is comparable across backbones.
REPRESENTATION_INPUT_CHANNELS = {
    "proposed": (7, 7),
    "target_edacm_vmd": (7, 7),
    "target_edacm_hr_adavmd": (7, 7),  # legacy 7-mode name used by the headline export
    "edacm_only": (1, 1),
    "raw_logmag": (1, 1),
    "raw_real_imag": (2, 2),
    "edacm_vmd_fixed": (7, 7),
}


def representation_input_channels(representation: str) -> "tuple[int, int]":
    """Return ``(time_channels, freq_channels)`` for a representation name."""
    if representation in REPRESENTATION_INPUT_CHANNELS:
        return REPRESENTATION_INPUT_CHANNELS[representation]
    # Unknown representations default to the canonical 7-channel layout (matches the
    # model defaults and the headline "proposed"/"target_edacm_hr_adavmd" data).
    return (7, 7)

