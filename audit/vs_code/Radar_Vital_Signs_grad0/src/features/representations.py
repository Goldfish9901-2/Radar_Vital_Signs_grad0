"""Feature representation assembly for training windows.

This module is the single routing point for converting a complex RDA window into
model-ready tensors. The proposed path is intentionally explicit:
EDACM phase representation -> HR-AdaVMD -> time/frequency features. Simpler
representations are retained here as baselines for comparison experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

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


def _time_freq_bundle(
    x_time: np.ndarray,
    representation_alias: str,
    base_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the standard (x_time, x_freq) bundle for a time-domain tensor.

    Every ablation representation emits BOTH x_time (C, L) and x_freq (C, 129)
    so the dual time+freq backbones consume them uniformly — this is what makes
    the representation factorial comparable. x_freq is the Hann-windowed, log
    RFFT of x_time with per-channel z-scoring (see :func:`frequency_features`).
    """
    x_time = np.asarray(x_time, dtype=np.float32)
    x_freq, freq_hz, freq_meta = frequency_features(x_time)
    meta = dict(base_meta)
    meta.update(
        {
            "representation_method": representation_alias,
            "representation_alias": representation_alias,
            "feature_domains": ["time", "frequency"],
            "x_time_shape": list(x_time.shape),
            "x_freq_shape": list(x_freq.shape),
            **freq_meta,
        }
    )
    return {
        "x": x_time,
        "x_time": x_time,
        "x_freq": x_freq,
        "freq_hz": freq_hz,
        "meta": meta,
    }


def _complex_mapping_bundle(
    radar: np.ndarray,
    components,
    alias: str,
    base_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Full-cube complex->real mapping kept as a multi-channel time series.

    IMPORTANT (scope / naming): this is NOT a re-derivation of RDA from raw ADC.
    For PhysDrive the input ``radar`` is already the publisher-processed RDA cube
    (Case C: static removal + localization + crop already applied upstream). So
    these mappings are *post-RDA re-representations* of the official cube: they
    change only how the complex tensor is turned into real model inputs.

    Design: keep the frame axis as the time axis (HR lives in the temporal
    oscillation at 20 Hz) and treat every spatial (D, A, R) cell as its own
    channel. x_time therefore has shape (n_components * D*A*R, F). This
    deliberately preserves ALL spatial bins, unlike EDACM which collapses to a
    single target bin, so we can test whether HR information is lost when the
    complex->real mapping discards bins. The dual time+freq contract is kept via
    :func:`_time_freq_bundle` (Hann RFFT over the frame axis per channel).

    Hypothesis gate (Phase C): if a full-cube mapping recovers HR signal that the
    single-bin EDACM phase mapping misses, then the information loss is in the
    complex->feature step, not the backbone.
    """
    base_meta = base_meta or {}
    comps = [fn(radar) for fn in components]  # each (F, D, A, R) real
    chans = [c.reshape(-1, c.shape[0]).astype(np.float32) for c in comps]  # (D*A*R, F)
    x_time = np.concatenate(chans, axis=0).astype(np.float32)  # (n*D*A*R, F)
    meta = dict(base_meta)
    meta.update({
        "complex_mapping": alias,
        "spatial_cells": int(chans[0].shape[0]),
        "time_length": int(chans[0].shape[1]),
        "note": "post-RDA complex->real re-representation (not raw-ADC RDA re-derivation)",
    })
    return _time_freq_bundle(x_time, alias, meta)


def radar_to_feature_bundle(
    radar: np.ndarray,
    representation: str,
) -> Dict[str, Any]:
    if representation == "real_imag":
        x = np.stack([np.real(radar), np.imag(radar)], axis=0).astype(np.float32)
    elif representation == "magnitude":
        x = np.abs(radar).astype(np.float32)
    elif representation == "log_magnitude":
        x = np.log1p(np.abs(radar)).astype(np.float32)
    elif representation == "target_edacm":
        phase, phase_meta = target_edacm_signal(radar)
        x = phase[None, :].astype(np.float32)
        return {"x": x, "meta": phase_meta}
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
        return {
            "x": x_time.astype(np.float32),
            "x_time": x_time.astype(np.float32),
            "x_freq": x_freq.astype(np.float32),
            "x_rda": x_rda.astype(np.float32),
            "freq_hz": freq_hz.astype(np.float32),
            "meta": phase_meta,
        }
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
    # --- Phase C: full-cube complex->real mappings (post-RDA re-representations) ---
    # Hypothesis: EDACM's single-target-bin phase selection may discard HR-bearing
    # energy present in other bins / other complex components. These preserve the
    # whole (D,A,R) cube as parallel channels so the probe can locate that signal.
    elif representation == "rda_real":
        return _complex_mapping_bundle(radar, [np.real], "rda_real")
    elif representation == "rda_imag":
        return _complex_mapping_bundle(radar, [np.imag], "rda_imag")
    elif representation == "rda_magnitude":
        return _complex_mapping_bundle(radar, [np.abs], "rda_magnitude")
    elif representation == "rda_phase":
        return _complex_mapping_bundle(radar, [np.angle], "rda_phase")
    elif representation == "rda_real_imag":
        return _complex_mapping_bundle(radar, [np.real, np.imag], "rda_real_imag")
    elif representation == "rda_mag_phase":
        return _complex_mapping_bundle(radar, [np.abs, np.angle], "rda_mag_phase")
    # --- Phase A: inter-frame phase difference (micro-motion) ---
    # angle(x[t] * conj(x[t-1])) per spatial cell captures the *change* in complex
    # reflection between consecutive 20 Hz frames -- the cardioballistic
    # micro-Doppler that the absolute complex state (magnitude/phase) smears.
    # Real-valued, fed as a multi-channel time series (no EDACM). Post-RDA
    # re-representation (Case C): PhysDrive's cube is publisher-processed RDA.
    elif representation == "rda_phase_diff":
        phase_diff = np.angle(radar[1:] * np.conj(radar[:-1]))  # (F-1, D, A, R) real
        x_time = phase_diff.reshape(-1, phase_diff.shape[0]).astype(np.float32)  # (D*A*R, F-1)
        meta = {
            "complex_mapping": "rda_phase_diff",
            "spatial_cells": int(x_time.shape[0]),
            "time_length": int(x_time.shape[1]),
            "note": "inter-frame phase difference (micro-motion); post-RDA re-representation (not raw-ADC RDA)",
        }
        return _time_freq_bundle(x_time, "rda_phase_diff", meta)
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return {"x": x, "meta": {}}


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
    # Phase C full-cube complex->real mappings. Channel count = n_real_components
    # * (D*A*R). For the unified RDA cube (D=8, A=16, R=8) that is 1024 per real
    # component; two components => 2048. Time axis = frames (length F).
    "rda_real": (1024, 1024),
    "rda_imag": (1024, 1024),
    "rda_magnitude": (1024, 1024),
    "rda_phase": (1024, 1024),
    "rda_real_imag": (2048, 2048),
    "rda_mag_phase": (2048, 2048),
    # Phase A inter-frame phase difference: real (F-1, D, A, R) -> (D*A*R, F-1).
    "rda_phase_diff": (1024, 1024),
}


# Phase C: post-RDA complex->real re-representations (see _complex_mapping_bundle).
# Used to test whether HR information is lost in the complex->feature mapping
# (EDACM collapses to a single target bin + phase) rather than in the backbone.
# These are NOT raw-ADC RDA re-derivations; for PhysDrive the input is the
# publisher-processed RDA cube.
COMPLEX_MAPPING_CHOICES = (
    "rda_real",
    "rda_imag",
    "rda_magnitude",
    "rda_phase",
    "rda_real_imag",
    "rda_mag_phase",
    "rda_phase_diff",  # Phase A
)


def representation_input_channels(representation: str) -> "tuple[int, int]":
    """Return ``(time_channels, freq_channels)`` for a representation name."""
    if representation in REPRESENTATION_INPUT_CHANNELS:
        return REPRESENTATION_INPUT_CHANNELS[representation]
    # Unknown representations default to the canonical 7-channel layout (matches the
    # model defaults and the headline "proposed"/"target_edacm_hr_adavmd" data).
    return (7, 7)

