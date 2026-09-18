"""Spectral helpers shared by every HR readout.

Kept free of torch so that CPU-only analysis scripts (e.g. the temporal
phase diagnostics) can reuse the *identical* peak-picking code as the
training/evaluation entry points. If this file changes, both the FFT/STFT
baseline and the phase diagnostics change together by construction.
"""

from __future__ import annotations

import math

import numpy as np


def band_mask(freq_hz: np.ndarray, hr_min_bpm: float, hr_max_bpm: float) -> np.ndarray:
    return (freq_hz >= hr_min_bpm / 60.0) & (freq_hz <= hr_max_bpm / 60.0)


def rfft_power(
    signal: np.ndarray,
    sampling_rate_hz: float,
    pad: int = 1,
    detrend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum (Hanning windowed) with optional zero padding.

    Returns (freq_hz, power). `pad` multiplies the FFT length and therefore
    divides the bin width: with 256-frame windows the raw bin width is
    4.69 bpm @20 Hz and 7.03 bpm @30 Hz, which is of the same order as the
    within-subject HR variation we are trying to resolve.
    """
    x = np.asarray(signal, dtype=np.float64)
    if detrend:
        x = x - float(np.mean(x))
    n = int(x.size)
    nfft = max(n * max(1, int(pad)), n)
    window = np.hanning(n) if n > 1 else np.ones(1, dtype=np.float64)
    spec = np.fft.rfft(x * window, n=nfft)
    power = np.abs(spec) ** 2
    freq_hz = np.fft.rfftfreq(nfft, d=1.0 / float(sampling_rate_hz))
    return freq_hz.astype(np.float32), power.astype(np.float64)


def peak_from_spectrum(
    spectrum: np.ndarray,
    freq_hz: np.ndarray,
    hr_min_bpm: float,
    hr_max_bpm: float,
    refine: bool = False,
) -> float:
    """Peak HR (bpm) inside the cardiac band, optionally sub-bin refined."""
    mask = band_mask(freq_hz, hr_min_bpm, hr_max_bpm)
    if not np.any(mask):
        return math.nan
    band_freq = freq_hz[mask]
    band_power = spectrum[mask]
    if band_power.size == 0 or not np.isfinite(band_power).any():
        return math.nan
    idx = int(np.nanargmax(band_power))
    if not refine or band_power.size < 3 or idx <= 0 or idx >= band_power.size - 1:
        return float(band_freq[idx] * 60.0)
    # Parabolic interpolation around the peak bin.
    y0 = float(band_power[idx - 1])
    y1 = float(band_power[idx])
    y2 = float(band_power[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    step = float(band_freq[1] - band_freq[0]) if band_freq.size > 1 else 0.0
    if not np.isfinite(denom) or abs(denom) < 1e-12 or step <= 0.0:
        return float(band_freq[idx] * 60.0)
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return float((band_freq[idx] + delta * step) * 60.0)


def band_power_fraction(
    freq_hz: np.ndarray,
    power: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> float:
    """Fraction of total power inside [lo_hz, hi_hz]."""
    total = float(np.sum(power))
    if total <= 0.0:
        return math.nan
    sel = (freq_hz >= lo_hz) & (freq_hz <= hi_hz)
    return float(np.sum(power[sel]) / total)


def power_at_hz(freq_hz: np.ndarray, power: np.ndarray, target_hz: float) -> float:
    """Power in the bin nearest to `target_hz`."""
    if freq_hz.size == 0:
        return math.nan
    idx = int(np.argmin(np.abs(freq_hz - target_hz)))
    return float(power[idx])


def synthesize_cardiac_phase(
    hr_series_bpm: np.ndarray,
    sampling_rate_hz: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Integrate an instantaneous HR series into a clean cardiac phase signal.

    This is the positive control of the temporal line of work: a signal that
    *by construction* contains the reference HR at the same frame rate and
    window length as the radar features. If the readout cannot recover HR from
    it, the limitation is numerical (window length / bin resolution / VMD),
    not physiological.
    """
    hr = np.asarray(hr_series_bpm, dtype=np.float64)
    phase = 2.0 * np.pi * np.cumsum(hr / 60.0) / float(sampling_rate_hz)
    return amplitude * np.sin(phase)
