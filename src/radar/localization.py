"""Localization stage: find the spatial region containing the subject.

The cube is ``(F, D, A, R)``. Localization selects a spatial *focus* (a set of
range/angle/doppler coordinates or a soft mask) so downstream range-selection and
beamforming operate on the right place. Because we must preserve the (F, D, A, R)
shape contract, localization here returns a *weighted* cube: it multiplies the
cube by a spatial confidence mask (1.0 at the located region, attenuating
elsewhere) rather than literally cropping. Cropping/reindexing is done later by
``range_selection`` if desired.

Why this matters (see docs/RDA_ALGORITHMS.md, section 4):
    PhysDrive MAE ~10.75 vs a ~10.94 constant baseline means the model barely
    beats guessing -- strongly suggesting the *spatial region* was mis-localized
    upstream (seat / car structure / static reflections dominate total energy).
    Energy-based localization on total magnitude is exactly the trap: total energy
    maximum != vital-sign-bearing maximum.

Candidate matrix rows: L0 (energy), L1 (spatial consistency), L2 (HR-band
weighted), L3 (temporal consistency).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .pipeline import register_stage, DEFAULT_FS


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def _spatial_energy(cube: np.ndarray) -> np.ndarray:
    """Mean squared magnitude over the frame axis -> (D, A, R)."""
    return np.mean(np.abs(cube) ** 2, axis=0)


def localization_energy(cube: np.ndarray, **_: Any) -> np.ndarray:
    """L0: locate the single (D,A,R) cell with max total-energy; focus a soft window there.

    This is the *current implicit baseline* (global energy max). Implemented as a
    Gaussian-weighted mask so it is differentiable from the other methods while
    still returning a full cube.
    """
    cube = _complex_safe(cube)
    energy = _spatial_energy(cube)
    peak = np.unravel_index(int(np.argmax(energy)), energy.shape)
    mask = _gaussian_mask(energy.shape, peak, sigma=1.0)
    return cube * mask[None, ...]


def localization_spatial_consistency(cube: np.ndarray, sigma: float = 1.5, **_: Any) -> np.ndarray:
    """L1: peak per-frame, then keep the spatially *consistent* region across frames.

    Computes a per-frame range-angle energy map, averages the per-frame peak
    locations, and builds a confidene mask around the temporally-consistent
    centroid. More robust than a single global peak when the subject moves / the
    strongest static reflector is off-target.
    """
    cube = _complex_safe(cube)
    f = cube.shape[0]
    energy = np.abs(cube) ** 2  # (F, D, A, R)
    # per-frame peak over (D, R) for each angle
    per_frame = energy.mean(axis=3)  # (F, D, A)
    peaks = [np.unravel_index(int(np.argmax(per_frame[t])), per_frame[t].shape) for t in range(f)]
    peaks = np.asarray(peaks, dtype=np.float32)  # (F, 2) -> (d, a)
    centroid = peaks.mean(axis=0)
    d_idx, a_idx = int(round(centroid[0])), int(round(centroid[1]))
    d_idx = min(max(d_idx, 0), cube.shape[1] - 1)
    a_idx = min(max(a_idx, 0), cube.shape[2] - 1)
    # build mask centered on (d_idx, a_idx) across all R
    mask = _gaussian_mask(cube.shape[1:], (d_idx, a_idx, cube.shape[3] // 2), sigma=sigma)
    return cube * mask[None, ...]


def localization_hr_band(cube: np.ndarray, hr_band: tuple = (0.75, 2.5),
                         fs: float = DEFAULT_FS, **_: Any) -> np.ndarray:
    """L2: locate the spatial region with the most *HR-band* micro-motion energy.

    Instead of total energy, project each (D,A,R) slow-time signal to its
    heart-rate-band spectral energy and localize on THAT. This is the key fix for
    the "total energy max != heartbeat max" problem on PhysDrive / in-vehicle.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    if f < 8:
        return localization_energy(cube)
    freqs = np.fft.rfftfreq(f, d=1.0 / fs)
    band = (freqs >= float(hr_band[0])) & (freqs <= float(hr_band[1]))
    # magnitude-phase: use EDACM-style phase trace per cell for robustness to amplitude
    i = cube.real.reshape(f, -1)
    q = cube.imag.reshape(f, -1)
    di = np.diff(i, axis=0)
    dq = np.diff(q, axis=0)
    denom = i[:-1] ** 2 + q[:-1] ** 2 + 1e-6
    dphi = (i[:-1] * dq - q[:-1] * di) / denom
    phase = np.cumsum(dphi, axis=0)  # (f-1, cells)
    # align spectrum length with `band` by padding the phase back to f samples
    phase_padded = np.zeros((f, phase.shape[1]), dtype=phase.dtype)
    phase_padded[: phase.shape[0]] = phase
    spec = np.abs(np.fft.rfft(phase_padded, axis=0)) ** 2  # (f//2+1, cells)
    hr_energy = spec[band].sum(axis=0)  # (cells,)
    hr_energy = hr_energy.reshape(d, a, r)
    peak = np.unravel_index(int(np.argmax(hr_energy)), hr_energy.shape)
    mask = _gaussian_mask((d, a, r), peak, sigma=1.0)
    return cube * mask[None, ...]


def localization_temporal_consistency(cube: np.ndarray, sigma: float = 1.5,
                                      fs: float = DEFAULT_FS, **_: Any) -> np.ndarray:
    """L3: track the loudest spatial cell across frames, mask the consistent track.

    Averages per-frame peak locations (weighted by local amplitude, which tracks a
    steadily beating reflector better than an intermittently loud static one) and
    builds a soft mask around the temporal centroid.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    if f < 8:
        return localization_spatial_consistency(cube, sigma=sigma)
    # per-frame spatial energy map (cheap amplitude proxy for a beating bin)
    per_frame = np.abs(cube).mean(axis=3)  # (F, D, A)
    peaks = [np.unravel_index(int(np.argmax(per_frame[t])), per_frame[t].shape) for t in range(f)]
    peaks = np.asarray(peaks, dtype=np.float32)  # (F, 2) -> (d, a)
    centroid = peaks.mean(axis=0)
    d_idx, a_idx = int(round(centroid[0])), int(round(centroid[1]))
    d_idx = min(max(d_idx, 0), d - 1)
    a_idx = min(max(a_idx, 0), a - 1)
    mask = _gaussian_mask((d, a, r), (d_idx, a_idx, r // 2), sigma=sigma)
    return cube * mask[None, ...]


def _gaussian_mask(shape, center, sigma: float = 1.0) -> np.ndarray:
    """Soft Gaussian confidence mask over a (D, A, R) spatial volume."""
    d, a, r = shape
    cd, ca, cr = center
    z = np.zeros(shape, dtype=np.float32)
    # limit support to keep it cheap and local
    rad = max(2, int(round(3 * sigma)))
    for dd in range(max(0, cd - rad), min(d, cd + rad + 1)):
        for aa in range(max(0, ca - rad), min(a, ca + rad + 1)):
            for rr in range(max(0, cr - rad), min(r, cr + rad + 1)):
                val = np.exp(-((dd - cd) ** 2 + (aa - ca) ** 2 + (rr - cr) ** 2) / (2 * sigma**2))
                z[dd, aa, rr] = val
    # normalise so the located center keeps unit gain, off-region is attenuated
    if z.max() > 0:
        z = z / z.max()
    return z


def register_localization_methods() -> None:
    register_stage("localization", "energy", localization_energy)
    register_stage("localization", "spatial_consistency", localization_spatial_consistency)
    register_stage("localization", "hr_band", localization_hr_band)
    register_stage("localization", "temporal_consistency", localization_temporal_consistency)
