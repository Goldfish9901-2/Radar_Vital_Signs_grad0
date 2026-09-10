"""Beamforming / angle-processing stage across RX channels.

The angle axis (axis 2) represents spatial samples across RX antennas. Simple FFT
beamforming is the current baseline; this stage exposes alternatives that apply a
spatial filter across the angle axis. To preserve the (F, D, A, R) shape contract
we keep the same number of angle bins but apply a spatial filter that reweights /
steers them. The result is a cube where the angular response is shaped by the
chosen beamformer (e.g. MVDR suppresses spatially-incoherent clutter).

Candidate matrix rows: B0 (FFT, current), B1 (Bartlett / conventional),
B2 (MVDR / Capon adaptive).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .pipeline import register_stage


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def _angle_covariance(cube: np.ndarray) -> np.ndarray:
    """Spatial covariance over the angle axis, averaged over (F, D, R).

    Returns (A, A) complex matrix.
    """
    f, d, a, r = cube.shape
    flat = cube.transpose(1, 3, 0, 2).reshape(f * r * d, a)  # (cells, A)
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = (flat.conj().T @ flat) / max(1, flat.shape[0] - 1)
    return cov


def beamforming_fft(cube: np.ndarray, **_: Any) -> np.ndarray:
    """B0: current baseline -- FFT along the angle axis (identity in our cube since
    the loader already provides angle-resolved bins). Returns cube unchanged but
    kept explicit so the stage is always meaningful."""
    return _complex_safe(cube)


def beamforming_bartlett(cube: np.ndarray, look_directions: int = 16, **_: Any) -> np.ndarray:
    """B1: conventional (Bartlett) beamformer -- steer a spatial filter and
    re-project the cube onto the angle axis.

    For each angle bin we form a steering vector across the RX axis and compute
    the Bartlett spectrum, then re-weight each angle bin by its mean Bartlett
    power (broadening/keeping the dominant direction, attenuating others). Shape
    is preserved.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    # steering vectors: crude linear array assumption, angle bin -> direction cosine
    idx = np.arange(a, dtype=np.float32)
    # normalized spatial frequency per angle bin (treat bin index as direction proxy)
    u = np.linspace(-0.5, 0.5, a, dtype=np.float32)
    # Bartlett power per (f, d, r) cell across angle
    out = np.zeros_like(cube)
    flat = cube.transpose(1, 3, 0, 2).reshape(d * r * f, a)  # (cells, A)
    # steering matrix: A x A, S[k,j] = exp(-2j pi u[k] * j)
    j = np.arange(a, dtype=np.float32)
    S = np.exp(-2j * np.pi * np.outer(u, j)).astype(np.complex64)  # (A, A)
    # Bartlett spectrum: |S^H x|^2 ; gives (cells, A)
    spec = np.abs((S.conj().T @ flat.T)).T ** 2  # (cells, A)
    spec = spec.reshape(d, r, f, a).transpose(2, 0, 3, 1)  # (F, D, A, R)
    # normalize per (F,D,R) so the strongest steering direction is kept ~unity
    spec = spec / (spec.max(axis=2, keepdims=True) + 1e-8)
    return cube * spec


def beamforming_mvdr(cube: np.ndarray, diag_load: float = 1e-2, **_: Any) -> np.ndarray:
    """B2: MVDR / Capon adaptive beamformer -- suppress spatially incoherent
    interference (e.g. multipath inside a car cabin) while keeping the steered
    direction.

    For each angle bin we estimate the spatial covariance across RX, invert it
    with a diagonal load for stability, and form the MVDR filter; the resulting
    angular response reweights the cube. This is the stage that answers "does
    angular sophistication help?" without leaving the shape contract.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    cov = _angle_covariance(cube)  # (A, A)
    # diagonal load for numerical stability on tiny cubes
    cov_loaded = cov + diag_load * np.trace(cov) / a * np.eye(a, dtype=np.complex64)
    try:
        inv = np.linalg.inv(cov_loaded)
    except np.linalg.LinAlgError:
        return cube
    # steering vectors per angle bin (same proxy as Bartlett)
    u = np.linspace(-0.5, 0.5, a, dtype=np.float32)
    j = np.arange(a, dtype=np.float32)
    S = np.exp(-2j * np.pi * np.outer(u, j)).astype(np.complex64)  # (A, A)
    # MVDR angular gain per steering direction k: 1 / (s_k^H R^-1 s_k)
    gains = np.zeros(a, dtype=np.float32)
    for k in range(a):
        s = S[k]
        denom = np.real(s.conj() @ inv @ s)
        gains[k] = 1.0 / max(float(denom), 1e-8)
    # normalize so max gain ~ unity
    gains = gains / (gains.max() + 1e-8)
    # apply gain per angle bin (same gain across all F,D,R for this cheap version)
    weight = gains[None, None, :, None].astype(np.complex64)
    return cube * weight


def register_beamforming_methods() -> None:
    register_stage("beamforming", "fft", beamforming_fft)
    register_stage("beamforming", "bartlett", beamforming_bartlett)
    register_stage("beamforming", "mvdr", beamforming_mvdr)
