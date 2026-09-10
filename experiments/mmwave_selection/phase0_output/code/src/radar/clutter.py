"""Clutter suppression stage: remove static / stationary reflection.

The cube is complex ``(F, D, A, R)``. Stationary reflectors (walls, seat frames,
body mass not moving) produce a slow-time (frame-axis) component with ~zero
Doppler; vital-sign micro-motion lives in the fast oscillatory part. All methods
here operate along the frame axis (axis 0) and return a cube of the same shape.

Candidate matrix rows (see docs/RDA_ALGORITHMS.md): C0..C4.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .pipeline import register_stage, DEFAULT_FS


def _complex_safe(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.complex64)


def clutter_none(cube: np.ndarray, **_: Any) -> np.ndarray:
    """Baseline: no extra clutter suppression (cube assumed already static-removed)."""
    return _complex_safe(cube)


def clutter_mean_subtraction(cube: np.ndarray, **_: Any) -> np.ndarray:
    """C0: subtract the slow-time mean per (D, A, R) cell (zeros out DC)."""
    cube = _complex_safe(cube)
    mean = cube.mean(axis=0, keepdims=True)
    return cube - mean


def clutter_temporal_highpass(cube: np.ndarray, cutoff: float = 0.1,
                              fs: float = DEFAULT_FS, **_: Any) -> np.ndarray:
    """C1: 1st-order IIR high-pass along frames to kill sub-``cutoff``-Hz drift.

    ``cutoff`` is in Hz relative to the frame rate; mapped to an alpha via the
    standard one-pole high-pass recurrence. Cheap, label-free, fully reversible
    shape-wise.
    """
    cube = _complex_safe(cube)
    n = cube.shape[0]
    if n < 2:
        return cube
    # alpha chosen so the pole sits at `cutoff` Hz at the cube frame rate `fs`.
    rc = 1.0 / max(float(cutoff), 1e-3)
    dt = 1.0 / float(fs)
    alpha = rc / (rc + dt)
    out = np.empty_like(cube)
    out[0] = 0.0
    for t in range(1, n):
        out[t] = alpha * (out[t - 1] + cube[t] - cube[t - 1])
    return out


def clutter_mti(cube: np.ndarray, **_: Any) -> np.ndarray:
    """C2: 2-pulse MTI canceler (frame difference) to remove zero-Doppler clutter."""
    cube = _complex_safe(cube)
    if cube.shape[0] < 2:
        return cube
    diff = np.diff(cube, axis=0)
    # pad first frame so the (F, D, A, R) shape is preserved
    return np.concatenate([cube[:1], diff], axis=0)


def clutter_pca(cube: np.ndarray, n_components: int = 1, **_: Any) -> np.ndarray:
    """C3: PCA/SVD on slow-time vectors per (D,A,R) cell; drop top principal axis.

    The most energetic slow-time mode is usually the static / very-low-freq
    reflector. Projecting it out suppresses clutter while keeping fast micro-motion.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    flat = cube.reshape(f, d * a * r)  # (F, cells)
    # Work in real augmented space to keep it simple and label-free.
    real = np.concatenate([flat.real, flat.imag], axis=0)  # (2F, cells)
    mean = real.mean(axis=0, keepdims=True)
    centered = real - mean
    # covariance over cells (cheap: cells >> 2F typically for small cubes, but use
    # the smaller dimension for the eigen-decomposition)
    cov = centered @ centered.T / max(1, centered.shape[1] - 1)  # (2F, 2F)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return cube
    # largest eigenvalues are the dominant slow modes -> project them out
    k = min(int(n_components), eigvals.size)
    top_idx = np.argsort(eigvals)[::-1][:k]
    proj = eigvecs @ (eigvecs.T @ centered)  # reconstruction of dominant modes
    cleaned = centered - proj[top_idx].sum(axis=0) if k > 0 else centered
    cleaned = cleaned + mean
    out = cleaned[:f] + 1j * cleaned[f:]
    return out.reshape(f, d, a, r).astype(np.complex64)


def clutter_rpca(cube: np.ndarray, lamb: float = 0.05, n_iter: int = 30, **_: Any) -> np.ndarray:
    """C4: lightweight low-rank + sparse (RPCA) split via inexact ALM on slow-time.

    Models the cube as low-rank (static / structured clutter) + sparse (micro-
    motion transients). We keep the *sparse* part as the cleaned signal. This is a
    reduced-complexity PCP, not a full RPCA, intended for CPU sanity checks.
    """
    cube = _complex_safe(cube)
    f, d, a, r = cube.shape
    mat = cube.reshape(f, d * a * r).astype(np.complex64)
    # operate on magnitude-normalised real matrix for stability
    mag = np.abs(mat)
    norm = float(np.max(mag)) + 1e-8
    X = (mat / norm).astype(np.complex64)
    # real + imag stacked
    Xr = np.concatenate([X.real, X.imag], axis=0)
    m, n = Xr.shape
    Y = np.zeros_like(Xr)
    mu = 1.0 / (4.0 * np.linalg.norm(Xr, 2) + 1e-8)
    svthr = 1.0 / mu
    lrank = np.zeros_like(Xr)
    for _ in range(n_iter):
        # sparse update (hard threshold)
        S = np.sign(Xr - lrank + Y / mu) * np.maximum(
            np.abs(Xr - lrank + Y / mu) - lamb / mu, 0.0
        )
        # low-rank update (singular value threshold)
        U, s, Vt = np.linalg.svd(Xr - S + Y / mu, full_matrices=False)
        s_thr = np.maximum(s - svthr, 0.0)
        lrank = (U * s_thr) @ Vt
        Y = Y + mu * (Xr - lrank - S)
    sparse = (Xr - lrank)[:f] + 1j * (Xr - lrank)[f:]
    return (sparse * norm).reshape(f, d, a, r).astype(np.complex64)


def register_clutter_methods() -> None:
    register_stage("clutter", "none", clutter_none)
    register_stage("clutter", "mean_subtraction", clutter_mean_subtraction)
    register_stage("clutter", "temporal_highpass", clutter_temporal_highpass)
    register_stage("clutter", "mti", clutter_mti)
    register_stage("clutter", "pca", clutter_pca)
    register_stage("clutter", "rpca", clutter_rpca)
