"""Signal decomposition algorithms for radar vital signs.

All decompose functions share the interface:
    signal (N,) -> modes (K_out, N)

Each returns z-scored modes padded/truncated to ``k_out`` rows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def zscore_1d(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if y.size == 0 or not np.isfinite(y).any():
        return np.zeros_like(y, dtype=np.float32)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if np.isfinite(std) and std > 1e-6:
        y = (y - mean) / std
    else:
        y = y - mean
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def _pad_or_truncate(modes: np.ndarray, k_out: int) -> np.ndarray:
    if modes.shape[0] >= k_out:
        return modes[:k_out]
    pad = np.zeros((k_out - modes.shape[0], modes.shape[1]), dtype=np.float32)
    return np.concatenate([modes, pad], axis=0)


def decompose(
    signal: np.ndarray,
    method: str = "vmd",
    k: int = 7,
    alpha: float = 2000.0,
    max_iter: int = 120,
    tol: float = 1e-5,
    sampling_rate_hz: float = 20.0,
) -> np.ndarray:
    methods = {
        "vmd": lambda: vmd_decompose(signal, k=k, alpha=alpha, max_iter=max_iter, tol=tol),
        "svmd": lambda: svmd_decompose(signal, k_out=k, alpha=alpha, max_iter=max_iter, tol=tol),
        "iapvmd": lambda: iapvmd_decompose(signal, k_out=k, max_iter=max_iter, tol=tol, sampling_rate_hz=sampling_rate_hz),
        "evmd": lambda: evmd_decompose(signal, k=k, alpha=alpha, max_iter=max_iter, tol=tol),
        "ewt": lambda: ewt_decompose(signal, k_out=k, sampling_rate_hz=sampling_rate_hz),
    }
    if method not in methods:
        raise ValueError(f"Unsupported decomposition method: {method}. Choose from {list(methods.keys())}")
    return methods[method]()


# ---------------------------------------------------------------------------
# VMD (baseline ADMM)
# ---------------------------------------------------------------------------

def vmd_decompose(
    signal: np.ndarray,
    k: int = 7,
    alpha: float = 2000.0,
    max_iter: int = 120,
    tol: float = 1e-5,
) -> np.ndarray:
    x = zscore_1d(signal)
    n = int(x.size)
    if n == 0:
        return np.empty((k, 0), dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros((k, n), dtype=np.float32)

    freqs = np.fft.fftfreq(n).astype(np.float32)
    spectrum = np.fft.fft(x).astype(np.complex64)
    positive = freqs >= 0

    u_hat = np.zeros((k, n), dtype=np.complex64)
    omega = np.linspace(0.0, 0.5, k + 2, dtype=np.float32)[1:-1]
    lambda_hat = np.zeros(n, dtype=np.complex64)
    tau = 0.0

    for _ in range(max_iter):
        previous = u_hat.copy()
        sum_modes = np.sum(u_hat, axis=0)
        for mode_idx in range(k):
            residual = spectrum - (sum_modes - u_hat[mode_idx]) - lambda_hat / 2.0
            denom = 1.0 + alpha * (freqs - omega[mode_idx]) ** 2
            update = residual / denom
            update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            update = np.clip(update.real, -1e6, 1e6) + 1j * np.clip(update.imag, -1e6, 1e6)
            u_hat[mode_idx] = update.astype(np.complex64)
            power = np.abs(u_hat[mode_idx, positive]) ** 2
            power_sum = float(np.sum(power))
            if np.isfinite(power_sum) and power_sum > 1e-12:
                new_omega = float(np.sum(freqs[positive] * power) / power_sum)
                if np.isfinite(new_omega):
                    omega[mode_idx] = float(np.clip(new_omega, 0.0, 0.5))
        lambda_hat = lambda_hat + tau * (np.sum(u_hat, axis=0) - spectrum)
        diff = np.linalg.norm(u_hat - previous) / (np.linalg.norm(previous) + 1e-12)
        if not np.isfinite(diff):
            u_hat = previous
            break
        if diff < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=1)).astype(np.float32)
    modes[~np.isfinite(modes)] = 0.0
    return np.stack([zscore_1d(mode) for mode in modes], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# SVMD (Successive VMD via Variational Mode Extraction)
# ---------------------------------------------------------------------------

def _vme_extract(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    positive: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    omega_init: float,
) -> tuple[np.ndarray, float]:
    n = spectrum.size
    u_hat = np.zeros(n, dtype=np.complex64)
    omega = omega_init
    lambda_hat = np.zeros(n, dtype=np.complex64)

    for _ in range(max_iter):
        prev_u = u_hat.copy()
        residual = spectrum - lambda_hat / 2.0
        denom = 1.0 + alpha * (freqs - omega) ** 2
        update = residual / denom
        update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        update = np.clip(update.real, -1e6, 1e6) + 1j * np.clip(update.imag, -1e6, 1e6)
        u_hat = update.astype(np.complex64)

        power = np.abs(u_hat[positive]) ** 2
        power_sum = float(np.sum(power))
        if np.isfinite(power_sum) and power_sum > 1e-12:
            new_omega = float(np.sum(freqs[positive] * power) / power_sum)
            if np.isfinite(new_omega):
                omega = float(np.clip(new_omega, 0.0, 0.5))

        diff = np.linalg.norm(u_hat - prev_u) / (np.linalg.norm(prev_u) + 1e-12)
        if not np.isfinite(diff):
            break
        if diff < tol:
            break

    return u_hat, omega


def svmd_decompose(
    signal: np.ndarray,
    k_out: int = 7,
    alpha: float = 2000.0,
    max_iter: int = 120,
    tol: float = 1e-5,
    max_modes: int = 15,
    residual_threshold: float = 0.01,
) -> np.ndarray:
    x = zscore_1d(signal)
    n = int(x.size)
    if n == 0:
        return np.empty((k_out, 0), dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros((k_out, n), dtype=np.float32)

    freqs = np.fft.fftfreq(n).astype(np.float32)
    positive = freqs >= 0
    total_energy = float(np.sum(np.abs(x) ** 2))
    if total_energy < 1e-12:
        return np.zeros((k_out, n), dtype=np.float32)

    modes_list: list[np.ndarray] = []
    residual_signal = x.copy()

    for _ in range(min(max_modes, k_out * 2)):
        residual_spectrum = np.fft.fft(residual_signal).astype(np.complex64)
        residual_energy = float(np.sum(np.abs(residual_signal) ** 2))
        if residual_energy / total_energy < residual_threshold:
            break

        power_spectrum = np.abs(residual_spectrum[positive]) ** 2
        power_spectrum[0] = 0
        if power_spectrum.size > 1:
            peak_idx = int(np.argmax(power_spectrum[1:])) + 1
            omega_init = float(freqs[positive][peak_idx]) if peak_idx < positive.sum() else 0.25
        else:
            omega_init = 0.25

        u_hat_k, _ = _vme_extract(
            residual_spectrum, freqs, positive, alpha, max_iter, tol, omega_init
        )

        mode_k = np.real(np.fft.ifft(u_hat_k)).astype(np.float32)
        mode_k[~np.isfinite(mode_k)] = 0.0
        modes_list.append(mode_k)

        residual_signal = residual_signal - mode_k

    if not modes_list:
        return np.zeros((k_out, n), dtype=np.float32)

    modes = np.stack(modes_list, axis=0)
    modes = np.stack([zscore_1d(m) for m in modes], axis=0).astype(np.float32)
    return _pad_or_truncate(modes, k_out)


# ---------------------------------------------------------------------------
# IAPVMD (Improved Adaptive Parameter VMD)
# ---------------------------------------------------------------------------

def _energy_loss_rate(signal: np.ndarray, modes: np.ndarray) -> float:
    e_signal = float(np.sum(signal ** 2))
    if e_signal < 1e-12:
        return 1.0
    e_modes = float(np.sum(modes ** 2))
    return abs(e_signal - e_modes) / e_signal


def _mode_discrimination(
    modes: np.ndarray,
    sampling_rate_hz: float,
    hr_band: tuple[float, float] = (0.75, 2.5),
    resp_band: tuple[float, float] = (0.1, 0.75),
) -> bool:
    n = modes.shape[1]
    if n < 16:
        return False
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    resp_mask = (freqs >= resp_band[0]) & (freqs <= resp_band[1])
    if not hr_mask.any() or not resp_mask.any():
        return False

    hr_mode_idx = -1
    resp_mode_idx = -1
    window = np.hanning(n).astype(np.float32)

    for idx in range(modes.shape[0]):
        spectrum = np.abs(np.fft.rfft(modes[idx] * window))
        hr_energy = float(np.sum(spectrum[hr_mask] ** 2))
        resp_energy = float(np.sum(spectrum[resp_mask] ** 2))
        total = hr_energy + resp_energy
        if total < 1e-12:
            continue
        if hr_energy > resp_energy and hr_energy / total > 0.6:
            if hr_mode_idx == -1:
                hr_mode_idx = idx
        elif resp_energy > hr_energy and resp_energy / total > 0.6:
            if resp_mode_idx == -1:
                resp_mode_idx = idx

    return hr_mode_idx >= 0 and resp_mode_idx >= 0 and hr_mode_idx != resp_mode_idx


def iapvmd_decompose(
    signal: np.ndarray,
    k_out: int = 7,
    max_iter: int = 120,
    tol: float = 1e-5,
    sampling_rate_hz: float = 20.0,
    k_range: tuple[int, int] = (3, 10),
    alpha_candidates: Optional[list[float]] = None,
) -> np.ndarray:
    x = zscore_1d(signal)
    n = int(x.size)
    if n == 0:
        return np.empty((k_out, 0), dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros((k_out, n), dtype=np.float32)

    if alpha_candidates is None:
        alpha_candidates = [500.0, 1000.0, 2000.0, 3000.0, 5000.0]

    best_params: Optional[tuple[int, float]] = None
    best_loss = float("inf")
    best_modes: Optional[np.ndarray] = None

    for k_try in range(k_range[0], k_range[1] + 1):
        for alpha_try in alpha_candidates:
            modes_raw = vmd_decompose(x, k=k_try, alpha=alpha_try, max_iter=max_iter, tol=tol)
            raw_modes_unnorm = np.real(
                np.fft.ifft(
                    np.fft.fft(x) * np.ones((k_try, 1), dtype=np.complex64),
                    axis=1,
                )
            ).astype(np.float32)
            elr = _energy_loss_rate(x, modes_raw)
            discriminated = _mode_discrimination(modes_raw, sampling_rate_hz)

            if discriminated and elr < best_loss:
                best_loss = elr
                best_params = (k_try, alpha_try)
                best_modes = modes_raw

    if best_modes is None:
        best_modes = vmd_decompose(x, k=k_out, alpha=2000.0, max_iter=max_iter, tol=tol)

    return _pad_or_truncate(best_modes, k_out)


# ---------------------------------------------------------------------------
# EVMD (Enhanced VMD with adaptive per-mode penalty)
# ---------------------------------------------------------------------------

def evmd_decompose(
    signal: np.ndarray,
    k: int = 7,
    alpha: float = 2000.0,
    max_iter: int = 120,
    tol: float = 1e-5,
) -> np.ndarray:
    x = zscore_1d(signal)
    n = int(x.size)
    if n == 0:
        return np.empty((k, 0), dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros((k, n), dtype=np.float32)

    freqs = np.fft.fftfreq(n).astype(np.float32)
    spectrum = np.fft.fft(x).astype(np.complex64)
    positive = freqs >= 0

    u_hat = np.zeros((k, n), dtype=np.complex64)
    omega = np.linspace(0.0, 0.5, k + 2, dtype=np.float32)[1:-1]
    lambda_hat = np.zeros(n, dtype=np.complex64)
    tau = 0.0
    alpha_k = np.full(k, alpha, dtype=np.float32)

    for iteration in range(max_iter):
        previous = u_hat.copy()
        sum_modes = np.sum(u_hat, axis=0)

        energies = np.array([float(np.sum(np.abs(u_hat[m]) ** 2)) for m in range(k)])
        total_energy = float(np.sum(energies)) + 1e-12
        gamma = energies / total_energy

        for mode_idx in range(k):
            if iteration > 0:
                mode_time = np.real(np.fft.ifft(u_hat[mode_idx])).astype(np.float32)
                analytic = mode_time + 1j * np.imag(
                    np.fft.ifft(np.abs(np.fft.fft(mode_time)))
                )
                analytic[~np.isfinite(analytic)] = 0.0
                phase = np.unwrap(np.angle(analytic))
                if phase.size > 1:
                    inst_freq = np.abs(np.diff(phase))
                    mean_inst_freq = float(np.nanmean(inst_freq)) / (2.0 * np.pi)
                else:
                    mean_inst_freq = omega[mode_idx]
                sigma_k = 0.05
                beta = float(np.exp(-((mean_inst_freq - omega[mode_idx]) ** 2) / (2.0 * sigma_k ** 2)))
                alpha_k[mode_idx] = alpha * max(beta, 0.1) * max(gamma[mode_idx], 0.1 / k)
                alpha_k[mode_idx] = float(np.clip(alpha_k[mode_idx], alpha * 0.01, alpha * 10.0))

            residual = spectrum - (sum_modes - u_hat[mode_idx]) - lambda_hat / 2.0
            denom = 1.0 + alpha_k[mode_idx] * (freqs - omega[mode_idx]) ** 2
            update = residual / denom
            update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            update = np.clip(update.real, -1e6, 1e6) + 1j * np.clip(update.imag, -1e6, 1e6)
            u_hat[mode_idx] = update.astype(np.complex64)

            power = np.abs(u_hat[mode_idx, positive]) ** 2
            power_sum = float(np.sum(power))
            if np.isfinite(power_sum) and power_sum > 1e-12:
                new_omega = float(np.sum(freqs[positive] * power) / power_sum)
                if np.isfinite(new_omega):
                    omega[mode_idx] = float(np.clip(new_omega, 0.0, 0.5))
            sum_modes = np.sum(u_hat, axis=0)

        lambda_hat = lambda_hat + tau * (np.sum(u_hat, axis=0) - spectrum)
        diff = np.linalg.norm(u_hat - previous) / (np.linalg.norm(previous) + 1e-12)
        if not np.isfinite(diff):
            u_hat = previous
            break
        if diff < tol:
            break

    modes = np.real(np.fft.ifft(u_hat, axis=1)).astype(np.float32)
    modes[~np.isfinite(modes)] = 0.0
    return np.stack([zscore_1d(mode) for mode in modes], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# EWT (Empirical Wavelet Transform)
# ---------------------------------------------------------------------------

def _detect_spectral_boundaries(
    spectrum_mag: np.ndarray,
    k_out: int,
    min_peak_ratio: float = 0.05,
) -> np.ndarray:
    n_freq = spectrum_mag.size
    if n_freq < 4:
        return np.linspace(0, n_freq, k_out + 1).astype(np.float32)

    smoothed = spectrum_mag.copy()
    smoothed[0] = 0
    threshold = min_peak_ratio * float(np.max(smoothed))

    peaks: list[int] = []
    for i in range(1, n_freq - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1] and smoothed[i] > threshold:
            peaks.append(i)

    if len(peaks) < 2:
        return np.linspace(0, n_freq, k_out + 1).astype(np.float32)

    peaks_sorted = sorted(peaks, key=lambda i: -smoothed[i])
    n_boundaries = min(k_out - 1, len(peaks_sorted) - 1)
    if n_boundaries < 1:
        return np.linspace(0, n_freq, k_out + 1).astype(np.float32)

    selected_peaks = sorted(peaks_sorted[: n_boundaries + 1])
    boundaries = [0.0]
    for i in range(len(selected_peaks) - 1):
        mid = (selected_peaks[i] + selected_peaks[i + 1]) / 2.0
        boundaries.append(mid)
    boundaries.append(float(n_freq))

    while len(boundaries) < k_out + 1:
        max_gap = 0
        max_gap_idx = 0
        for i in range(len(boundaries) - 1):
            gap = boundaries[i + 1] - boundaries[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
        if max_gap < 2.0:
            boundaries.append(boundaries[-1] + 1.0)
        else:
            new_b = (boundaries[max_gap_idx] + boundaries[max_gap_idx + 1]) / 2.0
            boundaries.insert(max_gap_idx + 1, new_b)

    return np.sort(np.array(boundaries[: k_out + 1], dtype=np.float32))


def _meyer_scaling(t: np.ndarray, omega: float) -> np.ndarray:
    abs_t = np.abs(t)
    out = np.zeros_like(t)
    mask = abs_t <= omega * 2.0 / 3.0
    out[mask] = 1.0
    transition = (abs_t > omega * 2.0 / 3.0) & (abs_t <= omega * 4.0 / 3.0)
    if transition.any():
        x = (3.0 * abs_t[transition] / (2.0 * omega) - 1.0)
        x = np.clip(x, 0.0, 1.0)
        f = x ** 2 * (3.0 - 2.0 * x)
        out[transition] = np.cos(np.pi / 2.0 * f)
    return out


def _meyer_wavelet(t: np.ndarray, omega_lo: float, omega_hi: float) -> np.ndarray:
    out = np.zeros_like(t)
    abs_t = np.abs(t)
    in_band = (abs_t > omega_lo * 2.0 / 3.0) & (abs_t <= omega_hi * 4.0 / 3.0)
    if in_band.any():
        scale_lo = _meyer_scaling(t, omega_lo)
        scale_hi = _meyer_scaling(t, omega_hi)
        out[in_band] = scale_lo[in_band] * np.where(
            abs_t[in_band] <= omega_hi * 2.0 / 3.0,
            1.0,
            np.cos(np.pi / 2.0 * np.clip(
                (3.0 * abs_t[in_band] / (2.0 * omega_hi) - 1.0), 0.0, 1.0
            ) ** 2 * (3.0 - 2.0 * np.clip(
                (3.0 * abs_t[in_band] / (2.0 * omega_hi) - 1.0), 0.0, 1.0
            )))
        )
    return out


def ewt_decompose(
    signal: np.ndarray,
    k_out: int = 7,
    sampling_rate_hz: float = 20.0,
) -> np.ndarray:
    x = zscore_1d(signal)
    n = int(x.size)
    if n == 0:
        return np.empty((k_out, 0), dtype=np.float32)
    if not np.isfinite(x).any():
        return np.zeros((k_out, n), dtype=np.float32)

    rfft_mag = np.abs(np.fft.rfft(x)).astype(np.float32)
    boundaries_idx = _detect_spectral_boundaries(rfft_mag, k_out)

    n_rfft = rfft_mag.size
    freq_axis = np.arange(n_rfft, dtype=np.float32)

    modes_list: list[np.ndarray] = []
    for i in range(len(boundaries_idx) - 1):
        lo = float(boundaries_idx[i])
        hi = float(boundaries_idx[i + 1])
        if hi <= lo:
            modes_list.append(np.zeros(n, dtype=np.float32))
            continue

        filt = np.zeros(n_rfft, dtype=np.float32)
        center = (lo + hi) / 2.0
        width = max(hi - lo, 1.0)
        in_band = (freq_axis >= lo) & (freq_axis <= hi)
        if i == 0:
            filt[in_band] = _meyer_scaling(freq_axis[in_band], hi)
        else:
            band_mask = freq_axis[in_band]
            dist_lo = np.abs(band_mask - lo) / max(width * 0.33, 0.5)
            dist_hi = np.abs(band_mask - hi) / max(width * 0.33, 0.5)
            passband = (band_mask >= lo + width * 0.15) & (band_mask <= hi - width * 0.15)
            filt[in_band] = np.where(passband, 1.0, np.exp(-0.5 * np.minimum(dist_lo, dist_hi) ** 2))

        spectrum_rfft = np.fft.rfft(x)
        filtered = spectrum_rfft * filt
        mode = np.fft.irfft(filtered, n=n).astype(np.float32)
        mode[~np.isfinite(mode)] = 0.0
        modes_list.append(mode)

    while len(modes_list) < k_out:
        modes_list.append(np.zeros(n, dtype=np.float32))

    modes = np.stack(modes_list[:k_out], axis=0)
    return np.stack([zscore_1d(m) for m in modes], axis=0).astype(np.float32)
