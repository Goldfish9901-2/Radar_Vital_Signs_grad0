"""Build fixed-window training data from unified exported radar datasets.

Input:
    exports/{DATASET}/manifest.csv
    exports/{DATASET}/samples/*.npz

Output:
    training_exports/windows/{DATASET}/{train,val,test}/*.npz
    training_exports/manifest.csv
    training_exports/build_config.json
    training_exports/summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_EXPORTS_DIR = ROOT / "exports"
DEFAULT_OUTPUT_DIR = ROOT / "training_exports"
DEFAULT_DATASETS = ("FTU", "BGT60TR13C", "PhysDrive")
DEFAULT_WINDOW_SIZE = 256
DEFAULT_STRIDE = 128
DEFAULT_MIN_LABEL_COVERAGE = 0.8
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_BGT_LONG_SPLIT = "test"
DEFAULT_REPRESENTATION = "target_edacm_vmd"
DEFAULT_EDACM_TOP_BINS = 3
DEFAULT_VMD_K = 7
DEFAULT_VMD_ALPHA = 2000.0
DEFAULT_VMD_MAX_ITER = 120
DEFAULT_VMD_TOL = 1e-5
DEFAULT_SAMPLING_RATE_HZ = 20.0
DEFAULT_RDA_REPRESENTATION = "log_magnitude"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-window training dataset from unified exports."
    )
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Datasets to include.",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--representation",
        choices=["target_edacm_vmd", "target_edacm", "log_magnitude", "magnitude", "real_imag"],
        default=DEFAULT_REPRESENTATION,
        help="How to convert complex radar tensors for model input.",
    )
    parser.add_argument(
        "--target",
        default="heart_rate",
        help="Training target. This builder currently supports heart_rate only.",
    )
    parser.add_argument(
        "--label-reduction",
        choices=["mean", "median", "center"],
        default="mean",
        help="How to convert frame-level labels into one window label.",
    )
    parser.add_argument(
        "--min-label-coverage",
        type=float,
        default=DEFAULT_MIN_LABEL_COVERAGE,
        help="Minimum non-NaN label fraction inside a window for required targets.",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "window_zscore"],
        default="window_zscore",
        help="Feature normalization applied after representation conversion.",
    )
    parser.add_argument(
        "--max-windows-per-sample",
        type=int,
        default=None,
        help="Optional cap for quick experiments/debugging.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPLIT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def print_stage(message: str) -> None:
    print(f"\n[builder] {message}", flush=True)


def print_progress(current: int, total: int, message: str) -> None:
    if total <= 0:
        bar = "------------------------"
        pct = 0.0
    else:
        ratio = min(max(current / total, 0.0), 1.0)
        done = int(round(ratio * 24))
        bar = "#" * done + "-" * (24 - done)
        pct = ratio * 100.0
    print(f"[builder] [{bar}] {pct:5.1f}% {current}/{total} {message}", flush=True)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def required_targets(target: str) -> Tuple[str, ...]:
    if target != "heart_rate":
        raise ValueError("This training dataset builder currently supports HR only: --target heart_rate")
    return ("heart_rate",)


def validate_split_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split-ratios must contain exactly 3 values")
    vals = tuple(float(x) for x in ratios)
    if any(x < 0 for x in vals):
        raise ValueError("split-ratios cannot contain negative values")
    total = sum(vals)
    if total <= 0:
        raise ValueError("split-ratios must sum to a positive value")
    return tuple(x / total for x in vals)  # type: ignore[return-value]


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_group_keys(
    group_keys: Sequence[str],
    ratios: Tuple[float, float, float],
    seed: int,
    forced_test_groups: Optional[set[str]] = None,
) -> Dict[str, str]:
    forced_test_groups = forced_test_groups or set()
    unique_groups = sorted(set(group_keys))
    flexible_groups = [g for g in unique_groups if g not in forced_test_groups]
    flexible_groups.sort(key=lambda g: stable_unit_interval(g, seed))

    n = len(flexible_groups)
    assignment: Dict[str, str] = {g: "test" for g in forced_test_groups if g in unique_groups}
    if n == 0:
        return assignment
    if n == 1:
        assignment[flexible_groups[0]] = "train"
        return assignment
    if n == 2:
        assignment[flexible_groups[0]] = "train"
        assignment[flexible_groups[1]] = "test"
        return assignment

    train_ratio, val_ratio, test_ratio = ratios
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test

    for group in flexible_groups[:n_train]:
        assignment[group] = "train"
    for group in flexible_groups[n_train : n_train + n_val]:
        assignment[group] = "val"
    for group in flexible_groups[n_train + n_val :]:
        assignment[group] = "test"
    return assignment


def build_domain_split_map(
    samples: Sequence[Dict[str, Any]],
    ratios: Tuple[float, float, float],
    seed: int,
    bgt_long_participants: set[str],
) -> Dict[str, str]:
    groups_by_dataset: Dict[str, List[str]] = {}
    for sample in samples:
        groups_by_dataset.setdefault(sample["dataset"], []).append(get_group_key(sample))

    split_map: Dict[str, str] = {}
    forced_test_groups = {
        f"BGT60TR13C/participant/{participant_id}"
        for participant_id in bgt_long_participants
    }
    for dataset, group_keys in groups_by_dataset.items():
        dataset_forced_test = forced_test_groups if dataset == "BGT60TR13C" else set()
        split_map.update(
            split_group_keys(
                group_keys=group_keys,
                ratios=ratios,
                seed=seed,
                forced_test_groups=dataset_forced_test,
            )
        )
    return split_map


def get_group_key(sample: Dict[str, Any]) -> str:
    dataset = sample["dataset"]
    row = sample.get("source_row", {})
    sample_tag = sample["sample_tag"]
    if dataset in {"FTU", "BGT60TR13C"}:
        participant_id = row.get("participant_id")
        if participant_id not in (None, ""):
            return f"{dataset}/participant/{participant_id}"
    if dataset == "PhysDrive":
        session_id = row.get("session_id")
        if session_id:
            return f"{dataset}/session/{session_id}"
    return f"{dataset}/sample/{sample_tag}"


def is_bgt_long_sample(sample: Dict[str, Any]) -> bool:
    return (
        sample.get("dataset") == "BGT60TR13C"
        and sample.get("source_row", {}).get("measurement_type") == "long"
    )


def collect_bgt_long_participants(samples: Sequence[Dict[str, Any]]) -> set[str]:
    participants: set[str] = set()
    for sample in samples:
        if not is_bgt_long_sample(sample):
            continue
        participant_id = sample.get("source_row", {}).get("participant_id")
        if participant_id not in (None, ""):
            participants.add(str(participant_id))
    return participants


def resolve_npz_path(row: Dict[str, str], exports_dir: Path, dataset: str) -> Optional[Path]:
    raw = row.get("npz_path", "")
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    candidate = exports_dir / dataset / "samples" / path.name
    if candidate.exists():
        return candidate
    return path


def collect_exported_samples(exports_dir: Path, datasets: Sequence[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    seen_npz: set[Tuple[str, str]] = set()
    for dataset in datasets:
        manifest_path = exports_dir / dataset / "manifest.csv"
        for row in read_manifest(manifest_path):
            if row.get("status") != "ok":
                continue
            npz_path = resolve_npz_path(row, exports_dir, dataset)
            if npz_path is None:
                continue
            sample_id = (dataset, str(npz_path.resolve()))
            if sample_id in seen_npz:
                continue
            seen_npz.add(sample_id)
            sample_tag = row.get("sample_tag") or npz_path.stem
            samples.append(
                {
                    "dataset": dataset,
                    "sample_tag": sample_tag,
                    "npz_path": npz_path,
                    "source_row": row,
                }
            )
    return samples


def detrend_linear(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    if y.size <= 1:
        return y - np.nanmean(y)
    t = np.linspace(-1.0, 1.0, y.size, dtype=np.float32)
    valid = np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.nan_to_num(y - np.nanmean(y), nan=0.0).astype(np.float32)
    slope, intercept = np.polyfit(t[valid], y[valid], deg=1)
    out = y - (slope * t + intercept).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


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


def edacm_phase(z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    series = np.asarray(z, dtype=np.complex64).reshape(-1)
    i = np.real(series).astype(np.float32)
    q = np.imag(series).astype(np.float32)
    di = np.diff(i, prepend=i[:1])
    dq = np.diff(q, prepend=q[:1])
    denom = i * i + q * q + np.float32(eps)
    delta_phase = (i * dq - q * di) / denom
    phase = np.cumsum(delta_phase, dtype=np.float32)
    return detrend_linear(phase)


def select_target_bin_indices(radar: np.ndarray, top_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    energy = np.mean(np.abs(radar) ** 2, axis=0)
    flat = energy.reshape(-1)
    count = min(max(1, int(top_bins)), int(flat.size))
    selected_flat = np.argpartition(flat, -count)[-count:]
    selected_flat = selected_flat[np.argsort(flat[selected_flat])[::-1]]
    selected_multi = np.asarray(np.unravel_index(selected_flat, energy.shape)).T.astype(np.int32)
    weights = flat[selected_flat].astype(np.float32)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.full(count, 1.0 / count, dtype=np.float32)
    else:
        weights = weights / np.sum(weights)
    return selected_multi, weights


def target_edacm_signal(
    radar: np.ndarray,
    top_bins: int = DEFAULT_EDACM_TOP_BINS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    selected_bins, weights = select_target_bin_indices(radar, top_bins=top_bins)
    phases = []
    for d_idx, a_idx, r_idx in selected_bins:
        z = radar[:, int(d_idx), int(a_idx), int(r_idx)]
        phases.append(edacm_phase(z))
    stacked = np.stack(phases, axis=0).astype(np.float32)
    fused = np.sum(stacked * weights[:, None], axis=0)
    fused = zscore_1d(fused)
    meta = {
        "phase_method": "EDACM",
        "edacm_top_bins": int(top_bins),
        "target_bins_dar": selected_bins.tolist(),
        "target_bin_weights": [float(x) for x in weights.tolist()],
        "phase_preprocess": "linear_detrend_zscore",
    }
    return fused.astype(np.float32), meta


def vmd_decompose(
    signal: np.ndarray,
    k: int = DEFAULT_VMD_K,
    alpha: float = DEFAULT_VMD_ALPHA,
    max_iter: int = DEFAULT_VMD_MAX_ITER,
    tol: float = DEFAULT_VMD_TOL,
) -> np.ndarray:
    """Variational mode decomposition for one real-valued fixed-length signal."""
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
    elif representation == "target_edacm_vmd":
        phase, phase_meta = target_edacm_signal(radar)
        x_time = vmd_decompose(phase, k=DEFAULT_VMD_K)
        x_freq, freq_hz, freq_meta = frequency_features(x_time)
        x_rda = rda_log_magnitude(radar)
        phase_meta.update(
            {
                "vmd_k": DEFAULT_VMD_K,
                "vmd_alpha": DEFAULT_VMD_ALPHA,
                "vmd_max_iter": DEFAULT_VMD_MAX_ITER,
                "vmd_tol": DEFAULT_VMD_TOL,
                "vmd_output_shape": list(x_time.shape),
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
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return {"x": x, "meta": {}}


def normalize_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32, copy=False)
    if mode != "window_zscore":
        raise ValueError(f"Unsupported normalize mode: {mode}")
    y = x.astype(np.float32, copy=True)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if np.isfinite(std) and std > 1e-6:
        y = (y - mean) / std
    else:
        y = y - mean
    y[~np.isfinite(y)] = 0.0
    return y


def build_frame_times(time: np.ndarray, n_frames: int) -> np.ndarray:
    ref_time = np.asarray(time, dtype=np.float32).reshape(-1)
    if n_frames <= 0:
        return np.array([], dtype=np.float32)
    if ref_time.size == n_frames:
        return ref_time
    if ref_time.size >= 2:
        start = float(np.nanmin(ref_time))
        end = float(np.nanmax(ref_time))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            return np.linspace(start, end, n_frames, dtype=np.float32)
    return np.arange(n_frames, dtype=np.float32)


def interpolate_label_to_frames(
    label: np.ndarray,
    ref_time: np.ndarray,
    frame_time: np.ndarray,
) -> np.ndarray:
    y = np.asarray(label, dtype=np.float32).reshape(-1)
    t = np.asarray(ref_time, dtype=np.float32).reshape(-1)
    n = min(y.size, t.size)
    if n == 0:
        return np.full(frame_time.shape, np.nan, dtype=np.float32)
    y = y[:n]
    t = t[:n]
    valid = np.isfinite(y) & np.isfinite(t)
    if int(valid.sum()) == 0:
        return np.full(frame_time.shape, np.nan, dtype=np.float32)
    if int(valid.sum()) == 1:
        return np.full(frame_time.shape, float(y[valid][0]), dtype=np.float32)
    order = np.argsort(t[valid])
    valid_t = t[valid][order]
    valid_y = y[valid][order]
    unique_t, unique_idx = np.unique(valid_t, return_index=True)
    unique_y = valid_y[unique_idx]
    if unique_t.size == 1:
        return np.full(frame_time.shape, float(unique_y[0]), dtype=np.float32)
    out = np.interp(frame_time, unique_t, unique_y, left=np.nan, right=np.nan)
    return out.astype(np.float32)


def reduce_label(values: np.ndarray, mode: str) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(valid))
    if mode == "median":
        return float(np.median(valid))
    if mode == "center":
        center = len(values) // 2
        if np.isfinite(values[center]):
            return float(values[center])
        return float(valid[len(valid) // 2])
    raise ValueError(f"Unsupported label reduction: {mode}")


def label_coverage(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).sum() / values.size)


def iter_window_starts(n_frames: int, window_size: int, stride: int) -> List[int]:
    if n_frames < window_size:
        return []
    return list(range(0, n_frames - window_size + 1, stride))


def save_window(
    path: Path,
    feature_bundle: Dict[str, Any],
    labels: Dict[str, float],
    label_series: Dict[str, np.ndarray],
    meta: Dict[str, Any],
) -> None:
    arrays: Dict[str, Any] = {
        "label_heart_rate": np.float32(labels.get("heart_rate", np.nan)),
        "label_series_heart_rate": label_series["heart_rate"].astype(np.float32),
        "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
    }
    for key in ("x", "x_time", "x_freq", "x_rda", "freq_hz"):
        if key in feature_bundle:
            arrays[key] = np.asarray(feature_bundle[key], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite to continue."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "windows").mkdir(parents=True, exist_ok=True)


def build_training_dataset(args: argparse.Namespace) -> None:
    exports_dir = Path(args.exports_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_ratios = validate_split_ratios(args.split_ratios)
    required = required_targets(args.target)

    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not 0.0 <= args.min_label_coverage <= 1.0:
        raise ValueError("--min-label-coverage must be between 0 and 1")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    print_stage("Stage 1/4: collect exported samples")
    samples = collect_exported_samples(exports_dir, args.datasets)
    bgt_long_participants = collect_bgt_long_participants(samples)
    domain_split_map = build_domain_split_map(
        samples=samples,
        ratios=split_ratios,
        seed=args.seed,
        bgt_long_participants=bgt_long_participants,
    )
    print(f"[builder] collected {len(samples)} exported sample(s)", flush=True)
    if bgt_long_participants:
        print(
            (
                f"[builder] BGT60 long participant(s) forced to {DEFAULT_BGT_LONG_SPLIT}: "
                f"{', '.join(sorted(bgt_long_participants))}"
            ),
            flush=True,
        )

    manifest_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "samples_seen": len(samples),
        "samples_used": 0,
        "windows_written": 0,
        "windows_skipped_label": 0,
        "windows_skipped_short": 0,
        "by_dataset": {},
        "by_split": {"train": 0, "val": 0, "test": 0},
        "groups_by_dataset_split": {
            dataset: {"train": [], "val": [], "test": []}
            for dataset in args.datasets
        },
    }

    print_stage("Stage 2/4: build fixed windows")
    for sample_idx, sample in enumerate(samples, start=1):
        dataset = sample["dataset"]
        sample_tag = sample["sample_tag"]
        sample_key = f"{dataset}/{sample_tag}"
        group_key = get_group_key(sample)
        split = domain_split_map[group_key]
        if group_key not in summary["groups_by_dataset_split"][dataset][split]:
            summary["groups_by_dataset_split"][dataset][split].append(group_key)
        npz_path = Path(sample["npz_path"])

        try:
            data = np.load(npz_path, allow_pickle=False)
            radar = data["radar"]
            ref_time = data["time"]
            frame_time = build_frame_times(ref_time, int(radar.shape[0]))
            frame_labels = {
                "heart_rate": interpolate_label_to_frames(
                    data["heart_rate"], ref_time, frame_time
                ),
            }
        except Exception as exc:  # noqa: BLE001
            print_progress(sample_idx, len(samples), f"skip {sample_key}: {exc}")
            continue

        starts = iter_window_starts(int(radar.shape[0]), args.window_size, args.stride)
        if not starts:
            summary["windows_skipped_short"] += 1
            print_progress(sample_idx, len(samples), f"short sample {sample_key}")
            continue
        if args.max_windows_per_sample is not None:
            starts = starts[: args.max_windows_per_sample]

        sample_windows = 0
        dataset_summary = summary["by_dataset"].setdefault(
            dataset,
            {"samples_used": 0, "windows_written": 0, "windows_skipped_label": 0},
        )

        for window_idx, start in enumerate(starts):
            end = start + args.window_size
            labels: Dict[str, float] = {}
            coverage: Dict[str, float] = {}
            keep = True
            for target_name in ("heart_rate",):
                window_values = frame_labels[target_name][start:end]
                coverage[target_name] = label_coverage(window_values)
                labels[target_name] = reduce_label(window_values, args.label_reduction)
            for target_name in required:
                if coverage[target_name] < args.min_label_coverage:
                    keep = False
                if not np.isfinite(labels[target_name]):
                    keep = False
            if not keep:
                summary["windows_skipped_label"] += 1
                dataset_summary["windows_skipped_label"] += 1
                continue

            radar_window = radar[start:end]
            feature_bundle = radar_to_feature_bundle(radar_window, args.representation)
            feature_bundle["x"] = normalize_features(feature_bundle["x"], args.normalize)
            if "x_time" in feature_bundle:
                feature_bundle["x_time"] = normalize_features(feature_bundle["x_time"], args.normalize)
            if "x_freq" in feature_bundle:
                feature_bundle["x_freq"] = normalize_features(feature_bundle["x_freq"], args.normalize)
            if "x_rda" in feature_bundle:
                feature_bundle["x_rda"] = normalize_features(feature_bundle["x_rda"], args.normalize)
            x = np.asarray(feature_bundle["x"], dtype=np.float32)
            x_freq = feature_bundle.get("x_freq")
            x_rda = feature_bundle.get("x_rda")
            feature_meta = feature_bundle.get("meta", {})
            out_name = f"{safe_name(dataset)}__{safe_name(sample_tag)}__w{window_idx:05d}.npz"
            out_path = output_dir / "windows" / dataset / split / out_name
            meta = {
                "dataset": dataset,
                "sample_tag": sample_tag,
                "group_key": group_key,
                "source_npz_path": str(npz_path),
                "domain_split": split,
                "window_start": int(start),
                "window_end": int(end),
                "window_size": int(args.window_size),
                "stride": int(args.stride),
                "representation": args.representation,
                "normalize": args.normalize,
                "label_reduction": args.label_reduction,
                "label_coverage": coverage,
                "feature_meta": feature_meta,
            }
            save_window(
                path=out_path,
                feature_bundle=feature_bundle,
                labels=labels,
                label_series={
                    "heart_rate": frame_labels["heart_rate"][start:end],
                },
                meta=meta,
            )
            manifest_rows.append(
                {
                    "window_path": str(out_path),
                    "dataset": dataset,
                    "sample_tag": sample_tag,
                    "group_key": group_key,
                    "domain_split": split,
                    "split": split,
                    "window_start": int(start),
                    "window_end": int(end),
                    "x_shape": str(tuple(x.shape)),
                    "x_freq_shape": str(tuple(np.asarray(x_freq).shape)) if x_freq is not None else "",
                    "x_rda_shape": str(tuple(np.asarray(x_rda).shape)) if x_rda is not None else "",
                    "label_heart_rate": labels["heart_rate"],
                    "coverage_heart_rate": coverage["heart_rate"],
                }
            )
            sample_windows += 1
            summary["windows_written"] += 1
            summary["by_split"][split] += 1
            dataset_summary["windows_written"] += 1

        if sample_windows > 0:
            summary["samples_used"] += 1
            dataset_summary["samples_used"] += 1
        print_progress(
            sample_idx,
            len(samples),
            f"{sample_key} -> {sample_windows} window(s)",
        )

    print_stage("Stage 3/4: write manifest and config")
    write_manifest(output_dir / "manifest.csv", manifest_rows)
    write_json(
        output_dir / "build_config.json",
        {
            "exports_dir": str(exports_dir),
            "output_dir": str(output_dir),
            "datasets": list(args.datasets),
            "window_size": args.window_size,
            "stride": args.stride,
            "representation": args.representation,
            "target": args.target,
            "required_targets": list(required),
            "label_reduction": args.label_reduction,
            "min_label_coverage": args.min_label_coverage,
            "normalize": args.normalize,
            "edacm_top_bins": DEFAULT_EDACM_TOP_BINS,
            "vmd_k": DEFAULT_VMD_K,
            "vmd_alpha": DEFAULT_VMD_ALPHA,
            "vmd_max_iter": DEFAULT_VMD_MAX_ITER,
            "vmd_tol": DEFAULT_VMD_TOL,
            "sampling_rate_hz": DEFAULT_SAMPLING_RATE_HZ,
            "x_rda_representation": DEFAULT_RDA_REPRESENTATION,
            "output_arrays": ["x", "x_time", "x_freq", "x_rda", "freq_hz"],
            "split_mode": "within_dataset_grouped",
            "grouping": {
                "FTU": "participant_id",
                "BGT60TR13C": "participant_id",
                "PhysDrive": "session_id",
            },
            "bgt_long_split": DEFAULT_BGT_LONG_SPLIT,
            "bgt_long_participants": sorted(bgt_long_participants),
            "split_ratios": list(split_ratios),
            "seed": args.seed,
            "max_windows_per_sample": args.max_windows_per_sample,
        },
    )

    print_stage("Stage 4/4: write summary")
    for dataset, splits in summary["groups_by_dataset_split"].items():
        for split in splits:
            summary["groups_by_dataset_split"][dataset][split] = sorted(splits[split])
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[builder] output_dir: {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    build_training_dataset(args)


if __name__ == "__main__":
    main()
