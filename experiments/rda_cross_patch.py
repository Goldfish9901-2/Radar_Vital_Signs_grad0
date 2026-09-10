"""补齐 clutter × range 交叉矩阵缺失的 3 个 cell: (mti|highpass|pca) × global_center.

复用两轮 kernel 完全一致的 metric:
  RDA (T,D,A,R) complex
    -> suppress_clutter(method)          # 与 rda_clutter_kernel 相同
    -> global_center: complex_mean over (D,A, all non-DC R)  # 与 experiment.py 相同
    -> unwrap
    -> Butterworth bandpass(0.7-2.5 Hz) -> Hann -> rfft 峰 -> BPM
    -> MAE vs label 全段均值

已有 cell 来自两轮 kernel (直接合并, 不重算):
  (current|mti|highpass|pca) × peak      <- rda_clutter_suppression_ablation
  current × global_center                <- range_selection_ablation

本脚本只输出缺失 3 cell 的 summary, 之后由 rda_cross_analyze.py 合并全部 8 cell。
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.sparse.linalg import svds

EXPORTS = {
    "FTU": Path("/srv/ext-timeshift/radar_vital_signs/exports/FTU/samples"),
    "BGT60TR13C": Path("/srv/ext-timeshift/radar_vital_signs/exports/BGT60TR13C/samples"),
    "PhysDrive": Path("/srv/ext-timeshift/radar_vital_signs/exports/PhysDrive/samples"),
}
OUT = Path("/srv/ext-timeshift/radar_vital_signs/run/upstream/experiments/rda_clutter/global_center_patch")
HR_BAND = (0.7, 2.5)
EXCLUDE_FIRST = 2
METHODS = ("mti", "highpass", "pca")


def dataset_fs(name: str) -> float:
    return {"FTU": 20.0, "BGT60TR13C": 30.0, "PhysDrive": 20.0}[name]


def load_sample(path: Path):
    d = np.load(path, allow_pickle=True)
    return (
        np.asarray(d["radar"], dtype=np.complex64),
        np.asarray(d["time"], dtype=np.float64),
        np.asarray(d["heart_rate"], dtype=np.float64),
    )


def suppress_clutter(radar: np.ndarray, method: str, fs: float) -> np.ndarray:
    if method == "current":
        return radar
    if method == "mti":
        return radar - radar.mean(axis=0, keepdims=True)
    if method == "highpass":
        sos = signal.butter(4, 0.5, btype="highpass", fs=fs, output="sos")
        T = radar.shape[0]
        x = radar.reshape(T, -1)
        xc = signal.sosfiltfilt(sos, x, axis=0)
        return xc.reshape(radar.shape)
    if method == "pca":
        # svds: 只算 top-2 左奇异向量 (与 full SVD top-K 投影等价, 快 ~100x)
        K = 2
        T = radar.shape[0]
        x = radar.reshape(T, -1)
        xc = x - x.mean(axis=0, keepdims=True)
        if min(xc.shape) <= K:
            return (xc - xc.mean(axis=0, keepdims=True)).reshape(radar.shape)
        U, s, _ = svds(xc, k=K)
        order = np.argsort(s)[::-1]  # svds 不保证降序
        U = U[:, order]
        proj = U @ (U.conj().T @ xc)
        return (xc - proj).reshape(radar.shape)
    raise ValueError(method)


def extract_phase_global_center(radar: np.ndarray) -> np.ndarray:
    z = radar[:, :, :, EXCLUDE_FIRST:].mean(axis=(1, 2, 3))  # (T,)
    phase = np.unwrap(np.angle(z))
    return phase - phase.mean()


def hr_from_phase(phase: np.ndarray, fs: float, band: Tuple[float, float] = HR_BAND) -> float:
    if phase.size < 8:
        return float("nan")
    sos = signal.butter(4, list(band), btype="bandpass", fs=fs, output="sos")
    pf = signal.sosfiltfilt(sos, phase)
    w = np.hanning(phase.size)
    spec = np.abs(np.fft.rfft(pf * w))
    freqs = np.fft.rfftfreq(phase.size, d=1.0 / fs)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return float("nan")
    f_peak = freqs[m][np.argmax(spec[m])]
    return float(f_peak * 60.0)


def _process_one(args):
    """(dataset, fs, file_path) -> dict {method: mae}"""
    dataset, fs, fpath = args
    radar, t, hr = load_sample(Path(fpath))
    out = {}
    for m in METHODS:
        cube = suppress_clutter(radar, m, fs)
        ph = extract_phase_global_center(cube)
        est = hr_from_phase(ph, fs)
        lab = hr[np.isfinite(hr)]
        mae = float(np.mean(np.abs(est - lab))) if lab.size and np.isfinite(est) else None
        if mae is not None:
            out[m] = mae
    return out


def run_dataset(dataset: str, methods: List[str], limit: int | None = None, nproc: int = 4):
    fs = dataset_fs(dataset)
    files = sorted(EXPORTS[dataset].glob("*.npz"))
    if limit:
        files = files[:limit]
    print(f"[{dataset}] {len(files)} samples fs={fs}Hz methods={methods} nproc={nproc}", flush=True)

    tasks = [(dataset, fs, str(f)) for f in files]
    t0 = time.monotonic()
    with Pool(nproc) as pool:
        results_by_method: dict = {m: [] for m in methods}
        for i, res in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=16), start=1):
            for m in methods:
                if m in res:
                    results_by_method[m].append(res[m])
            if i % 200 == 0 or i == len(files):
                el = time.monotonic() - t0
                print(f"  {i}/{len(files)} elapsed={el:.0f}s eta={el/i*(len(files)-i):.0f}s", flush=True)

    summary = {"dataset": dataset, "fs": fs, "n": len(files)}
    for m, maes in results_by_method.items():
        summary[m] = {
            "mae_bpm": round(float(np.mean(maes)), 3),
            "n": len(maes),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"summary_{dataset}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["FTU", "BGT60TR13C", "PhysDrive"]
    for ds in datasets:
        run_dataset(ds, list(METHODS))


if __name__ == "__main__":
    main()
