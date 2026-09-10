"""RDA-domain clutter suppression ablation on FTU / BGT60 / PhysDrive.

B 主线: 4 个 clutter 方法 × 严格固定的下游 pipeline:
    RDA (T,D,A,R) complex
      -> suppress_clutter(method)   # current / mti / highpass / pca
      -> selection:  max-energy range bin (在 clutter-suppressed cube 上)
      -> complex_mean over (D,A)   # 空间聚焦
      -> unwrap
      -> Butterworth bandpass(0.7-2.5 Hz) -> Hann -> rfft 峰 -> BPM
A 附加: whole-window FFT vs segmented FFT (10s 窗, 5s 步长) HR 轨迹估计。

用法:
    .venv/bin/python -m src.data.rda_clutter_baseline --dataset FTU --clutter all
    .venv/bin/python -m src.data.rda_clutter_baseline --dataset all --clutter all --no-seg
输出:
    experiments/rda_clutter/{dataset}/{clutter}_{window}.json   (per-sample + 聚合)
    experiments/rda_clutter/summary_{dataset}.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = Path("/srv/ext-timeshift/radar_vital_signs/exports")

CLUTTER_METHODS = ("current", "mti", "highpass", "pca")
DATASETS = ("FTU", "BGT60TR13C", "PhysDrive")
HR_BAND = (0.7, 2.5)  # Hz -> 42-150 BPM
SEG_WINDOW_S = 10.0
SEG_STEP_S = 5.0
SEG_GT_HALF_WIN_S = 2.5  # label 对齐窗口 ±2.5s


# --------------------------------------------------------------------------- #
# Clutter suppression (slow-time 域, 作用于复数 RDA cube)
# --------------------------------------------------------------------------- #
def suppress_clutter(radar: np.ndarray, method: str, fs: float) -> np.ndarray:
    """radar: (T, D, A, R) complex。返回相同形状。"""
    if method == "current":
        return radar
    if method == "mti":
        # slow-time mean subtraction: 对每个 (d,a,r) cell 移除时间均值 (DC/静态杂波)
        return radar - radar.mean(axis=0, keepdims=True)
    if method == "highpass":
        # 沿 slow-time 对每个 cell 做 0.5Hz 高通 (移除 DC + 呼吸低频 + 漂移)
        sos = signal.butter(4, 0.5, btype="highpass", fs=fs, output="sos")
        T = radar.shape[0]
        x = radar.reshape(T, -1)
        xc = signal.sosfiltfilt(sos, x, axis=0)
        return xc.reshape(radar.shape)
    if method == "pca":
        # 移除前 K 个左奇异向量投影: dominant static / spatially-coherent components
        K = 2
        T = radar.shape[0]
        x = radar.reshape(T, -1)
        xc = x - x.mean(axis=0, keepdims=True)
        U, _, _ = np.linalg.svd(xc, full_matrices=False)
        proj = U[:, :K] @ (U[:, :K].conj().T @ xc)
        return (xc - proj).reshape(radar.shape)
    raise ValueError(f"unknown clutter method: {method}")


# --------------------------------------------------------------------------- #
# Fixed downstream pipeline
# --------------------------------------------------------------------------- #
def select_range_bin(radar: np.ndarray, exclude_first: int = 2) -> Tuple[int, np.ndarray]:
    """max-energy range bin (对 (D,A) 求平均后的 range 能量)。

    与 mmwave_baseline.max_energy 一致: 排除前 exclude_first 个 DC 伪影 bin
    (FTU/BGT60/PhysDrive 均观察到 bin 0/1 为 DC 耦合)。
    """
    energy = np.mean(np.abs(radar) ** 2, axis=(0, 1, 2))  # (R,)
    return int(np.argmax(energy[exclude_first:]) + exclude_first), energy


def extract_phase(radar: np.ndarray, r: int) -> np.ndarray:
    """complex_mean over (D,A) -> unwrap。返回 (T,) 相位序列。"""
    cell = radar[:, :, :, r]  # (T, D, A)
    z = cell.mean(axis=(1, 2))
    phase = np.unwrap(np.angle(z))
    return phase - phase.mean()


def hr_from_phase(phase: np.ndarray, fs: float, band: Tuple[float, float] = HR_BAND) -> float:
    """Butterworth bandpass -> Hann -> rfft -> band 内最强峰 -> BPM。"""
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


def hr_sequence_seg(
    phase: np.ndarray, fs: float, window_s: float = SEG_WINDOW_S, step_s: float = SEG_STEP_S
) -> Tuple[np.ndarray, np.ndarray]:
    """分段 FFT HR 轨迹。返回 (窗中心时间, 估计 BPM)。"""
    n = phase.size
    wlen = int(window_s * fs)
    wstep = int(step_s * fs)
    if n < wlen:
        return np.array([n / 2.0 / fs]), np.array([hr_from_phase(phase, fs)])
    tcs, ests = [], []
    for s in range(0, n - wlen + 1, wstep):
        seg = phase[s : s + wlen]
        bpm = hr_from_phase(seg, fs)
        tcs.append((s + wlen / 2.0) / fs)
        ests.append(bpm)
    return np.asarray(tcs), np.asarray(ests)


# --------------------------------------------------------------------------- #
# Reference 对齐
# --------------------------------------------------------------------------- #
def seq_mae(est: float, label: np.ndarray) -> float:
    """whole-window: 单一估计 vs label 全段均值。"""
    lab = label[np.isfinite(label)]
    if lab.size == 0 or not np.isfinite(est):
        return float("nan")
    return float(np.mean(np.abs(est - lab)))


def traj_mae(tcs: np.ndarray, ests: np.ndarray, t_label: np.ndarray, label: np.ndarray) -> float:
    """segmented: 每窗估计 vs 窗中心 ±2.5s 内 label 均值。"""
    errs = []
    for tc, e in zip(tcs, ests):
        if not np.isfinite(e):
            continue
        m = (np.abs(t_label - tc) <= SEG_GT_HALF_WIN_S) & np.isfinite(label)
        if m.sum() == 0:
            continue
        errs.append(np.abs(e - np.mean(label[m])))
    if not errs:
        return float("nan")
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_sample(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    radar = np.asarray(d["radar"], dtype=np.complex64)
    t = np.asarray(d["time"], dtype=np.float64)
    hr = np.asarray(d["heart_rate"], dtype=np.float64)
    return radar, t, hr


def dataset_fs(name: str) -> float:
    return {"FTU": 20.0, "BGT60TR13C": 30.0, "PhysDrive": 20.0}[name]


def parse_strata(dataset: str, filename: str) -> str:
    if dataset == "FTU":
        # p01_Angle_0_deg__80_cm_r1 -> distance "80_cm"
        m = re.search(r"__(\d+_cm)_r\d", filename)
        return m.group(1) if m else "unknown"
    if dataset == "BGT60TR13C":
        # p01_0_3m_short -> "0_3m"
        m = re.search(r"_(\d_\dm)_", filename)
        return m.group(1) if m else "unknown"
    # PhysDrive: AFH1_000 -> "AFH1"
    return filename.split("_")[0]


def iter_samples(dataset: str, limit: Optional[int]) -> List[Path]:
    d = EXPORTS / dataset / "samples"
    files = sorted(d.glob("*.npz"))
    if limit:
        files = files[:limit]
    return files


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_dataset(dataset: str, clutter_methods: List[str], do_seg: bool, limit: Optional[int]):
    outdir = ROOT / "experiments" / "rda_clutter" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    fs = dataset_fs(dataset)
    files = iter_samples(dataset, limit)
    print(f"[{dataset}] {len(files)} samples, fs={fs}Hz, clutter={clutter_methods}, seg={do_seg}", flush=True)

    # results[clutter] = {strata: {mae_whole[], mae_seg[], ...}}
    results: Dict[str, Dict[str, Dict[str, list]]] = {
        m: {} for m in clutter_methods
    }
    meta = {"dataset": dataset, "fs": fs, "clutter": clutter_methods,
            "window_s": SEG_WINDOW_S, "step_s": SEG_STEP_S}

    t0 = time.monotonic()
    for i, f in enumerate(files, start=1):
        radar, t, hr = load_sample(f)
        strata = parse_strata(dataset, f.stem)
        for m in clutter_methods:
            r = results[m].setdefault(strata, {"mae_whole": [], "mae_seg": [], "sel_r": []})
            cube = suppress_clutter(radar, m, fs)
            rb, _ = select_range_bin(cube)
            r["sel_r"].append(rb)
            phase = extract_phase(cube, rb)
            est_whole = hr_from_phase(phase, fs)
            r["mae_whole"].append(seq_mae(est_whole, hr))
            if do_seg:
                tcs, ests = hr_sequence_seg(phase, fs)
                r["mae_seg"].append(traj_mae(tcs, ests, t, hr))
        if i % 50 == 0 or i == len(files):
            el = time.monotonic() - t0
            print(f"  [{dataset}] {i}/{len(files)}  elapsed={el:.0f}s  eta={el/i*(len(files)-i):.0f}s", flush=True)

    # 聚合
    summary = dict(meta)
    for m in clutter_methods:
        pooled_w, pooled_s, sel = [], [], []
        by_strata = {}
        for s, rr in results[m].items():
            w = np.array(rr["mae_whole"], dtype=float)
            sg = np.array(rr["mae_seg"], dtype=float)
            pooled_w.extend(w[np.isfinite(w)].tolist())
            pooled_s.extend(sg[np.isfinite(sg)].tolist())
            sel.extend(rr["sel_r"])
            by_strata[s] = {
                "n": int(len(rr["mae_whole"])),
                "mae_whole_bpm": round(float(np.nanmean(w)), 3),
                "mae_seg_bpm": round(float(np.nanmean(sg)), 3) if do_seg else None,
            }
        summary[m] = {
            "n": int(len(pooled_w)),
            "mae_whole_bpm": round(float(np.mean(pooled_w)), 3),
            "mae_seg_bpm": round(float(np.mean(pooled_s)), 3) if do_seg else None,
            "sel_range_bin_mean": round(float(np.mean(sel)), 2) if sel else None,
            "sel_range_bin_std": round(float(np.std(sel)), 2) if sel else None,
            "by_strata": by_strata,
        }

    with open(outdir / f"summary_{dataset}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{dataset}] summary -> {outdir / ('summary_' + dataset + '.json')}")
    print(json.dumps({m: {k: v for k, v in summary[m].items() if k != 'by_strata'}
                      for m in clutter_methods}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    ap.add_argument("--clutter", choices=list(CLUTTER_METHODS) + ["all"], default="all")
    ap.add_argument("--no-seg", action="store_true", help="跳过 segmented FFT")
    ap.add_argument("--limit", type=int, default=None, help="限制样本数(测试)")
    args = ap.parse_args()

    methods = list(CLUTTER_METHODS) if args.clutter == "all" else [args.clutter]
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    do_seg = not args.no_seg
    for ds in datasets:
        run_dataset(ds, methods, do_seg=do_seg, limit=args.limit)


if __name__ == "__main__":
    main()
