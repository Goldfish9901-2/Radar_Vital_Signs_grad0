"""mmwave-897-2026 全 cube linear probe（本地全量）。

背景: 单 bin selection / 融合 / 投票全部停在 15-18 BPM;
split-half 证明 oracle 1.64 是选择偏差, 真实上限 ≈15 BPM。
问题: "HR 信息是否存在于别的表示中?" -> 不选 bin, 直接线性探测
全 cube 频谱矩阵能否读出 HR。

表示 (label-free, per session):
    single_bin   max_energy bin 的 HR 带频谱 (64,)           [对照]
    spec_mean    62 bin 频谱平均 (64,)
    spec_max     62 bin 每频点 max (64,)
    spec_top5    62 bin 每频点 top-5 能量和 (64,)
    spec_matrix  62 x 64 频谱矩阵展开 (3968,)
    log_spec_mat log1p 后的 spec_matrix (3968,)

评估: subject-level GroupKFold(5) + Ridge(alpha CV), 报告 R^2 / MAE。

注意 (2026-09 修正): 旧结果 mmwave_probe_results.json 数字不可用, 当时有两个 bug:
    (1) 静默 / zero-energy bin 产生 NaN/inf 特征;
    (2) Ridge 无截距 -> 强正则下退化为预测 0, 报出 R^2=-15 / MAE=77 等假结果
        (小样本时 alpha 未正则化还会爆到 MAE≈387)。
现已修正: 特征 nan_to_num + 零范数保护, 标准化只在训练折拟合(无泄漏),
内层 fold 为空/过小则跳过, Ridge 在训练折上中心化 y(等价截距),
预测裁剪 [30, 200], 并加入 constant 基线作参照。

修正后可信结果（subject-level GroupKFold(5)）:
    mmwave_probe_fixed_full440.json  <- 权威结果, 全量 110 人 x 4 场景 = 440 session
        constant      R^2=0.000  MAE 14.28
        single_bin    R^2=0.176  MAE 12.23   (最好)
        spec_matrix   R^2=0.172  MAE 12.36
        各表示差别很小 (R^2 0.14-0.18) -> 扩大空间/频谱表示未带来进一步收益;
        在所测表示下线性 probe 性能稳定在 R^2≈0.18, 这是 saturation 而非信息论 ceiling
        (12.23 是 linear baseline / reference, 不是数据集的绝对下限, 非线性 NN 可能更好)
    mmwave_probe_fixed_40pax.json    <- 40 人先导, 略偏乐观 (MAE 10.8)
    mmwave_probe_fixed_10pax.json    <- 10 人先导, 样本太小不可信

结论: 线性读数确实有跨 subject 的 HR 信号 (~12.2 bpm), 优于
      FFT 峰估计 15.7 与 split-half 诚实上限 18.3, 但 R^2 仅 ≈0.18。
旧文件 mmwave_probe_results.json 仅作溯源保留, 不可引用。

用法:
    python -m src.data.mmwave_probe --dataset /tmp/mmwave_preview --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.mmwave_loader import MMWaveDataLoader
from src.data.mmwave_baseline import (
    REST_HR_BAND_HZ, POSTEX_HR_BAND_HZ,
    butter_bandpass, extract_phase, hr_from_ecg,
)

EXCLUDE_FIRST = 2
F_GRID = np.arange(0.5, 4.01, 0.05)          # 71 个频点
ALPHAS = [1e-3, 1e-1, 10.0, 1e3]


def bin_spectrum(radar: np.ndarray, b: int, fs: float,
                 hr_band: Tuple[float, float]) -> np.ndarray:
    """单 bin 在 F_GRID 上的 HR 带幅度谱 (归一化)。"""
    cell = radar[:, :, b].mean(axis=1)
    ph = np.unwrap(np.angle(cell))
    ph = ph - ph.mean()
    filtered = butter_bandpass(ph, fs, hr_band)
    n = len(filtered)
    from scipy.signal import get_window
    windowed = filtered * get_window('hann', n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.abs(np.fft.rfft(windowed, nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.interp(F_GRID, freqs, spec)
    amp = np.log1p(amp)
    nrm = np.linalg.norm(amp)
    if not np.isfinite(nrm) or nrm < 1e-12:
        return np.zeros_like(amp)
    return np.nan_to_num(amp / nrm, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize(vec: np.ndarray) -> np.ndarray:
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(radar: np.ndarray, fs: float,
                   hr_band: Tuple[float, float],
                   rep: str) -> np.ndarray:
    n_bins = radar.shape[2]
    idx = np.arange(EXCLUDE_FIRST, n_bins)
    specs = np.stack([bin_spectrum(radar, b, fs, hr_band) for b in idx])
    n_b = specs.shape[0]                       # 62
    vec: np.ndarray
    if rep == 'single_bin':
        energy = np.abs(radar[:, :, EXCLUDE_FIRST:]).mean(axis=(0, 1)) ** 2
        vec = specs[int(np.argmax(energy))]
    elif rep == 'spec_mean':
        vec = specs.mean(axis=0)
    elif rep == 'spec_max':
        vec = specs.max(axis=0)
    elif rep == 'spec_top5':
        top = np.sort(specs, axis=0)[-5:, :]
        vec = top.sum(axis=0)
    elif rep == 'spec_matrix':
        vec = specs.flatten()
    elif rep == 'log_spec_mat':
        vec = np.log1p(specs + 1e-12).flatten()
    else:
        raise ValueError(rep)
    return _sanitize(vec)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def group_folds(groups: np.ndarray, n_folds: int = 5):
    """按 group(participant) 划分 folds, 保证同 participant 不跨 fold。"""
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    for i in range(n_folds):
        test_mask = np.isin(groups, folds[i])
        yield ~test_mask, test_mask


def evaluate(loader, participants) -> dict:
    # 收集特征
    reps = ['single_bin', 'spec_mean', 'spec_max', 'spec_top5',
            'spec_matrix', 'log_spec_mat']
    feats = {r: [] for r in reps}
    targets = []
    groups = []
    for pid in participants:
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                if not (loader._participant_dir(pid) / posture / condition).exists():
                    continue
                radar = loader.load_radar_data(pid, posture, condition)
                ref = loader.load_reference_data(pid, posture, condition)
                fs = loader.FRAME_RATE_HZ
                hr_band = (POSTEX_HR_BAND_HZ if condition == 'Post-exercise'
                           else REST_HR_BAND_HZ)
                hr_ecg, _, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
                if not np.isfinite(hr_ecg):
                    continue
                for r in reps:
                    feats[r].append(build_features(radar, fs, hr_band, r))
                targets.append(hr_ecg)
                groups.append(pid)
        if pid % 20 == 0:
            print(f"P{pid:03d} features done", flush=True)

    y = np.array(targets, dtype=float)
    groups = np.array(groups)
    n = len(y)
    print(f"n_samples={n} n_groups={len(np.unique(groups))} "
          f"y_mean={y.mean():.1f} y_std={y.std():.1f}", flush=True)

    def _r2(pred, yt):
        v = np.var(yt)
        if v <= 1e-9:
            return 0.0
        return float(1 - np.mean((pred - yt) ** 2) / v)

    results = {}
    # 常数基线 (预测训练集均值) 作为参照
    const_pred = np.full(n, y.mean())
    results['constant'] = {
        'r2': _r2(const_pred, y),
        'mae_bpm': float(np.mean(np.abs(const_pred - y))),
        'best_alpha': None, 'n_features': 0,
    }
    print(f"{'constant':<15} R2={results['constant']['r2']:.4f} "
          f"MAE={results['constant']['mae_bpm']:.2f}", flush=True)

    for r in reps:
        X = _sanitize(np.stack(feats[r]))
        # 外循环: subject-level GroupKFold(5), 每折内层 CV 选 alpha
        # 标准化只在训练折上拟合 (避免泄漏); 预测裁剪到合理 HR 范围
        preds = np.full(n, np.nan)
        for tr_m, te_m in group_folds(groups, 5):
            trX, try_ = X[tr_m], y[tr_m]
            ytr_mean = try_.mean()
            yb = try_ - ytr_mean
            mu = trX.mean(axis=0)
            sd = trX.std(axis=0) + 1e-12
            trXs = (trX - mu) / sd
            best_alpha, best_cv = ALPHAS[0], -np.inf
            for a in ALPHAS:
                scores = []
                for tr2, te2 in group_folds(groups[tr_m], 5):
                    if tr2.sum() < 2 or te2.sum() < 1:
                        continue
                    b = ridge_fit(trXs[tr2], yb[tr2], a)
                    pred = trXs[te2] @ b + ytr_mean
                    scores.append(_r2(pred, try_[te2]))
                if not scores:
                    continue
                m = float(np.mean(scores))
                if m > best_cv:
                    best_cv, best_alpha = m, a
            b = ridge_fit(trXs, yb, best_alpha)
            teXs = (X[te_m] - mu) / sd
            preds[te_m] = np.clip(teXs @ b + ytr_mean, 30.0, 200.0)
        r2 = _r2(preds, y)
        mae = float(np.mean(np.abs(preds - y)))
        results[r] = {
            'r2': float(r2), 'mae_bpm': mae,
            'best_alpha': best_alpha, 'n_features': X.shape[1],
        }
        print(f"{r:<15} R2={r2:.4f} MAE={mae:.2f} alpha={best_alpha}",
              flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="全 cube linear probe")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path,
                        default="/tmp/mmwave_probe_results.json")
    args = parser.parse_args()

    loader = MMWaveDataLoader(str(args.dataset))
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]

    results = evaluate(loader, participants)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果写入: {args.output}")


if __name__ == '__main__':
    main()
