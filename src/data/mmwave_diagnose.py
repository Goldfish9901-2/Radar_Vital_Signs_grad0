"""mmwave-897-2026 本地诊断：per-bin HR 质量图 vs label-free 特征。

回答三个问题:
    Q1: oracle bin 均匀分布是"信号散布在所有 bin"还是"每 session 只有一个真实 bin 且位置多变"?
        -> per-bin |err| < 3 bpm 的 good bins 数量分布 + 聚集性(连续区间长度)
    Q2: label-free 特征(能量/HR带能量/相位方差)能否预测 good bins?
        -> good bin 在 energy top-K / hr_band top-K 中的命中率
    Q3: bin 2 (0.63m) 为何总是 max energy? 是 DC 耦合伪影吗?
        -> bin 2 相对其余 bin 的能量比, 去均值(MTI)后是否改变

用法:
    python -m src.data.mmwave_diagnose --dataset /tmp/mmwave_preview --all
    python -m src.data.mmwave_diagnose --dataset /tmp/mmwave_preview --participants 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.mmwave_loader import MMWaveDataLoader
from src.data.mmwave_baseline import (
    REST_HR_BAND_HZ, POSTEX_HR_BAND_HZ,
    butter_bandpass, extract_phase, hr_from_ecg, hr_from_phase,
)

GOOD_ERR_BPM = 3.0          # good bin 阈值
EXCLUDE_FIRST = 2           # 与 baseline 一致


def per_bin_hr(radar: np.ndarray, fs: float,
               hr_band: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """所有 bin 的 HR 估计与 HR 带能量。返回 (hrs (64,), band_power (64,))。"""
    n = radar.shape[2]
    hrs = np.full(n, np.nan)
    powers = np.full(n, np.nan)
    for i in range(n):
        ph = extract_phase(radar, i)
        h, _, _ = hr_from_phase(ph, fs, hr_band)
        hrs[i] = h
        # HR 带能量: 直接带通后方差近似
        bp = butter_bandpass(ph, fs, hr_band)
        powers[i] = float(np.var(bp))
    return hrs, powers


def analyze_session(loader: MMWaveDataLoader, pid: int, posture: str,
                    condition: str) -> dict:
    radar = loader.load_radar_data(pid, posture, condition)
    ref = loader.load_reference_data(pid, posture, condition)
    fs = loader.FRAME_RATE_HZ
    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ
    n_bins = radar.shape[2]

    # 参考真值
    hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
    if not np.isfinite(hr_ecg):
        return {'participant': pid, 'posture': posture, 'condition': condition,
                'skip': True, 'reason': 'no_ecg_hr'}

    hrs, powers = per_bin_hr(radar, fs, hr_band)
    errs = np.abs(hrs - hr_ecg)
    good = errs < GOOD_ERR_BPM
    good &= np.arange(n_bins) >= EXCLUDE_FIRST

    # 能量特征 (全时间平均, 与 baseline 一致)
    energy = np.abs(radar).mean(axis=(0, 1)) ** 2
    e_rank = np.argsort(energy)[::-1]

    # --- Q1: good bins 的聚集性 ---
    good_idx = np.where(good)[0]
    n_good = len(good_idx)
    # 连续区间
    runs: List[List[int]] = []
    for g in good_idx:
        if runs and g == runs[-1][-1] + 1:
            runs[-1].append(g)
        else:
            runs.append([g])
    max_run = max((len(r) for r in runs), default=0)
    n_runs = len(runs)

    # --- Q2: label-free 命中率 ---
    def hit_rate(candidates: np.ndarray) -> float:
        if len(candidates) == 0:
            return 0.0
        return float(np.mean(good[candidates]))

    topk = max(8, n_bins // 4)
    e_top = e_rank[e_rank >= EXCLUDE_FIRST][:topk]
    p_rank = np.argsort(powers)[::-1]
    p_top = p_rank[p_rank >= EXCLUDE_FIRST][:topk]

    # 组合: 能量 top-K 中再按 HR 带能量排序的前 N 个
    combo = e_top[np.argsort(powers[e_top])[::-1]]
    combo_top8 = combo[:8]

    # --- Q3: bin 2 能量占比 + MTI 效果 ---
    e_bin2 = energy[2] if n_bins > 2 else 0.0
    e_med = float(np.median(energy[EXCLUDE_FIRST:]))
    # 慢时间去均值 (MTI) 后的能量
    cell0 = radar[:, :, :]  # (frames, ant, bins)
    mti = cell0 - cell0.mean(axis=0, keepdims=True)
    e_mti = np.abs(mti).mean(axis=(0, 1)) ** 2
    e_mti_rank = np.argsort(e_mti)[::-1]
    e_mti_top = e_mti_rank[e_mti_rank >= EXCLUDE_FIRST][:topk]

    return {
        'participant': pid, 'posture': posture, 'condition': condition,
        'skip': False,
        'hr_ecg_bpm': float(hr_ecg), 'n_ecg_peaks': int(n_peaks),
        'n_frames': int(radar.shape[0]),
        # Q1
        'n_good_bins': int(n_good), 'max_run': int(max_run), 'n_runs': int(n_runs),
        'good_bins': good_idx.tolist(),
        # Q2
        'hit_energy_topk': hit_rate(e_top),
        'hit_power_topk': hit_rate(p_top),
        'hit_combo_top8': hit_rate(combo_top8),
        'best_energy_bin': int(e_top[0]),
        'best_energy_is_good': bool(good[e_top[0]]),
        'best_power_bin_is_good': bool(good[p_top[0]]),
        # Q3
        'energy_bin2_ratio': float(e_bin2 / (e_med + 1e-12)),
        'bin2_rank': int(np.where(e_rank == 2)[0][0]) if 2 in e_rank else -1,
        'hit_mti_topk': hit_rate(e_mti_top),
        'oracle_err': float(errs[np.nanargmin(errs)]),
    }


def summarize(results: List[dict]) -> dict:
    valid = [r for r in results if not r.get('skip')]
    n = len(valid)

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in valid])) if n else float('nan')

    def pct(key: str) -> float:
        return 100.0 * mean(key) if n else float('nan')

    # 按场景分层
    layers = {}
    for posture in ('Lying', 'Sitting'):
        for condition in ('Rest', 'Post-exercise'):
            sub = [r for r in valid if r['posture'] == posture
                   and r['condition'] == condition]
            layers[f'{posture}/{condition}'] = {
                'n': len(sub),
                'n_good_bins': float(np.mean([r['n_good_bins'] for r in sub])) if sub else float('nan'),
                'hit_energy_topk': 100 * float(np.mean([r['hit_energy_topk'] for r in sub])) if sub else float('nan'),
                'hit_power_topk': 100 * float(np.mean([r['hit_power_topk'] for r in sub])) if sub else float('nan'),
                'hit_combo_top8': 100 * float(np.mean([r['hit_combo_top8'] for r in sub])) if sub else float('nan'),
                'hit_mti_topk': 100 * float(np.mean([r['hit_mti_topk'] for r in sub])) if sub else float('nan'),
                'oracle_err': float(np.mean([r['oracle_err'] for r in sub])) if sub else float('nan'),
            }

    return {
        'n_sessions': len(results), 'n_valid': n,
        'Q1_good_bins': {
            'mean': mean('n_good_bins'), 'median': float(np.median(
                [r['n_good_bins'] for r in valid])) if n else float('nan'),
            'max_run_mean': mean('max_run'), 'n_runs_mean': mean('n_runs'),
            'pct_sessions_no_good_bin': 100 * float(np.mean(
                [r['n_good_bins'] == 0 for r in valid])) if n else float('nan'),
        },
        'Q2_label_free_hit_rate': {
            'energy_topk': pct('hit_energy_topk'),
            'hrband_power_topk': pct('hit_power_topk'),
            'combo_energy_x_hrband_top8': pct('hit_combo_top8'),
            'mti_energy_topk': pct('hit_mti_topk'),
            'best_energy_bin_good': pct('best_energy_is_good'),
            'best_power_bin_good': pct('best_power_bin_is_good'),
        },
        'Q3_bin2_artifact': {
            'energy_bin2_over_median': mean('energy_bin2_ratio'),
            'bin2_is_global_argmax_pct': 100 * float(np.mean(
                [r['bin2_rank'] == 0 for r in valid])) if n else float('nan'),
        },
        'oracle_err_mean': mean('oracle_err'),
        'by_stratum': layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="mmwave per-bin 诊断")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    loader = MMWaveDataLoader(str(args.dataset))
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]

    results = []
    for pid in participants:
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                if not (loader._participant_dir(pid) / posture / condition).exists():
                    continue
                try:
                    r = analyze_session(loader, pid, posture, condition)
                except Exception as e:  # noqa: BLE001
                    r = {'participant': pid, 'posture': posture,
                         'condition': condition, 'skip': True, 'reason': str(e)}
                results.append(r)
                flag = '' if not r.get('skip') else ' (skip)'
                print(f"P{pid:03d} {posture}/{condition}: "
                      f"good_bins={r.get('n_good_bins', '?')} "
                      f"run={r.get('max_run', '?')} "
                      f"hitE={r.get('hit_energy_topk', 0):.2f} "
                      f"hitP={r.get('hit_power_topk', 0):.2f}"
                      f"{flag}", flush=True)

    summ = summarize(results)
    print("\n===== 诊断汇总 =====")
    print(json.dumps(summ, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'summary': summ, 'sessions': results}, f, indent=2,
                      ensure_ascii=False)
        print(f"\n结果写入: {args.output}")


if __name__ == '__main__':
    main()
