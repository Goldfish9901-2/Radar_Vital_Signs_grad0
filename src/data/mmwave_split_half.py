"""mmwave-897-2026 split-half oracle 验证（本地全量）。

假设: oracle = 62 bin 各试一次取误差最小, 可能是多重比较偏差。
验证: 前一半帧选 bin -> 后一半帧验证 HR。
    - 若 bin 是时间上稳定的真实信息: split-half oracle ≈ 全段 oracle
    - 若 oracle 只是"碰巧"选中: split-half oracle 显著变差

同时报告:
    - good bins (|err|<3bpm) 在前后两半的重叠率 (空间稳定性)
    - per-session 全段 oracle 与 split-half oracle 的分布

用法:
    python -m src.data.mmwave_split_half --dataset /tmp/mmwave_preview --all
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
    extract_phase, hr_from_ecg, hr_from_phase,
)

GOOD_ERR = 3.0


def per_bin_hr(radar: np.ndarray, fs: float,
               hr_band: Tuple[float, float]) -> np.ndarray:
    n = radar.shape[2]
    hrs = np.full(n, np.nan)
    for i in range(n):
        ph = extract_phase(radar, i)
        h, _, _ = hr_from_phase(ph, fs, hr_band)
        hrs[i] = h
    return hrs


def analyze(loader, pid, posture, condition) -> dict:
    radar = loader.load_radar_data(pid, posture, condition)
    ref = loader.load_reference_data(pid, posture, condition)
    fs = loader.FRAME_RATE_HZ
    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ

    hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
    if not np.isfinite(hr_ecg):
        return {'skip': True, 'participant': pid, 'posture': posture,
                'condition': condition}

    n_frames = radar.shape[0]
    half = max(30, n_frames // 2)
    radar_A = radar[:half]
    radar_B = radar[half:]

    hrs_full = per_bin_hr(radar, fs, hr_band)
    hrs_A = per_bin_hr(radar_A, fs, hr_band)
    errs_full = np.abs(hrs_full - hr_ecg)
    errs_A = np.abs(hrs_A - hr_ecg)

    valid = np.arange(2, radar.shape[2])
    b_full = valid[np.nanargmin(errs_full[valid])]
    b_A = valid[np.nanargmin(errs_A[valid])]

    # 用 A 中选中的 bin 在 B 中验证
    ph_B = extract_phase(radar_B, int(b_A))
    hr_B, _, _ = hr_from_phase(ph_B, fs, hr_band)
    err_split = abs(hr_B - hr_ecg)

    # good bins 空间重叠
    good_A = set(np.where(errs_A < GOOD_ERR)[0])
    good_B = set(np.where(np.abs(per_bin_hr(radar_B, fs, hr_band)
                                 - hr_ecg) < GOOD_ERR)[0])
    inter = len(good_A & good_B)
    union = len(good_A | good_B) or 1

    return {
        'skip': False,
        'participant': pid, 'posture': posture, 'condition': condition,
        'hr_ecg_bpm': float(hr_ecg),
        'n_frames': int(n_frames), 'half': int(half),
        'oracle_full_err': float(errs_full[b_full]),
        'oracle_A_err': float(errs_A[b_A]),
        'split_half_err': float(err_split),
        'b_full': int(b_full), 'b_A': int(b_A),
        'n_good_A': int(len(good_A)), 'n_good_B': int(len(good_B)),
        'jaccard_good': float(inter / union),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="split-half oracle 验证")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path,
                        default="/tmp/mmwave_split_half_results.json")
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
                    r = analyze(loader, pid, posture, condition)
                except Exception as e:  # noqa: BLE001
                    r = {'skip': True, 'participant': pid, 'posture': posture,
                         'condition': condition, 'reason': str(e)}
                results.append(r)
        if pid % 20 == 0:
            print(f"P{pid:03d} done", flush=True)

    valid = [r for r in results if not r['skip']]
    n = len(valid)
    full = np.array([r['oracle_full_err'] for r in valid])
    A = np.array([r['oracle_A_err'] for r in valid])
    split = np.array([r['split_half_err'] for r in valid])
    jac = np.array([r['jaccard_good'] for r in valid])

    summary = {
        'n_sessions': len(results), 'n_valid': n,
        'oracle_full_mae': float(full.mean()),
        'oracle_A_mae': float(A.mean()),
        'split_half_mae': float(split.mean()),
        'split_vs_full_ratio': float(split.mean() / (full.mean() + 1e-12)),
        'pct_split_within_3bpm': float(np.mean(split < GOOD_ERR) * 100),
        'pct_full_within_3bpm': float(np.mean(full < GOOD_ERR) * 100),
        'jaccard_good_mean': float(jac.mean()),
        'pct_no_good_in_A': float(np.mean([r['n_good_A'] == 0 for r in valid]) * 100),
        'by_stratum': {},
    }
    for posture in loader.POSTURES:
        for condition in loader.CONDITIONS:
            sub = [r for r in valid if r['posture'] == posture
                   and r['condition'] == condition]
            if not sub:
                continue
            f = np.array([r['oracle_full_err'] for r in sub])
            s = np.array([r['split_half_err'] for r in sub])
            summary['by_stratum'][f'{posture}/{condition}'] = {
                'n': len(sub),
                'oracle_full_mae': float(f.mean()),
                'split_half_mae': float(s.mean()),
            }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'summary': summary, 'sessions': valid}, f, indent=2,
                      ensure_ascii=False)
        print(f"\n结果写入: {args.output}")


if __name__ == '__main__':
    main()
