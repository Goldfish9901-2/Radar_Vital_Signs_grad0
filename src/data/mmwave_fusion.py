"""mmwave-897-2026 no-selection 融合基线（本地全量 110 人）。

诊断结论:
    - 单 bin selection 无法定位 good bin (能量 top-K 命中率 <20%)
    - bin 2 是 DC 耦合伪影 (能量超中位数 2 万倍)
    - 每 session 平均 9.6 个 good bins 散布在多处

因此测试"放弃 selection"的路线:
    F0 complex_mean_all    所有 64 bin 复数平均 -> 单相位 -> HR
    F1 power_weighted      按 HR 带功率加权融合各 bin 相位频谱
    F2 coherent_topk       top-K 能量 bin 复数平均 (排除前 N 个伪影 bin)
    F3 antenna_vote        8 虚拟天线各自 HR, 取中位数
    F5 oracle              对照 (64 bin 扫描最优)

并对比排除前 2/5/10 个 bin 对 max_energy 的影响 (验证 DC 伪影假设)。

用法:
    python -m src.data.mmwave_fusion --dataset /tmp/mmwave_preview --all
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
    butter_bandpass, hr_from_ecg, hr_from_phase,
)


def _phase_from_cell(cell: np.ndarray) -> np.ndarray:
    """单一路径解调: 复数平均 -> unwrap -> 去均值。"""
    ph = np.angle(cell)
    ph = np.unwrap(ph)
    return ph - ph.mean()


def method_hr(radar: np.ndarray, method: str, fs: float,
              hr_band: Tuple[float, float],
              exclude_first: int = 2) -> Tuple[float, dict]:
    """返回 (hr_bpm, info)。"""
    n_frames, n_ant, n_bins = radar.shape

    if method == 'complex_mean_all':
        # 全部 bin 复数平均 (包含伪影 bin, 只排除 DC bin 0-1)
        cell = radar[:, :, exclude_first:].mean(axis=(1, 2))
        ph = _phase_from_cell(cell)
        h, pf, pa = hr_from_phase(ph, fs, hr_band)
        return h, {'peak_freq': pf, 'peak_amp': pa}

    if method == 'power_weighted':
        # 每个 bin 的 HR 带功率 -> 权重, 加权融合带通信号
        cells = radar[:, :, exclude_first:].mean(axis=1)  # (frames, bins)
        n_sel = cells.shape[1]
        weights = np.zeros(n_sel)
        sigs = []
        for j in range(n_sel):
            ph = _phase_from_cell(cells[:, j])
            bp = butter_bandpass(ph, fs, hr_band)
            weights[j] = float(np.var(bp))
            sigs.append(bp)
        W = weights.sum()
        fused = np.zeros_like(sigs[0])
        for j, s in enumerate(sigs):
            fused += (weights[j] / (W + 1e-12)) * s
        h, pf, pa = hr_from_phase(fused, fs, hr_band)
        return h, {'n_bins': n_sel, 'max_w_bin': int(np.argmax(weights))}

    if method == 'coherent_topk':
        # 排除前 N 个伪影 bin 后 top-K 能量 bin 复数平均
        energy = np.abs(radar[:, :, exclude_first:]).mean(axis=(0, 1)) ** 2
        topk = max(4, n_bins // 8)
        idx = np.argsort(energy)[::-1][:topk] + exclude_first
        cell = radar[:, :, idx].mean(axis=(1, 2))
        ph = _phase_from_cell(cell)
        h, pf, pa = hr_from_phase(ph, fs, hr_band)
        return h, {'topk': topk, 'bins': idx.tolist()}

    if method == 'antenna_vote':
        # 8 天线各自 HR, 中位数融合
        hrs = []
        for a in range(n_ant):
            cell = radar[:, a, exclude_first:].mean(axis=1)
            ph = _phase_from_cell(cell)
            h, _, _ = hr_from_phase(ph, fs, hr_band)
            hrs.append(h)
        return float(np.median(hrs)), {'antenna_hrs': hrs}

    raise ValueError(f"未知方法: {method}")


def evaluate_session(loader, pid, posture, condition, method,
                     exclude_first) -> dict:
    radar = loader.load_radar_data(pid, posture, condition)
    ref = loader.load_reference_data(pid, posture, condition)
    fs = loader.FRAME_RATE_HZ
    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ

    hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
    if not np.isfinite(hr_ecg):
        return {'participant': pid, 'posture': posture, 'condition': condition,
                'skip': True}

    if method == 'oracle':
        best_h, best_err = np.nan, np.inf
        for i in range(2, radar.shape[2]):
            cell = radar[:, :, i].mean(axis=1)
            ph = _phase_from_cell(cell)
            h, _, _ = hr_from_phase(ph, fs, hr_band)
            if abs(h - hr_ecg) < best_err:
                best_err, best_h = abs(h - hr_ecg), h
        err = abs(best_h - hr_ecg)
        return {'participant': pid, 'posture': posture, 'condition': condition,
                'hr_radar_bpm': float(best_h), 'hr_ecg_bpm': float(hr_ecg),
                'abs_err_bpm': float(err), 'n_ecg_peaks': int(n_peaks),
                'skip': False}

    try:
        h, info = method_hr(radar, method, fs, hr_band, exclude_first)
    except Exception as e:  # noqa: BLE001
        return {'participant': pid, 'posture': posture, 'condition': condition,
                'skip': True, 'reason': str(e)}
    err = abs(h - hr_ecg) if np.isfinite(h) else float('nan')
    return {'participant': pid, 'posture': posture, 'condition': condition,
            'hr_radar_bpm': float(h) if np.isfinite(h) else None,
            'hr_ecg_bpm': float(hr_ecg),
            'abs_err_bpm': float(err) if np.isfinite(err) else None,
            'n_ecg_peaks': int(n_peaks), 'skip': False}


def main() -> None:
    parser = argparse.ArgumentParser(description="no-selection 融合基线")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path,
                        default="/tmp/mmwave_fusion_results.json")
    args = parser.parse_args()

    loader = MMWaveDataLoader(str(args.dataset))
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]

    methods = ['complex_mean_all', 'power_weighted', 'coherent_topk',
               'antenna_vote', 'oracle']
    # max_energy 排除不同 bin 数的对照 (验证 DC 伪影假设)
    energy_excludes = {2: 'max_energy_x2', 5: 'max_energy_x5',
                       10: 'max_energy_x10'}

    all_results = {m: [] for m in methods}
    for name in energy_excludes.values():
        all_results[name] = []

    for pid in participants:
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                if not (loader._participant_dir(pid) / posture / condition).exists():
                    continue
                for m in methods:
                    all_results[m].append(evaluate_session(
                        loader, pid, posture, condition, m, exclude_first=2))
                # max_energy 对照: 直接选最大能量 bin (排除前 x 个)
                radar = loader.load_radar_data(pid, posture, condition)
                ref = loader.load_reference_data(pid, posture, condition)
                fs = loader.FRAME_RATE_HZ
                hr_band = (POSTEX_HR_BAND_HZ if condition == 'Post-exercise'
                           else REST_HR_BAND_HZ)
                hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
                for x, name in energy_excludes.items():
                    energy = np.abs(radar).mean(axis=(0, 1)) ** 2
                    b = int(np.argmax(energy[x:]) + x)
                    cell = radar[:, :, b].mean(axis=1)
                    ph = _phase_from_cell(cell)
                    h, _, _ = hr_from_phase(ph, fs, hr_band)
                    err = abs(h - hr_ecg) if np.isfinite(hr_ecg) else float('nan')
                    all_results[name].append({
                        'participant': pid, 'posture': posture,
                        'condition': condition,
                        'hr_radar_bpm': float(h) if np.isfinite(h) else None,
                        'hr_ecg_bpm': float(hr_ecg) if np.isfinite(hr_ecg) else None,
                        'abs_err_bpm': float(err) if np.isfinite(err) else None,
                        'n_ecg_peaks': int(n_peaks), 'skip': False,
                    })
                if pid in (1, 2, 3) and posture == 'Lying' and condition == 'Rest':
                    print(f"P{pid:03d} Lying/Rest done", flush=True)

    # 汇总
    summary = {}
    for m, rows in all_results.items():
        valid = [r for r in rows if not r.get('skip') and r.get('abs_err_bpm') is not None]
        errs = np.array([r['abs_err_bpm'] for r in valid])
        by_stratum = {}
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                sub = [r['abs_err_bpm'] for r in valid
                       if r['posture'] == posture and r['condition'] == condition]
                by_stratum[f'{posture}/{condition}'] = {
                    'n': len(sub),
                    'mae': float(np.mean(sub)) if sub else None,
                }
        summary[m] = {
            'n': len(valid),
            'mae_bpm': float(np.mean(errs)) if len(errs) else None,
            'rmse_bpm': float(np.sqrt(np.mean(errs ** 2))) if len(errs) else None,
            'median_bpm': float(np.median(errs)) if len(errs) else None,
            'by_stratum': by_stratum,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'summary': summary,
                       'results': {m: r for m, r in all_results.items()}},
                      f, indent=2, ensure_ascii=False)
        print(f"\n结果写入: {args.output}")


if __name__ == '__main__':
    main()
