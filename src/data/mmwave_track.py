"""mmwave-897-2026 HR 轨迹级验证（本地全量）。

背景: 单值估计（整段/分段 FFT、IBI）在 Post-exercise 场景全部失败 (<5bpm 命中 17-25%)。
可能原因: 运动后 HR 高且快速回落, 50s 内单值无法刻画。
本实验直接对比雷达与 ECG 的 5s 滑动 HR 轨迹:

    对每 session:
        radar bin2 相位 (10Hz) -> 5s 窗/1s 步 -> IBI -> HR_r(t)
        ECG (250Hz)           -> 5s 窗/1s 步 -> IBI -> HR_e(t)
        时间轴: 均相对 radar t0

    指标 (每 session):
        trajectory_pearson  轨迹相关 (重叠时刻)
        trajectory_mae      轨迹 MAE
        final_hr_err        末段 15s 中位 HR 差
        有效轨迹时刻数

按 posture x condition 分层汇总。

用法:
    python -m src.data.mmwave_track --dataset /tmp/mmwave_preview --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.mmwave_loader import MMWaveDataLoader
from src.data.mmwave_baseline import (
    REST_HR_BAND_HZ, POSTEX_HR_BAND_HZ,
    butter_bandpass, extract_phase,
)

BIN2 = 2
WIN_S = 5.0
STEP_S = 1.0
FINAL_S = 15.0


def _ibi_hr(sig: np.ndarray, fs: float, hr_band: tuple,
            prom_scale: float = 0.2) -> float:
    """带通 + find_peaks -> median IBI -> HR。"""
    filt = butter_bandpass(sig, fs, hr_band)
    if len(filt) < int(fs * 2):
        return float('nan')
    prom = prom_scale * np.std(filt)
    peaks, _ = find_peaks(filt, prominence=prom, distance=int(fs / 4.0))
    if len(peaks) < 3:
        return float('nan')
    ibis = np.diff(peaks) / fs
    ibis = ibis[(ibis >= 0.4) & (ibis <= 2.5)]
    if len(ibis) < 2:
        return float('nan')
    return 60.0 / np.median(ibis)


def sliding_track(sig: np.ndarray, fs: float, hr_band: tuple,
                  win_s: float, step_s: float) -> np.ndarray:
    n = len(sig)
    w, s = int(win_s * fs), int(step_s * fs)
    hrs = []
    for i in range(0, n - w + 1, s):
        hrs.append(_ibi_hr(sig[i:i + w], fs, hr_band))
    return np.array(hrs)


def analyze(loader, pid, posture, condition) -> dict:
    radar = loader.load_radar_data(pid, posture, condition)
    ref = loader.load_reference_data(pid, posture, condition)
    fs = loader.FRAME_RATE_HZ
    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ
    fs_ecg = ref['fs_ecg']

    # 时间轴: 雷达与 ECG 都相对 radar t0 (秒)
    rad_ts = loader.load_radar_timestamps(pid, posture, condition)
    t0 = rad_ts[0]
    rad_t = (rad_ts - t0).astype('timedelta64[ms]').astype(float) / 1000.0
    ecg_ts = ref['ecg_timestamps']
    ecg_t = (ecg_ts - t0).astype('timedelta64[ms]').astype(float) / 1000.0
    ecg_mv = ref['ecg_mv']

    # 雷达相位 (bin2, 8 天线平均)
    ph2 = extract_phase(radar, BIN2)
    hr_r = sliding_track(ph2, fs, hr_band, WIN_S, STEP_S)
    # ECG 轨迹 (R 峰 IBI, 使用较宽 band 以便运动场景)
    hr_e = sliding_track(ecg_mv, fs_ecg, (0.5, 4.0), WIN_S, STEP_S)

    # 时间点 (滑窗中心)
    n_r = len(hr_r)
    r_centers = (np.arange(n_r) * STEP_S + WIN_S / 2)
    e_centers = (np.arange(len(hr_e)) * STEP_S + WIN_S / 2)

    # 对齐: 取同时有效的最近时间点
    pairs = []
    for i, t in enumerate(r_centers):
        if not np.isfinite(hr_r[i]):
            continue
        j = int(round(t / STEP_S))
        if 0 <= j < len(hr_e) and np.isfinite(hr_e[j]):
            pairs.append((hr_r[i], hr_e[j], t))
    if len(pairs) < 4:
        return {'skip': True, 'participant': pid, 'posture': posture,
                'condition': condition, 'reason': 'too_few_track_points'}

    rr = np.array([p[0] for p in pairs])
    ee = np.array([p[1] for p in pairs])
    tt = np.array([p[2] for p in pairs])

    pear = float(np.corrcoef(rr, ee)[0, 1]) if np.std(rr) > 0 and np.std(ee) > 0 \
        else float('nan')
    mae = float(np.mean(np.abs(rr - ee)))

    # 末段 15s
    mask = tt >= (tt[-1] - FINAL_S)
    final_err = (float(np.median(rr[mask]) - np.median(ee[mask]))
                 if mask.sum() >= 2 else float('nan'))

    return {'skip': False, 'participant': pid, 'posture': posture,
            'condition': condition,
            'traj_pearson': pear, 'traj_mae': mae,
            'final_hr_err': final_err,
            'n_pairs': int(len(pairs)), 'n_radar_valid': int(np.isfinite(hr_r).sum()),
            'n_ecg_valid': int(np.isfinite(hr_e).sum()),
            'hr_ecg_final': float(np.median(ee[mask])) if mask.sum() >= 2 else None,
            'hr_radar_final': float(np.median(rr[mask])) if mask.sum() >= 2 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="HR 轨迹级验证")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path,
                        default="/tmp/mmwave_track_results.json")
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
    summary = {'n_sessions': len(results), 'n_valid': len(valid), 'by_stratum': {}}
    for key in ('traj_pearson', 'traj_mae'):
        vals = [r[key] for r in valid if r.get(key) is not None]
        summary[f'{key}_mean'] = float(np.nanmean(vals)) if vals else None
    fe = [abs(r['final_hr_err']) for r in valid
          if r.get('final_hr_err') is not None]
    summary['final_hr_mae'] = float(np.mean(fe)) if fe else None
    pe = [r['traj_pearson'] for r in valid if r.get('traj_pearson') is not None]
    summary['pct_pearson_gt_06'] = float(np.mean(np.array(pe) > 0.6) * 100) if pe else None
    summary['pct_pearson_gt_08'] = float(np.mean(np.array(pe) > 0.8) * 100) if pe else None

    for posture in loader.POSTURES:
        for condition in loader.CONDITIONS:
            sub = [r for r in valid if r['posture'] == posture
                   and r['condition'] == condition]
            if not sub:
                continue
            pv = [r['traj_pearson'] for r in sub if r.get('traj_pearson') is not None]
            mv = [r['traj_mae'] for r in sub if r.get('traj_mae') is not None]
            fv = [abs(r['final_hr_err']) for r in sub
                  if r.get('final_hr_err') is not None]
            summary['by_stratum'][f'{posture}/{condition}'] = {
                'n': len(sub),
                'pearson': float(np.nanmean(pv)) if pv else None,
                'pct_pearson_gt_08': float(np.mean(np.array(pv) > 0.8) * 100) if pv else None,
                'traj_mae': float(np.mean(mv)) if mv else None,
                'final_hr_mae': float(np.mean(fv)) if fv else None,
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
