"""mmwave-897-2026 跨 bin 频谱峰投票（本地全量 110 人）。

背景: 每 session 有 ~10 个 good bins (|err|<3bpm), 散布不聚集;
单 bin selection 与 no-selection 融合都停在 15-18 BPM。
关键: 所有 good bin 的 HR 带频谱峰值应出现在同一个频率 (同一颗心跳),
因此跨 bin 峰投票应能提取共峰, 无需定位单个 bin。

方法:
    V0 vote_plain       每 bin 频谱 top-3 峰等权投票
    V1 vote_snr         每 bin 频谱 top-3 峰按 SNR 加权投票
    V2 vote_topk_energy 只对能量 top-16 bin 投票 (排除伪影)
    V3 vote_peak_ratio  投票 + 峰高占比 (峰高/带总能量) 加权

评估: MAE / RMSE / 分层 MAE, 与 oracle (1.64) 对比。

用法:
    python -m src.data.mmwave_vote --dataset /tmp/mmwave_preview --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.mmwave_loader import MMWaveDataLoader
from src.data.mmwave_baseline import (
    REST_HR_BAND_HZ, POSTEX_HR_BAND_HZ,
    butter_bandpass, hr_from_ecg,
)

EXCLUDE_FIRST = 2


def _spectrum(phase: np.ndarray, fs: float,
              hr_band: Tuple[float, float]):
    from scipy.signal import get_window
    filtered = butter_bandpass(phase, fs, hr_band)
    n = len(filtered)
    windowed = filtered * get_window('hann', n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.fft.rfft(windowed, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    return freqs, np.abs(spec), mask


def _top_peaks(amp: np.ndarray, n_peaks: int = 3):
    """amp 中 top-N 峰的 (idx, amp) 列表。"""
    idx = np.argsort(amp)[::-1][:n_peaks]
    return [(int(i), float(amp[i])) for i in idx]


def vote_hr(radar: np.ndarray, fs: float, hr_band: Tuple[float, float],
            method: str) -> float:
    """跨 bin 投票 -> HR (bpm)。"""
    n_bins = radar.shape[2]
    energy = np.abs(radar[:, :, EXCLUDE_FIRST:]).mean(axis=(0, 1)) ** 2
    if method == 'vote_topk_energy':
        topk = max(8, n_bins // 4)
        sel = np.argsort(energy)[::-1][:topk] + EXCLUDE_FIRST
    else:
        sel = range(EXCLUDE_FIRST, n_bins)

    votes = {}   # freq -> weight
    for b in sel:
        cell = radar[:, :, b].mean(axis=1)
        ph_u = np.unwrap(np.angle(cell))
        ph = ph_u - ph_u.mean()
        freqs, amp, mask = _spectrum(ph, fs, hr_band)
        band_amp = amp[mask]
        peaks = _top_peaks(band_amp, n_peaks=3)
        for (pi, pa) in peaks:
            f = float(freqs[mask][pi])
            if method in ('vote_plain', 'vote_topk_energy'):
                w = 1.0
            elif method == 'vote_snr':
                noise = float(np.median(band_amp)) + 1e-12
                w = pa / noise
            elif method == 'vote_peak_ratio':
                w = pa / (float(np.sum(band_amp ** 2)) + 1e-12)
            else:
                raise ValueError(method)
            votes[f] = votes.get(f, 0.0) + w

    if not votes:
        return float('nan')
    best_f = max(votes, key=votes.get)
    return best_f * 60.0


def evaluate_session(loader, pid, posture, condition, method) -> dict:
    radar = loader.load_radar_data(pid, posture, condition)
    ref = loader.load_reference_data(pid, posture, condition)
    fs = loader.FRAME_RATE_HZ
    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ

    hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
    if not np.isfinite(hr_ecg):
        return {'participant': pid, 'posture': posture, 'condition': condition,
                'skip': True}
    try:
        h = vote_hr(radar, fs, hr_band, method)
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
    parser = argparse.ArgumentParser(description="跨 bin 频谱峰投票")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path,
                        default="/tmp/mmwave_vote_results.json")
    args = parser.parse_args()

    loader = MMWaveDataLoader(str(args.dataset))
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]

    methods = ['vote_plain', 'vote_snr', 'vote_topk_energy', 'vote_peak_ratio']
    all_results = {m: [] for m in methods}

    for pid in participants:
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                if not (loader._participant_dir(pid) / posture / condition).exists():
                    continue
                for m in methods:
                    all_results[m].append(evaluate_session(
                        loader, pid, posture, condition, m))
        if pid % 10 == 0:
            print(f"P{pid:03d} done", flush=True)

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
                    'n': len(sub), 'mae': float(np.mean(sub)) if sub else None,
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
