"""mmwave-897-2026 时域 IBI 路径 vs 分段 FFT（本地全量）。

背景: FFT-peak 整段估计 MAE ~13.8 (bin2); 分段 FFT 降到 ~10.7 但 pct<5 仅 45%。
论文验证的是 IBI (瞬时心搏间隔, ECG vs 雷达差 ~3ms), 对 HR 漂移更鲁棒。
本实验复现 IBI 提取路径并对比:

    fft_whole      整段 FFT-peak (baseline)
    fft_seg_med    10s 分段 FFT + 中位数
    ibi_bin2       时域 find_peaks -> median(IBI)
    ibi_antenna    8 天线各自 IBI -> 中位数融合
    ibi_clean      去离群 IBI (只保留 0.5-2s 间隔) + 中位数

按 posture x condition 分层报告 MAE / median / pct<5bpm。

用法:
    python -m src.data.mmwave_ibi --dataset /tmp/mmwave_preview --all
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
    butter_bandpass, extract_phase, hr_from_ecg, hr_from_phase,
)

BIN2 = 2


def ibi_from_phase(phase: np.ndarray, fs: float,
                   hr_band: tuple) -> float:
    """时域 IBI: 带通 -> find_peaks -> median(1/间隔*60)。"""
    filt = butter_bandpass(phase, fs, hr_band)
    # 峰高阈值: 相对信号幅度的自适应
    prom = 0.15 * np.std(filt)
    peaks, _ = find_peaks(filt, prominence=prom, distance=int(fs / 4.0))
    if len(peaks) < 3:
        return float('nan')
    ibis = np.diff(peaks) / fs
    ibis = ibis[(ibis >= 0.5) & (ibis <= 2.5)]   # 24-120 bpm
    if len(ibis) < 2:
        return float('nan')
    return 60.0 / np.median(ibis)


def evaluate_session(loader, pid, posture, condition) -> dict:
    radar = loader.load_radar_data(pid, posture, condition)
    ref = loader.load_reference_data(pid, posture, condition)
    fs = loader.FRAME_RATE_HZ
    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ

    hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
    if not np.isfinite(hr_ecg):
        return {'skip': True, 'participant': pid, 'posture': posture,
                'condition': condition}

    ph2 = extract_phase(radar, BIN2)
    h = {}
    # 1. 整段 FFT
    h['fft_whole'], _, _ = hr_from_phase(ph2, fs, hr_band)
    # 2. 分段 FFT 在 main 中单独计算 (此处留空)
    # 3. IBI (单 bin)
    h['ibi_bin2'] = ibi_from_phase(ph2, fs, hr_band)
    # 4. IBI (8 天线各自 -> 中位数)
    ibis_ant = []
    for a in range(radar.shape[1]):
        cell = radar[:, a, BIN2]
        ph = np.unwrap(np.angle(cell))
        ph = ph - ph.mean()
        v = ibi_from_phase(ph, fs, hr_band)
        if np.isfinite(v):
            ibis_ant.append(v)
    h['ibi_antenna'] = float(np.median(ibis_ant)) if ibis_ant else float('nan')

    out = {'skip': False, 'participant': pid, 'posture': posture,
           'condition': condition, 'hr_ecg_bpm': float(hr_ecg),
           'n_ecg_peaks': int(n_peaks), 'n_frames': int(radar.shape[0])}
    for k, v in h.items():
        out[f'{k}_bpm'] = float(v) if np.isfinite(v) else None
        out[f'{k}_err'] = float(abs(v - hr_ecg)) if np.isfinite(v) else None
    return out


def _seg_fft_median(phase: np.ndarray, fs: float,
                    hr_band: tuple, win_s=10, step_s=5) -> float:
    from scipy.signal import get_window
    n = len(phase)
    w, s = int(win_s * fs), int(step_s * fs)
    hrs = []
    for i in range(0, n - w + 1, s):
        seg = butter_bandpass(phase[i:i + w], fs, hr_band)
        wnd = seg * get_window('hann', len(seg))
        nfft = max(2048, 2 ** int(np.ceil(np.log2(len(seg)))))
        spec = np.abs(np.fft.rfft(wnd, nfft))
        freqs = np.fft.rfftfreq(nfft, d=1 / fs)
        m = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
        if m.sum() == 0:
            continue
        hrs.append(freqs[m][np.argmax(spec[m])] * 60)
    return float(np.median(hrs)) if hrs else float('nan')


def main() -> None:
    parser = argparse.ArgumentParser(description="IBI vs FFT 路径")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--participants", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path,
                        default="/tmp/mmwave_ibi_results.json")
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
                r = evaluate_session(loader, pid, posture, condition)
                if not r['skip']:
                    ph2 = extract_phase(
                        loader.load_radar_data(pid, posture, condition), BIN2)
                    fs = loader.FRAME_RATE_HZ
                    hr_band = (POSTEX_HR_BAND_HZ
                               if condition == 'Post-exercise'
                               else REST_HR_BAND_HZ)
                    v = _seg_fft_median(ph2, fs, hr_band)
                    r['fft_seg_med_bpm'] = float(v) if np.isfinite(v) else None
                    r['fft_seg_med_err'] = (float(abs(v - r['hr_ecg_bpm']))
                                            if np.isfinite(v) else None)
                results.append(r)
        if pid % 20 == 0:
            print(f"P{pid:03d} done", flush=True)

    # 汇总
    keys = ['fft_whole', 'fft_seg_med', 'ibi_bin2', 'ibi_antenna']
    summary = {}
    for k in keys:
        ek = f'{k}_err'
        valid = [r for r in results if not r['skip'] and r.get(ek) is not None]
        errs = np.array([r[ek] for r in valid])
        row = {'n': len(valid), 'mae': float(np.mean(errs)),
               'median': float(np.median(errs)),
               'pct_lt5': float(np.mean(errs < 5) * 100),
               'pct_lt10': float(np.mean(errs < 10) * 100)}
        by_stratum = {}
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                sub = [r[ek] for r in valid if r['posture'] == posture
                       and r['condition'] == condition]
                if sub:
                    a = np.array(sub)
                    by_stratum[f'{posture}/{condition}'] = {
                        'n': len(a), 'mae': float(a.mean()),
                        'pct_lt5': float(np.mean(a < 5) * 100)}
        row['by_stratum'] = by_stratum
        summary[k] = row
        print(f"{k:<12} MAE={row['mae']:.2f} med={row['median']:.2f} "
              f"<5bpm={row['pct_lt5']:.0f}%", flush=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'summary': summary, 'sessions': results}, f, indent=2,
                      ensure_ascii=False)
        print(f"\n结果写入: {args.output}")


if __name__ == '__main__':
    main()
