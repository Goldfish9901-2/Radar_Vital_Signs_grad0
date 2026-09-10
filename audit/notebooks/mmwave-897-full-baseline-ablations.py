"""mmwave-897-2026 全量远端评估（Kaggle script kernel）。

按 AGENTS.md 顺序执行:
  ① 全量 110 人 x 4 场景, 复现论文 FFT/phase HR baseline (max_energy)
  ② range-selection ablation: max_energy / mean_energy / hr_band / oracle
  ③ phase-extraction ablation (在 oracle bin 与 max_energy bin 上隔离比较):
     complex_mean / single_antenna / power_weighted / no_unwrap / conj_diff
  ④ representation validator: 每 session 输出 HR 带能量 / 谱SNR /
     相位方差 / 连续性 / 线性趋势
  ⑤ linear probe: 逐 range bin 跨 session 回归 HR 峰频 -> ECG HR 的 R^2

输入(attach):
  - goldfish9901/mmwave-897-2026   (数据)
  - goldfish9901/mmwave-897-code    (扁平化代码)
输出(/kaggle/working):
  - mmwave_full_results.csv         per-session 长表
  - summary.json                    method x posture x condition MAE/RMSE
  - phase_ablation.csv
  - representation_diag.csv
  - linear_probe.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import get_window

CODE_DIR = '/kaggle/input/mmwave-897-code'
DATA_DIR = '/kaggle/input/mmwave-897-2026'
sys.path.insert(0, CODE_DIR)

from mmwave_loader import MMWaveDataLoader  # noqa: E402
from mmwave_baseline import (  # noqa: E402
    butter_bandpass, extract_phase, hr_from_ecg, hr_from_phase,
    POSTEX_HR_BAND_HZ, REST_HR_BAND_HZ, select_range_bin,
)

OUT = Path('/kaggle/working')
OUT.mkdir(exist_ok=True)

t0 = time.time()
loader = MMWaveDataLoader(DATA_DIR)
sessions = loader.list_sessions()
print(f'[{time.time()-t0:.0f}s] sessions: {len(sessions)}', flush=True)

RANGE_METHODS = ['max_energy', 'mean_energy', 'hr_band', 'oracle']
PHASE_VARIANTS = [
    'complex_mean', 'single_antenna', 'power_weighted', 'no_unwrap', 'conj_diff',
]


def extract_phase_variant(radar: np.ndarray, range_bin: int,
                          variant: str) -> np.ndarray:
    """相位提取变体（全部在同一 range bin 上比较, 隔离 bin 影响）。"""
    cell = radar[:, :, range_bin]  # (frames, antennas)
    if variant == 'complex_mean':
        sig = cell.mean(axis=1)
        phase = np.unwrap(np.angle(sig))
        return phase - phase.mean()
    if variant == 'single_antenna':
        sig = cell[:, 0]
        phase = np.unwrap(np.angle(sig))
        return phase - phase.mean()
    if variant == 'power_weighted':
        w = np.abs(cell) ** 2
        sig = np.sum(cell * w, axis=1) / (np.sum(w, axis=1) + 1e-12)
        phase = np.unwrap(np.angle(sig))
        return phase - phase.mean()
    if variant == 'no_unwrap':
        sig = cell.mean(axis=1)
        phase = np.angle(sig)
        return phase - phase.mean()
    if variant == 'conj_diff':
        sig = cell.mean(axis=1)
        diff = sig[1:] * np.conj(sig[:-1])  # 相邻帧共轭相乘
        phase = np.concatenate([[0.0], np.cumsum(np.angle(diff))])
        return phase - phase.mean()
    raise ValueError(f'unknown variant: {variant}')


def representation_diagnostics(phase: np.ndarray, fs: float,
                               hr_band) -> dict:
    """AGENTS.md 第 8/9 步的 representation 物理诊断量。"""
    filtered = butter_bandpass(phase, fs, hr_band)
    n = len(filtered)
    windowed = filtered * get_window('hann', n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.fft.rfft(windowed, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    band_amp = np.abs(spec[mask])
    x = np.arange(len(phase))
    slope = float(np.polyfit(x, phase, 1)[0])
    return {
        'hr_band_energy': float(np.sum(band_amp ** 2)),
        'spectral_snr': float(band_amp.max() / (np.median(band_amp) + 1e-12)),
        'phase_var': float(np.var(phase)),
        'phase_continuity': float(np.mean(np.abs(np.diff(phase)) < np.pi)),
        'phase_slope': slope,
    }


def hr_peak_freq(radar: np.ndarray, range_bin: int, fs: float,
                 hr_band) -> float:
    """该 bin 的 HR 带内最强峰频率 (Hz)。linear probe 用。"""
    phase = extract_phase(radar, range_bin)
    _, f, _ = hr_from_phase(phase, fs, hr_band)
    return f


rows = []
phase_rows = []
diag_rows = []
lp_bins = np.arange(2, 64)  # linear probe 扫描的 range bins
lp_freq = {b: [] for b in lp_bins}   # bin -> [hr_freq, ...]
lp_ecg = []                          # 对应顺序的 ecg hr

fail = 0
for i, (pid, posture, condition) in enumerate(sessions):
    try:
        radar = loader.load_radar_data(pid, posture, condition)
        ref = loader.load_reference_data(pid, posture, condition)
        fs = loader.FRAME_RATE_HZ
        hr_band = (POSTEX_HR_BAND_HZ if condition == 'Post-exercise'
                   else REST_HR_BAND_HZ)
        range_m = loader.load_range_bins(pid, posture, condition)

        hr_ecg, n_peaks, mean_ibi = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
        if not np.isfinite(hr_ecg):
            continue

        # ---- ② range-selection ablation ----
        bin_of = {}
        for m in RANGE_METHODS:
            if m == 'oracle':
                best_bin, best_err = None, np.inf
                for b in range(2, radar.shape[2]):
                    ph = extract_phase(radar, b)
                    h, _, _ = hr_from_phase(ph, fs, hr_band)
                    if abs(h - hr_ecg) < best_err:
                        best_err, best_bin = abs(h - hr_ecg), b
                range_bin = best_bin
            else:
                range_bin = select_range_bin(radar, method=m,
                                             hr_band=hr_band, fs=fs)
            bin_of[m] = range_bin
            ph = extract_phase(radar, range_bin)
            hr_r, pf, pa = hr_from_phase(ph, fs, hr_band)
            rows.append({
                'participant': pid, 'posture': posture, 'condition': condition,
                'method': m, 'range_bin': int(range_bin),
                'range_m': float(range_m[range_bin]),
                'hr_radar': float(hr_r), 'hr_ecg': float(hr_ecg),
                'abs_err': float(abs(hr_r - hr_ecg)),
                'n_frames': int(radar.shape[0]), 'n_ecg_peaks': n_peaks,
            })

        # ---- ③ phase-extraction ablation (oracle bin + max_energy bin) ----
        for anchor, anchor_bin in [('oracle_bin', bin_of['oracle']),
                                   ('max_energy_bin', bin_of['max_energy'])]:
            for v in PHASE_VARIANTS:
                ph = extract_phase_variant(radar, anchor_bin, v)
                hr_r, pf, pa = hr_from_phase(ph, fs, hr_band)
                phase_rows.append({
                    'participant': pid, 'posture': posture,
                    'condition': condition, 'anchor': anchor,
                    'variant': v, 'range_bin': int(anchor_bin),
                    'hr_radar': float(hr_r), 'hr_ecg': float(hr_ecg),
                    'abs_err': float(abs(hr_r - hr_ecg)),
                })

        # ---- ④ representation validator (oracle bin) ----
        diag = representation_diagnostics(extract_phase(radar, bin_of['oracle']),
                                          fs, hr_band)
        diag_rows.append({
            'participant': pid, 'posture': posture, 'condition': condition,
            'range_bin': int(bin_of['oracle']),
            'hr_ecg': float(hr_ecg), **diag,
        })

        # ---- ⑤ linear probe: 每 bin 峰频 vs ECG HR ----
        lp_ecg.append(float(hr_ecg))
        for b in lp_bins:
            lp_freq[b].append(hr_peak_freq(radar, b, fs, hr_band))

    except Exception as e:  # noqa: BLE001
        fail += 1
        print(f'FAIL P{pid:03d} {posture}/{condition}: {e!r}', flush=True)

    if (i + 1) % 100 == 0:
        print(f'[{time.time()-t0:.0f}s] {i+1}/{len(sessions)} done '
              f'(fail={fail})', flush=True)

# ---------------- 输出 ----------------
df = pd.DataFrame(rows)
df.to_csv(OUT / 'mmwave_full_results.csv', index=False)

# summary
summary = {'mae_bpm': {}, 'rmse_bpm': {}}
for m in RANGE_METHODS:
    sub = df[df.method == m]
    for (posture, condition), g in sub.groupby(['posture', 'condition']):
        key = f'{m}|{posture}|{condition}'
        summary['mae_bpm'][key] = float(g.abs_err.mean())
        summary['rmse_bpm'][key] = float(np.sqrt((g.abs_err ** 2).mean()))
# 总体
for m in RANGE_METHODS:
    g = df[df.method == m]
    summary['mae_bpm'][f'{m}|ALL'] = float(g.abs_err.mean())
    summary['rmse_bpm'][f'{m}|ALL'] = float(np.sqrt((g.abs_err ** 2).mean()))

pdf = pd.DataFrame(phase_rows)
pdf.to_csv(OUT / 'phase_ablation.csv', index=False)
summary['phase_mae_bpm'] = {}
for (anchor, v), g in pdf.groupby(['anchor', 'variant']):
    summary['phase_mae_bpm'][f'{anchor}|{v}'] = float(g.abs_err.mean())

ddf = pd.DataFrame(diag_rows)
ddf.to_csv(OUT / 'representation_diag.csv', index=False)

# linear probe: 每个 bin 的 R^2 (峰频 -> ECG HR 线性回归)
ecg_arr = np.array(lp_ecg)
probe = {}
for b, freq_list in lp_freq.items():
    f = np.array(freq_list)
    if len(f) != len(ecg_arr) or np.std(f) < 1e-9:
        continue
    r = np.corrcoef(f, ecg_arr)[0, 1]
    probe[str(b)] = float(r * r)
probe['_n'] = int(len(ecg_arr))
with open(OUT / 'linear_probe.json', 'w') as fp:
    json.dump(probe, fp, indent=2)

summary['n_sessions'] = int(len(df))
summary['n_failed'] = fail
summary['runtime_s'] = round(time.time() - t0, 1)
with open(OUT / 'summary.json', 'w') as fp:
    json.dump(summary, fp, indent=2)

print('\n===== SUMMARY =====', flush=True)
print(f"sessions ok: {len(df)}, failed: {fail}, runtime: {summary['runtime_s']}s",
      flush=True)
for m in RANGE_METHODS:
    print(f"{m:<12} MAE={summary['mae_bpm'][f'{m}|ALL']:.2f} "
          f"RMSE={summary['rmse_bpm'][f'{m}|ALL']:.2f}", flush=True)
print('phase ablation (oracle_bin):', flush=True)
for k, v in summary['phase_mae_bpm'].items():
    if k.startswith('oracle_bin'):
        print(f'  {k:<24} MAE={v:.2f}', flush=True)
print('DONE', flush=True)
