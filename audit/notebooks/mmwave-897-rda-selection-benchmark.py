"""mmwave-897-2026 RDA selection benchmark（Kaggle script kernel）。

目标（AGENTS.md 第 14 节）: 在 110 人 x 4 场景全量上比较 label-free
range selection / localization 方法，统一 phase extraction 与 HR estimator，
回答: "某个纯信号处理的 selection 方法能否把 selection gap 拉下来"。

方法池:
  单 bin:
    max_energy        (baseline)
    mean_energy
    hr_band
    max_temporal_var
    max_phase_var
    max_phase_cohere
    max_band_energy
    hr_band_spatial
    multi_bin_window
    temporal_stability
  region (multi-bin):
    max_energy_region
    hr_band_spatial_region
    multi_bin_window_region
  诊断:
    oracle             (需要 ECG, 只作为上限参考, 报告 selection gap)

输出(/kaggle/working):
  selection_results.csv   per-session 长表
  selection_summary.csv   method x posture x condition 的 MAE/RMSE/Pearson/95%CI/gap
  selection_gap.json      method 级 ALL gap
  bin_dist.json           每个方法的 bin 选择分布 (诊断: 方法到底选到哪里)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import get_window


def _find_input(slug: str) -> str:
    for root in ('/kaggle/input',):
        for dp, dn, fn in os.walk(root):
            if dp.endswith(slug):
                return dp
    return f'/kaggle/input/{slug}'


CODE_DIR = _find_input('mmwave-897-code')
DATA_DIR = _find_input('mmwave-897-2026')
sys.path.insert(0, CODE_DIR)
print(f'CODE_DIR={CODE_DIR}', flush=True)
print(f'DATA_DIR={DATA_DIR}', flush=True)

from mmwave_loader import MMWaveDataLoader  # noqa: E402
from mmwave_baseline import (  # noqa: E402
    POSTEX_HR_BAND_HZ, REST_HR_BAND_HZ, extract_phase, extract_phase_region,
    hr_from_ecg, hr_from_phase, select_range_bin, select_target_region,
)

OUT = Path('/kaggle/working')
OUT.mkdir(exist_ok=True)

SINGLE_BIN_METHODS = [
    'max_energy', 'mean_energy', 'hr_band',
    'max_temporal_var', 'max_phase_var', 'max_phase_cohere',
    'max_band_energy', 'hr_band_spatial', 'multi_bin_window',
    'temporal_stability',
]
REGION_METHODS = [
    'max_energy_region', 'hr_band_spatial_region', 'multi_bin_window_region',
]
ALL_METHODS = SINGLE_BIN_METHODS + REGION_METHODS + ['oracle']

# per-session band-energy 缓存: key=(bin,) -> band energy
_band_cache: dict = {}


def band_energy_of(radar, b, fs, hr_band):
    key = (int(b), hr_band[0], hr_band[1])
    if key not in _band_cache:
        _band_cache[key] = None  # 占位, 避免递归
        from mmwave_baseline import _band_energy
        v, _ = _band_energy(extract_phase(radar, b), fs, hr_band)
        _band_cache[key] = v
    return _band_cache[key]


def oracle_bin(radar, fs, hr_band, hr_ecg):
    best_bin, best_err = None, np.inf
    for b in range(2, radar.shape[2]):
        ph = extract_phase(radar, b)
        h, _, _ = hr_from_phase(ph, fs, hr_band)
        if abs(h - hr_ecg) < best_err:
            best_err, best_bin = abs(h - hr_ecg), b
    return best_bin


t0 = time.time()
loader = MMWaveDataLoader(DATA_DIR)
sessions = loader.list_sessions()
print(f'[{time.time()-t0:.0f}s] sessions: {len(sessions)}', flush=True)

rows = []
fail = 0
for i, (pid, posture, condition) in enumerate(sessions):
    try:
        radar = loader.load_radar_data(pid, posture, condition)
        ref = loader.load_reference_data(pid, posture, condition)
        fs = loader.FRAME_RATE_HZ
        hr_band = (POSTEX_HR_BAND_HZ if condition == 'Post-exercise'
                   else REST_HR_BAND_HZ)
        range_m = loader.load_range_bins(pid, posture, condition)

        hr_ecg, n_peaks, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
        if not np.isfinite(hr_ecg):
            continue

        _band_cache.clear()

        def add(method, b, lo, hi, hr_r):
            rows.append({
                'participant': pid, 'posture': posture, 'condition': condition,
                'method': method, 'range_bin': int(b),
                'range_lo': int(lo), 'range_hi': int(hi),
                'range_m': float(range_m[b]),
                'hr_radar': float(hr_r), 'hr_ecg': float(hr_ecg),
                'abs_err': float(abs(hr_r - hr_ecg)),
                'n_frames': int(radar.shape[0]), 'n_ecg_peaks': n_peaks,
            })

        # ---- 单 bin 方法 ----
        for m in SINGLE_BIN_METHODS:
            try:
                b = select_range_bin(radar, method=m, hr_band=hr_band, fs=fs)
                hr_r, _, _ = hr_from_phase(extract_phase(radar, b), fs,
                                           hr_band)
                add(m, b, b, b + 1, hr_r)
            except Exception as e:  # noqa: BLE001
                print(f'FAIL {m} P{pid:03d} {posture}/{condition}: {e!r}',
                      flush=True)

        # ---- region 方法 ----
        for m in REGION_METHODS:
            try:
                lo, hi = select_target_region(radar, m, hr_band=hr_band,
                                              fs=fs)
                hr_r, _, _ = hr_from_phase(
                    extract_phase_region(radar, lo, hi), fs, hr_band)
                add(m, (lo + hi) // 2, lo, hi, hr_r)
            except Exception as e:  # noqa: BLE001
                print(f'FAIL {m} P{pid:03d} {posture}/{condition}: {e!r}',
                      flush=True)

        # ---- oracle (诊断上限) ----
        try:
            ob = oracle_bin(radar, fs, hr_band, hr_ecg)
            hr_r, _, _ = hr_from_phase(extract_phase(radar, ob), fs, hr_band)
            add('oracle', ob, ob, ob + 1, hr_r)
        except Exception as e:  # noqa: BLE001
            print(f'FAIL oracle P{pid:03d} {posture}/{condition}: {e!r}',
                  flush=True)

    except Exception as e:  # noqa: BLE001
        fail += 1
        print(f'FAIL P{pid:03d} {posture}/{condition}: {e!r}', flush=True)

    if (i + 1) % 100 == 0:
        print(f'[{time.time()-t0:.0f}s] {i+1}/{len(sessions)} done '
              f'(fail={fail})', flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'selection_results.csv', index=False)

# ---------------- 分层汇总 ----------------
from scipy.stats import pearsonr  # noqa: E402


def ci95(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return (float('nan'), float('nan'))
    m, sd = a.mean(), a.std(ddof=1)
    return (m - 1.96 * sd / np.sqrt(n), m + 1.96 * sd / np.sqrt(n))


def _pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(pearsonr(x, y)[0])


summary_rows = []
for m in ALL_METHODS:
    sub = df[df.method == m]
    for (posture, condition), g in sub.groupby(['posture', 'condition']):
        mae = float(g.abs_err.mean())
        rmse = float(np.sqrt((g.abs_err ** 2).mean()))
        r = _pearson(g.hr_radar, g.hr_ecg)
        lo, hi = ci95(g.abs_err.values)
        oracle_mae = float(df[(df.method == 'oracle')
                              & (df.posture == posture)
                              & (df.condition == condition)].abs_err.mean())
        summary_rows.append({
            'method': m, 'posture': posture, 'condition': condition,
            'n': int(len(g)),
            'mae_bpm': mae, 'rmse_bpm': rmse,
            'pearson': r,
            'ci95_lo': lo, 'ci95_hi': hi,
            'oracle_mae': oracle_mae,
            'selection_gap': mae - oracle_mae,
        })
    # ALL
    g = sub
    mae = float(g.abs_err.mean())
    rmse = float(np.sqrt((g.abs_err ** 2).mean()))
    r = _pearson(g.hr_radar, g.hr_ecg)
    lo, hi = ci95(g.abs_err.values)
    oracle_mae = float(df[df.method == 'oracle'].abs_err.mean())
    summary_rows.append({
        'method': m, 'posture': 'ALL', 'condition': 'ALL',
        'n': int(len(g)),
        'mae_bpm': mae, 'rmse_bpm': rmse,
        'pearson': r,
        'ci95_lo': lo, 'ci95_hi': hi,
        'oracle_mae': oracle_mae,
        'selection_gap': mae - oracle_mae,
    })

sdf = pd.DataFrame(summary_rows)
sdf.to_csv(OUT / 'selection_summary.csv', index=False)

# ---------------- bin 选择分布 (诊断) ----------------
bin_dist = {}
for m in ALL_METHODS:
    sub = df[df.method == m]
    hist, _ = np.histogram(sub.range_bin, bins=np.arange(0, 65), density=True)
    bin_dist[m] = [round(float(x), 4) for x in hist]
bin_dist['_range_m'] = [round(float(x), 3) for x in
                        loader.load_range_bins(1, 'Lying', 'Rest')]
with open(OUT / 'bin_dist.json', 'w') as fp:
    json.dump(bin_dist, fp, indent=1)

# ---------------- selection gap json ----------------
gap = {}
for m in ALL_METHODS:
    sub = df[df.method == m]
    o = df[df.method == 'oracle']
    gap[m] = {
        'mae': float(sub.abs_err.mean()),
        'mae_oracle': float(o.abs_err.mean()),
        'selection_gap': float(sub.abs_err.mean() - o.abs_err.mean()),
        'rmse': float(np.sqrt((sub.abs_err ** 2).mean())),
        'pearson': _pearson(sub.hr_radar, sub.hr_ecg),
    }
gap['_n_sessions'] = int(len(df))
gap['_n_failed'] = fail
gap['_runtime_s'] = round(time.time() - t0, 1)
with open(OUT / 'selection_gap.json', 'w') as fp:
    json.dump(gap, fp, indent=2)

# ---------------- 打印汇总 ----------------
print('\n===== SELECTION BENCH SUMMARY (ALL) =====', flush=True)
print(f"sessions ok: {len(df)}, failed: {fail}, "
      f"runtime: {gap['_runtime_s']}s", flush=True)
print(f"{'method':<24}{'MAE':>8}{'RMSE':>8}{'Pearson':>9}{'gap':>8}",
      flush=True)
for m in ALL_METHODS:
    g = gap[m]
    print(f"{m:<24}{g['mae']:>8.2f}{g['rmse']:>8.2f}"
          f"{g['pearson']:>9.3f}{g['selection_gap']:>8.2f}", flush=True)
print('\n===== 分层 MAE (method x posture x condition) =====', flush=True)
piv = sdf[sdf.posture != 'ALL'].pivot_table(
    index='method', columns=['posture', 'condition'], values='mae_bpm')
print(piv.round(2).to_string(), flush=True)
print('DONE', flush=True)
