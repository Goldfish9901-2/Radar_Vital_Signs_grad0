"""复现 Nature mmwave-897-2026 论文的 FFT/phase HR baseline。

复现论文 Fig.5 描述的流程 (Parralejo et al., Sci Data 13:897, 2026):
    1. 距离 FFT（数据集已提供 rFFT cube）
    2. target range bin 选择（论文未给准则，此处默认最大能量 bin，可切换）
    3. 零多普勒分量相位提取 = arctan 解调
    4. unwrap -> 去均值 -> 胸部微动信号
    5. 4 阶 Butterworth 带通: Rest 0.8-2.0 Hz / Post-exercise 0.8-3.5 Hz
    6. FFT 最强峰 = HR (bpm)

评估方式（论文以频谱峰匹配/IBI 验证，不报 MAE）:
    - radar HR 与 ECG R 峰平均 IBI 的 HR 对比
    - 频谱峰频率差

Usage:
    python -m src.data.mmwave_baseline --dataset /tmp/mmwave_preview \
        --participants 1 2 3 --posture Lying --condition Rest
    python -m src.data.mmwave_baseline --dataset /tmp/mmwave_preview --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.mmwave_loader import MMWaveDataLoader

# 论文参数
REST_HR_BAND_HZ = (0.8, 2.0)
POSTEX_HR_BAND_HZ = (0.8, 3.5)
BUTTER_ORDER = 4
ECG_HR_BAND_HZ = (0.5, 30.0)   # ECG R 峰检测预处理带通


def butter_bandpass(sig: np.ndarray, fs: float,
                    band: Tuple[float, float], order: int = BUTTER_ORDER) -> np.ndarray:
    """4 阶 Butterworth 带通（零相位 forward-backward 滤波）。"""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype='band')
    return filtfilt(b, a, sig)


def _band_energy(phase: np.ndarray, fs: float,
                 band: Tuple[float, float]) -> Tuple[float, np.ndarray]:
    """HR 带内频谱能量与带内幅度谱（共享 window/FFT 逻辑）。"""
    from scipy.signal import get_window
    filtered = butter_bandpass(phase, fs, band)
    n = len(filtered)
    windowed = filtered * get_window('hann', n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.fft.rfft(windowed, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_amp = np.abs(spec[mask])
    return float(np.sum(band_amp ** 2)), band_amp


def _topk_candidates(energy: np.ndarray, k: int,
                     exclude_first_bins: int) -> np.ndarray:
    """按总能量保留 top-K 候选 bin（人体反射必须有一定强度）。"""
    idx = np.argsort(energy)[::-1]
    return idx[idx >= exclude_first_bins][:k]


def select_range_bin(radar: np.ndarray, method: str = 'max_energy',
                     exclude_first_bins: int = 2,
                     hr_band: Tuple[float, float] = (0.8, 2.0),
                     fs: float = 10.0) -> int:
    """选择 target range bin（单 bin 方法）。

    radar: (frames, antennas, range_bins) complex
    method:
        max_energy        - 全时间平均能量最大的 bin（经典默认）
        mean_energy       - 能量质心
        hr_band           - 候选集内 HR 带频谱能量最大的 bin
        max_temporal_var  - 慢时间去均值后方差最大的 bin（时变微动集中）
        max_phase_var     - unwrap 相位方差最大的 bin
        max_phase_cohere  - 相位连续性(相邻帧相位差 < π 比例)最高的 bin
        max_band_energy   - 全慢时间带(0.3-4.0 Hz)频谱能量最大的 bin
        hr_band_spatial   - HR 带能量 + 邻域一致性加权的 bin
        multi_bin_window  - 连续 5-bin 窗口总能量最大的中心 bin
        temporal_stability- 时域稳定(能量变异系数小)的 top-K 能量候选
    """
    n_bins = radar.shape[2]
    if method in ('max_temporal_var', 'max_phase_var', 'max_phase_cohere',
                  'max_band_energy'):
        k = max(8, n_bins // 4)
        energy = np.abs(radar).mean(axis=(0, 1)) ** 2
        candidates = _topk_candidates(energy, k, exclude_first_bins)
        best_bin, best_score = None, -np.inf
        for i in candidates:
            cell = radar[:, :, i].mean(axis=1)      # (frames,)
            if method == 'max_temporal_var':
                score = float(np.var(cell - cell.mean()))
            elif method == 'max_phase_var':
                ph = np.unwrap(np.angle(cell))
                score = float(np.var(ph))
            elif method == 'max_phase_cohere':
                ph = np.unwrap(np.angle(cell))
                score = float(np.mean(np.abs(np.diff(ph)) < np.pi))
            else:  # max_band_energy: 慢时间频谱带能量 (排除 DC/呼吸混叠区)
                score, _ = _band_energy(np.unwrap(np.angle(cell)), fs,
                                        (0.3, 4.0))
            if score > best_score:
                best_score, best_bin = score, i
        return int(best_bin)
    if method == 'hr_band_spatial':
        # HR 带能量 + 邻域一致性: 对每个候选 bin 打分，
        # 分数 = 自身 HR 带能量 + 0.5*(左右邻域 HR 带能量)
        energy = np.abs(radar).mean(axis=(0, 1)) ** 2
        k = max(8, n_bins // 4)
        candidates = _topk_candidates(energy, k, exclude_first_bins)
        e_cache: dict = {}

        def _band_e(j: int) -> float:
            if j not in e_cache:
                e_cache[j], _ = _band_energy(extract_phase(radar, j), fs,
                                             hr_band)
            return e_cache[j]

        scores = {}
        for i in candidates:
            s = _band_e(i)
            if i - 1 >= exclude_first_bins:
                s += 0.5 * _band_e(i - 1)
            if i + 1 < n_bins:
                s += 0.5 * _band_e(i + 1)
            scores[i] = s
        return int(max(scores, key=scores.get))
    if method == 'multi_bin_window':
        # 连续 5-bin 窗口总能量最大的中心 bin
        energy = np.abs(radar).mean(axis=(0, 1)) ** 2
        half = 2
        best_c, best_sum = None, -np.inf
        for c in range(exclude_first_bins + half, n_bins - half):
            s = float(np.sum(energy[c - half:c + half + 1]))
            if s > best_sum:
                best_sum, best_c = s, c
        return int(best_c)
    if method == 'temporal_stability':
        # top-K 能量候选中选时域最稳定者（能量变异系数最小）
        energy = np.abs(radar).mean(axis=(0, 1)) ** 2
        k = max(8, n_bins // 4)
        candidates = _topk_candidates(energy, k, exclude_first_bins)
        best_bin, best_cv = None, np.inf
        for i in candidates:
            mag = np.abs(radar[:, :, i]).mean(axis=1)
            cv = float(mag.std() / (mag.mean() + 1e-12))
            if cv < best_cv:
                best_cv, best_bin = cv, i
        return int(best_bin)
    energy = np.abs(radar).mean(axis=(0, 1)) ** 2  # (range_bins,)
    if method == 'max_energy':
        return int(np.argmax(energy[exclude_first_bins:]) + exclude_first_bins)
    elif method == 'mean_energy':
        w = energy[exclude_first_bins:]
        r = np.arange(len(w))
        return int(np.sum(r * w) / np.sum(w) + exclude_first_bins)
    elif method == 'hr_band':
        # 候选集约束版: 先按总能量保留 top-K（人体反射存在），
        # 再在候选中选 HR 带内频谱能量最大者（心跳微动集中）。
        # 无约束的 hr_band 会被随机噪声 bin 的带内能量误导。
        k = max(8, n_bins // 4)
        candidates = _topk_candidates(energy, k, exclude_first_bins)
        best_bin, best_power = None, -np.inf
        for i in candidates:
            phase = extract_phase(radar, i)
            power, _ = _band_energy(phase, fs, hr_band)
            if power > best_power:
                best_power, best_bin = power, i
        return int(best_bin)
    elif method == 'oracle':
        raise ValueError(
            "oracle 需要 ECG 真值，请在 evaluate_session 中单独处理")
    raise ValueError(f"未知方法: {method}")


def select_target_region(radar: np.ndarray, method: str,
                         exclude_first_bins: int = 2,
                         half_width: int = 2,
                         hr_band: Tuple[float, float] = (0.8, 2.0),
                         fs: float = 10.0) -> Tuple[int, int]:
    """返回连续 target region 的 [start, end) bin 区间。

    用于 multi-bin 方法:
        multi_bin_window      - 总能量最高的 5-bin 窗口
        hr_band_spatial_region- HR 带能量最高的 bin 及其强邻域
        max_energy_region     - 能量最高 bin 的固定邻域（对照用）
    """
    n_bins = radar.shape[2]
    if method in ('multi_bin_window', 'multi_bin_window_region'):
        c = select_range_bin(radar, 'multi_bin_window',
                             exclude_first_bins, hr_band, fs)
        return max(0, c - half_width), min(n_bins, c + half_width + 1)
    if method == 'hr_band_spatial_region':
        energy = np.abs(radar).mean(axis=(0, 1)) ** 2
        k = max(8, n_bins // 4)
        candidates = _topk_candidates(energy, k, exclude_first_bins)
        best, best_p = None, -np.inf
        for i in candidates:
            p, _ = _band_energy(extract_phase(radar, i), fs, hr_band)
            if p > best_p:
                best_p, best = p, i
        # 扩展: 邻域中 HR 带能量 >= 50% 峰值且总能量 >= 40% 峰值者
        e_all = np.abs(radar).mean(axis=(0, 1)) ** 2
        e_peak = e_all.max()
        region = [best]
        for j in (best - 1, best + 1):
            if 0 <= j < n_bins and j >= exclude_first_bins:
                p, _ = _band_energy(extract_phase(radar, j), fs, hr_band)
                if p >= 0.5 * best_p and e_all[j] >= 0.4 * e_peak:
                    region.append(j)
        lo, hi = min(region), max(region) + 1
        return max(exclude_first_bins, lo), min(n_bins, hi)
    if method == 'max_energy_region':
        c = select_range_bin(radar, 'max_energy',
                             exclude_first_bins, hr_band, fs)
        return max(exclude_first_bins, c - half_width), min(n_bins, c + half_width + 1)
    raise ValueError(f"未知 region 方法: {method}")


def extract_phase_region(radar: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """从连续 range region 提取相位: region 内各 bin 复数平均后再解调。

    相比单 bin 更能抗 speckle/旁瓣，等价于对该 region 做空间聚焦。
    """
    cell = radar[:, :, lo:hi].mean(axis=(1, 2))   # (frames,)
    phase = np.unwrap(np.angle(cell))
    return phase - phase.mean()


def extract_phase(radar: np.ndarray, range_bin: int) -> np.ndarray:
    """从目标 range bin 提取胸部微动相位信号。

    零多普勒分量 = chirp 平均后的复值；8 虚拟天线复数平均（等效正前方波束）。
    返回 unwrap 并去均值后的相位 (frames,)。
    """
    cell = radar[:, :, range_bin]            # (frames, antennas)
    combined = cell.mean(axis=1)             # 虚拟天线复数平均
    phase = np.angle(combined)               # arctan 解调
    phase = np.unwrap(phase)
    return phase - phase.mean()


def hr_from_phase(phase: np.ndarray, fs: float,
                  hr_band: Tuple[float, float]) -> Tuple[float, float, float]:
    """FFT 最强峰 -> HR (bpm)。返回 (hr_bpm, peak_freq_hz, peak_amplitude)。"""
    from scipy.signal import get_window

    filtered = butter_bandpass(phase, fs, hr_band)
    # Hanning 窗 + 零填充到 2048，提高峰分辨率
    n = len(filtered)
    windowed = filtered * get_window('hann', n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.fft.rfft(windowed, nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.abs(spec)

    # 只在 HR 带通内找峰
    mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    peak_idx = int(np.argmax(amp[mask]))
    peak_freq = float(freqs[mask][peak_idx])
    return peak_freq * 60.0, peak_freq, float(amp[mask][peak_idx])


def hr_from_ecg(ecg_mv: np.ndarray, fs: float = 250.0) -> Tuple[float, int, float]:
    """R 峰检测 -> 平均 IBI -> HR (bpm)。返回 (hr_bpm, n_peaks, mean_ibi_s)。"""
    from scipy.signal import find_peaks

    filtered = butter_bandpass(ecg_mv, fs, ECG_HR_BAND_HZ)
    # 自适应阈值: 归一化 + 高度阈值 = 0.6 * 峰值中位数
    z = (filtered - filtered.mean()) / (filtered.std() + 1e-9)
    peaks, props = find_peaks(z, height=0.5 * np.percentile(z, 99),
                              distance=int(0.25 * fs))
    if len(peaks) < 2:
        return float('nan'), 0, float('nan')
    ibi = np.diff(peaks) / fs
    mean_ibi = float(np.mean(ibi))
    return 60.0 / mean_ibi, len(peaks), mean_ibi


REGION_METHODS = ('multi_bin_window_region', 'hr_band_spatial_region',
                  'max_energy_region')


def evaluate_session(loader: MMWaveDataLoader, participant_id: int,
                     posture: str, condition: str,
                     range_bin_method: str = 'max_energy',
                     verbose: bool = True) -> dict:
    radar = loader.load_radar_data(participant_id, posture, condition)
    ref = loader.load_reference_data(participant_id, posture, condition)
    fs = loader.FRAME_RATE_HZ

    hr_band = POSTEX_HR_BAND_HZ if condition == 'Post-exercise' else REST_HR_BAND_HZ

    if range_bin_method == 'oracle':
        # 上限参考: 扫描全部 bin 取与 ECG 误差最小者（需要真值，仅诊断用）
        hr_ecg_oracle, _, _ = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
        best_bin, best_err = None, np.inf
        for i in range(2, radar.shape[2]):
            ph = extract_phase(radar, i)
            h, _, _ = hr_from_phase(ph, fs, hr_band)
            if abs(h - hr_ecg_oracle) < best_err:
                best_err, best_bin = abs(h - hr_ecg_oracle), i
        range_bin = best_bin
    elif range_bin_method in REGION_METHODS:
        lo, hi = select_target_region(radar, range_bin_method,
                                      hr_band=hr_band, fs=fs)
        phase = extract_phase_region(radar, lo, hi)
        hr_radar, peak_freq, peak_amp = hr_from_phase(phase, fs, hr_band)
        hr_ecg, n_peaks, mean_ibi = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])
        err = abs(hr_radar - hr_ecg) if np.isfinite(hr_ecg) else float('nan')
        result = {
            'participant': participant_id,
            'posture': posture,
            'condition': condition,
            'range_bin': (lo + hi) // 2,
            'range_lo': int(lo), 'range_hi': int(hi),
            'range_m': float(loader.load_range_bins(
                participant_id, posture, condition)[(lo + hi) // 2]),
            'hr_radar_bpm': float(hr_radar),
            'hr_ecg_bpm': float(hr_ecg) if np.isfinite(hr_ecg) else None,
            'abs_err_bpm': float(err) if np.isfinite(err) else None,
            'peak_freq_hz': peak_freq,
            'n_ecg_peaks': n_peaks,
            'mean_ibi_s': mean_ibi,
            'n_frames': int(radar.shape[0]),
        }
        if verbose:
            print(
                f"P{participant_id:03d} {posture}/{condition}: "
                f"region=[{lo},{hi}) bin={(lo+hi)//2} "
                f"radar={hr_radar:6.1f} bpm  ecg={hr_ecg:6.1f} bpm  "
                f"err={result['abs_err_bpm']:.1f} bpm  "
                f"frames={result['n_frames']} ecg_peaks={n_peaks}"
            )
        return result
    else:
        range_bin = select_range_bin(radar, method=range_bin_method,
                                     hr_band=hr_band, fs=fs)
    phase = extract_phase(radar, range_bin)
    hr_radar, peak_freq, peak_amp = hr_from_phase(phase, fs, hr_band)

    hr_ecg, n_peaks, mean_ibi = hr_from_ecg(ref['ecg_mv'], ref['fs_ecg'])

    err = abs(hr_radar - hr_ecg) if np.isfinite(hr_ecg) else float('nan')
    result = {
        'participant': participant_id,
        'posture': posture,
        'condition': condition,
        'range_bin': range_bin,
        'range_m': float(loader.load_range_bins(participant_id, posture, condition)[range_bin]),
        'hr_radar_bpm': float(hr_radar),
        'hr_ecg_bpm': float(hr_ecg) if np.isfinite(hr_ecg) else None,
        'abs_err_bpm': float(err) if np.isfinite(err) else None,
        'peak_freq_hz': peak_freq,
        'n_ecg_peaks': n_peaks,
        'mean_ibi_s': mean_ibi,
        'n_frames': int(radar.shape[0]),
    }
    if verbose:
        print(
            f"P{participant_id:03d} {posture}/{condition}: "
            f"bin={result['range_bin']} ({result['range_m']:.2f}m) "
            f"radar={hr_radar:6.1f} bpm  ecg={hr_ecg:6.1f} bpm  "
            f"err={result['abs_err_bpm']:.1f} bpm  "
            f"frames={result['n_frames']} ecg_peaks={n_peaks}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="复现 mmwave-897-2026 论文 FFT HR baseline")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="数据集根目录（解压后的 mmwave-897-2026）")
    parser.add_argument("--participants", type=int, nargs="+", default=None,
                        help="参与者编号列表，如 1 2 3")
    parser.add_argument("--all", action="store_true", help="全部参与者")
    parser.add_argument("--posture", choices=('Lying', 'Sitting'), default='Lying')
    parser.add_argument("--condition", choices=('Rest', 'Post-exercise'), default='Rest')
    parser.add_argument("--range-bin-method", default='max_energy',
                        choices=('max_energy', 'mean_energy', 'hr_band',
                                 'max_temporal_var', 'max_phase_var',
                                 'max_phase_cohere', 'max_band_energy',
                                 'hr_band_spatial', 'multi_bin_window',
                                 'temporal_stability', 'oracle'))
    parser.add_argument("--output", type=Path, default=None,
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    loader = MMWaveDataLoader(str(args.dataset))

    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]

    results = []
    for pid in participants:
        try:
            results.append(evaluate_session(
                loader, pid, args.posture, args.condition, args.range_bin_method))
        except Exception as e:  # noqa: BLE001
            print(f"P{pid:03d} 失败: {e}", file=sys.stderr)

    if results:
        errs = [r['abs_err_bpm'] for r in results if r['abs_err_bpm'] is not None]
        summary = {
            'n_sessions': len(results),
            'n_valid': len(errs),
            'mae_bpm': float(np.mean(errs)) if errs else None,
            'rmse_bpm': float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
            'median_abs_err_bpm': float(np.median(errs)) if errs else None,
        }
        print("\n===== 汇总 =====")
        print(json.dumps(summary, indent=2))
        results.append({'__summary__': summary})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果写入: {args.output}")


if __name__ == '__main__':
    main()
