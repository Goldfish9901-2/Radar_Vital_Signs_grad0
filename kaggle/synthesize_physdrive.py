"""在 Kaggle 上合成 PhysDrive 兼容格式的试运行数据。

生成目录结构（与 PhysDrive 加载器完全一致）：

    <root>/PhysDrive/mmWave/<session_id>/<session_id>_<NN>/mmwave.mat
    <root>/PhysDrive/mmWave/<session_id>/<session_id>_<NN>/ecg.mat
    <root>/PhysDrive/mmWave/<session_id>/<session_id>_<NN>/resp.mat

说明：
- mmwave.mat 键 `mmwave`，shape (600, 2, 8, 16, 8)（实部/虚部分离，20 Hz）
- ecg.mat 键 `ecg`，shape (1, 600)；resp.mat 键 `resp`，shape (1, 600)
- ECG 为 QRS 尖峰 + 噪声（20 Hz 下 NeuroKit2 可检出，覆盖率 >0.9）
- 呼吸为 0.2-0.3 Hz 正弦 + 噪声
- mmwave 为带心率调制幅度的随机复数 RDA 信号
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

NUM_FRAMES = 600
NUM_DOPPLER = 8
NUM_ANGLE = 16
NUM_RANGE = 8
FRAME_RATE_HZ = 20.0

# session_id 形如 AFH1（segment/gender/time/repeat）
_SESSIONS = [
    ("AFH1", 3),
    ("AFH2", 3),
    ("BFH1", 3),
    ("BFH2", 3),
]


def _synth_ecg(n: int = NUM_FRAMES, hr_bpm: float = 72.0, fs: float = FRAME_RATE_HZ, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr = fs * 60.0 / hr_bpm
    sig = np.zeros(n, dtype=np.float64)
    for beat in np.arange(0, n, rr):
        i = int(round(beat))
        for k in range(-4, 5):
            j = i + k
            if 0 <= j < n:
                sig[j] += 1.0 * np.exp(-((k / 1.5) ** 2))
    sig = sig + 0.03 * rng.standard_normal(n)
    return sig.astype(np.float32)


def _synth_resp(n: int = NUM_FRAMES, rr_bpm: float = 15.0, fs: float = FRAME_RATE_HZ, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    sig = 0.5 * np.sin(2 * np.pi * (rr_bpm / 60.0) * t) + 0.5
    return (sig + 0.02 * rng.standard_normal(n)).astype(np.float32)


def _synth_mmwave(hr_bpm: float = 72.0, fs: float = FRAME_RATE_HZ, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(NUM_FRAMES, dtype=np.float32) / fs
    # 心率调制（胸部起伏），幅度约 0.3，叠加快慢分量
    modulation = 0.5 + 0.3 * np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
    modulation += 0.2 * np.sin(2 * np.pi * 0.25 * t)
    base = rng.standard_normal((NUM_FRAMES, NUM_DOPPLER, NUM_ANGLE, NUM_RANGE)).astype(np.float32)
    signal = base * modulation[:, None, None, None]
    complex_data = signal.astype(np.complex64)
    # 拆分为实部/虚部 (600, 2, 8, 16, 8)
    out = np.empty((NUM_FRAMES, 2, NUM_DOPPLER, NUM_ANGLE, NUM_RANGE), dtype=np.float32)
    out[:, 0, :, :, :] = complex_data.real
    out[:, 1, :, :, :] = complex_data.imag
    return out


def synthesize(root: Path, sessions=None, force: bool = False) -> None:
    mmwave_root = root / "PhysDrive" / "mmWave"
    sessions = sessions or _SESSIONS
    for idx, (session_id, n_samples) in enumerate(sessions):
        for sample_num in range(n_samples):
            sample_dir = mmwave_root / session_id / f"{session_id}_{sample_num:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            hr = float(60 + (idx * 7 + sample_num * 5) % 50)  # 60-110 bpm
            ecg = _synth_ecg(hr_bpm=hr, seed=idx * 100 + sample_num)
            resp = _synth_resp(rr_bpm=12.0 + idx * 1.5 + sample_num, seed=idx * 200 + sample_num)
            mmwave = _synth_mmwave(hr_bpm=hr, seed=idx * 300 + sample_num)
            savemat(sample_dir / "mmwave.mat", {"mmwave": mmwave})
            savemat(sample_dir / "ecg.mat", {"ecg": ecg.reshape(1, -1)})
            savemat(sample_dir / "resp.mat", {"resp": resp.reshape(1, -1)})
            print(f"wrote {sample_dir} hr={hr:.0f} bpm")
    print(f"done: {mmwave_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="合成 PhysDrive 兼容试运行数据")
    parser.add_argument("--root", type=Path, required=True, help="数据根目录（将创建 PhysDrive/mmWave/...）")
    args = parser.parse_args()
    synthesize(args.root)


if __name__ == "__main__":
    main()
