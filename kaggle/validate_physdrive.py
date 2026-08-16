"""验证真实 PhysDrive 数据（在 Kaggle 上挂载 xiaoyang274/physdrive 运行，也可本地冒烟）。

回答三个关键问题：
1. mmwave.mat 每段的帧数是否 == 600？维度顺序是否 (frames, 2, doppler, angle, range)？
2. 真实 .mat 是 v5/v7 还是 v7.3(HDF5)？当前 PhysDriveDataLoader 用 scipy.io.loadmat，
   若真实数据为 v7.3 将直接失败——必须在验证中暴露。
3. 20Hz ECG→HR 检测（NeuroKit2）的覆盖率与 HR 分布是否合理？

输出（--output-dir）：
- summary.json  全量 shape / 格式统计 + 抽样 ECG→HR 检测统计
- ecg_hr_examples/*.png 抽样样本的 ECG 波形 + HR 曲线

用法：
    python kaggle/validate_physdrive.py --root <physdrive_root> --output-dir <out>
    --root 应包含 mmWave/ 目录（自动探测 <root>/PhysDrive/mmWave 或 <root>/mmWave）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 仅做 shape/格式探测时不需要重依赖；ECG 检测需要 scipy + neurokit2
from src.data.loaders.export_all_datasets import (  # noqa: E402
    PHYSDRIVE_FRAME_RATE_HZ,
    build_physdrive_reference,
)

EXPECTED_TAIL = (2, 8, 16, 8)  # (real/imag, doppler, angle, range)
EXPECTED_FRAMES = 600

# 需要完整读取做 ECG→HR 检测的抽样上限
DEFAULT_ECG_CHECK_LIMIT = 24
# 每个 session 最多抽样检测的样本数
DEFAULT_ECG_PER_SESSION = 2


def find_mmwave_root(root: Path) -> Path:
    """官方 Kaggle dataset 解压后常见结构：
      - <root>/mmWave/mmWave/<session>/<session>_<NN>/
      - <root>/PhysDrive/mmWave/<session>/...
      - <root>/mmWave/<session>/...
    """
    candidates = [
        root / "mmWave" / "mmWave",
        root / "PhysDrive" / "mmWave",
        root / "mmWave",
        root / "mmWave_data",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"未找到 mmWave 目录，请确认 --root 指向 PhysDrive 数据根（含 mmWave/）。"
        f" 候选: {[str(c) for c in candidates]}"
    )


def mat_shape_v57(path: Path, key: str) -> Optional[Tuple[int, ...]]:
    """v5/v7 格式：用 whosmat 只读头部，快速拿到 shape。"""
    try:
        import scipy.io as sio

        for name, shape, _ in sio.whosmat(str(path)):
            if name == key:
                return tuple(int(x) for x in shape)
    except Exception:  # noqa: BLE001 (v7.3 会抛 NotImplementedError 等)
        return None
    return None


def mat_shape_h5(path: Path, key: str) -> Optional[Tuple[int, ...]]:
    """v7.3(HDF5) 格式：h5py 读取，shape 需反序恢复 MATLAB 维度。"""
    try:
        import h5py

        with h5py.File(str(path), "r") as f:
            if key not in f:
                return None
            shape = tuple(int(x) for x in f[key].shape)
        return tuple(reversed(shape))
    except Exception:  # noqa: BLE001
        return None


def probe_mat_file(
    path: Path, keys: Tuple[str, ...]
) -> Dict[str, Any]:
    """探测单个 .mat：变量名、shape、存储引擎（v5/v7 / v7.3）。"""
    try:
        import scipy.io as sio

        infos = sio.whosmat(str(path))
        shapes = {name: tuple(int(x) for x in shape) for name, shape, _ in infos}
        if any(k in shapes for k in keys):
            return {"format": "v5_v7", "variables": shapes, "size_bytes": path.stat().st_size}
    except Exception:  # noqa: BLE001
        pass

    # 回退 v7.3
    out: Dict[str, Any] = {"format": "v7_3", "variables": {}, "size_bytes": path.stat().st_size}
    try:
        import h5py

        with h5py.File(str(path), "r") as f:
            names = set(f.keys())
            for key in keys:
                if key in names:
                    shape = tuple(int(x) for x in f[key].shape)
                    out["variables"][key] = tuple(reversed(shape))
        return out
    except Exception as exc:  # noqa: BLE001
        out["read_error"] = str(exc)
        return out


def iter_samples(mmwave_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for session_dir in sorted(mmwave_root.iterdir()):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        for sample_dir in sorted(session_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if not sample_dir.name.startswith(session_id + "_"):
                continue
            mmwave = sample_dir / "mmwave.mat"
            if not mmwave.exists():
                continue
            rows.append(
                {
                    "session_id": session_id,
                    "sample_id": sample_dir.name,
                    "sample_dir": sample_dir,
                    "mmwave_mat": mmwave,
                    "ecg_mat": sample_dir / "ecg.mat",
                    "resp_mat": sample_dir / "resp.mat",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="验证真实 PhysDrive 数据格式与 ECG→HR 质量")
    parser.add_argument("--root", type=Path, required=True, help="PhysDrive 数据根目录（含 mmWave/）")
    parser.add_argument("--output-dir", type=Path, default=Path("physdrive_validation"))
    parser.add_argument("--max-samples", type=int, default=None, help="最多检查的样本数（None=全部）")
    parser.add_argument(
        "--ecg-check-limit",
        type=int,
        default=DEFAULT_ECG_CHECK_LIMIT,
        help="完整 ECG→HR 检测的抽样样本数上限",
    )
    parser.add_argument("--ecg-per-session", type=int, default=DEFAULT_ECG_PER_SESSION)
    args = parser.parse_args()

    mmwave_root = find_mmwave_root(args.root)
    print(f"mmWave root: {mmwave_root}", flush=True)
    samples = iter_samples(mmwave_root)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"发现样本数: {len(samples)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "examples").mkdir(parents=True, exist_ok=True)

    # ---------- 1. shape / 格式探测（快速，全量） ----------
    frame_counter: Counter = Counter()
    tail_counter: Counter = Counter()
    format_counter: Counter = Counter()
    anomaly_rows: List[Dict[str, Any]] = []
    ecg_len_counter: Counter = Counter()
    resp_len_counter: Counter = Counter()
    session_stats: Dict[str, Dict[str, Any]] = {}

    for s in samples:
        probe = probe_mat_file(s["mmwave_mat"], ("mmwave",))
        fmt = probe.get("format", "unknown")
        format_counter[fmt] += 1
        shape = probe.get("variables", {}).get("mmwave")
        record: Dict[str, Any] = {
            "session_id": s["session_id"],
            "sample_id": s["sample_id"],
            "mat_format": fmt,
            "mmwave_shape": list(shape) if shape else None,
        }
        if shape is not None:
            frame_counter[int(shape[0])] += 1
            tail_counter[tuple(shape[1:])] += 1
            if int(shape[0]) != EXPECTED_FRAMES:
                record["frame_note"] = f"frames={shape[0]} != {EXPECTED_FRAMES}"
            if tuple(shape[1:]) != EXPECTED_TAIL:
                record["tail_note"] = f"tail={shape[1:]} != {EXPECTED_TAIL}"
        else:
            record["frame_note"] = f"mmwave 变量缺失或读取失败: {probe.get('read_error', '')}"

        # ECG / resp 长度（v5/v7 或 v7.3）
        ecg_shape = None
        resp_shape = None
        if s["ecg_mat"].exists():
            ecg_probe = probe_mat_file(s["ecg_mat"], ("ecg",))
            ecg_shape = ecg_probe.get("variables", {}).get("ecg")
        if s["resp_mat"].exists():
            resp_probe = probe_mat_file(s["resp_mat"], ("resp",))
            resp_shape = resp_probe.get("variables", {}).get("resp")
        record["ecg_shape"] = list(ecg_shape) if ecg_shape else None
        record["resp_shape"] = list(resp_shape) if resp_shape else None
        if ecg_shape:
            ecg_len = int(ecg_shape[-1]) if ecg_shape else 0
            ecg_len_counter[ecg_len] += 1
        if resp_shape:
            resp_len = int(resp_shape[-1]) if resp_shape else 0
            resp_len_counter[resp_len] += 1

        if any(k in record for k in ("frame_note", "tail_note")) or fmt == "v7_3":
            anomaly_rows.append(record)

        st = session_stats.setdefault(
            s["session_id"],
            {"session_id": s["session_id"], "n_samples": 0, "frames": Counter(), "formats": Counter()},
        )
        st["n_samples"] += 1
        if shape:
            st["frames"][int(shape[0])] += 1
        st["formats"][fmt] += 1

    # ---------- 2. 抽样 ECG→HR 检测（完整读取，较慢） ----------
    ecg_checks: List[Dict[str, Any]] = []
    ecg_errors: List[Dict[str, Any]] = []
    per_session_used: Counter = Counter()
    checked = 0
    for s in samples:
        if checked >= args.ecg_check_limit:
            break
        if per_session_used[s["session_id"]] >= args.ecg_per_session:
            continue
        per_session_used[s["session_id"]] += 1
        checked += 1
        row: Dict[str, Any] = {
            "session_id": s["session_id"],
            "sample_id": s["sample_id"],
        }
        try:
            import scipy.io as sio

            ecg_raw = sio.loadmat(str(s["ecg_mat"]))["ecg"].flatten()
            resp_raw = sio.loadmat(str(s["resp_mat"]))["resp"].flatten()
            ref = build_physdrive_reference(
                {"ecg": ecg_raw, "respiration": resp_raw},
                frame_rate_hz=PHYSDRIVE_FRAME_RATE_HZ,
            )
            hr = ref["heart_rate"]
            finite = np.isfinite(hr)
            coverage = float(finite.mean())
            hr_vals = hr[finite]
            row.update(
                {
                    "ecg_len": int(ecg_raw.size),
                    "resp_len": int(resp_raw.size),
                    "hr_coverage": coverage,
                    "hr_segment_bpm": float(np.mean(hr_vals)) if hr_vals.size else None,
                    "hr_std_bpm": float(np.std(hr_vals)) if hr_vals.size else None,
                    "hr_min_bpm": float(np.min(hr_vals)) if hr_vals.size else None,
                    "hr_max_bpm": float(np.max(hr_vals)) if hr_vals.size else None,
                    "note": "segment-level mean RR (PhysDrive official style)",
                }
            )
            ecg_checks.append(row)

            # 绘图（前 4 个成功样本）
            if len([c for c in ecg_checks if c.get("hr_coverage") is not None]) <= 4:
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    t = ref["time"]
                    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    axes[0].plot(t, ecg_raw, lw=0.8)
                    axes[0].set_title(f"{s['session_id']}/{s['sample_id']} ECG @20Hz (n={ecg_raw.size})")
                    axes[0].set_ylabel("ECG (a.u.)")
                    axes[1].plot(t, hr, lw=1.0, marker=".", ms=2)
                    axes[1].set_title(f"HR coverage={coverage:.2f} mean={row['hr_segment_bpm']:.1f} bpm")
                    axes[1].set_ylabel("HR (bpm)")
                    axes[1].set_xlabel("time (s)")
                    fig.tight_layout()
                    out_png = args.output_dir / "examples" / f"{s['session_id']}_{s['sample_id']}.png"
                    fig.savefig(out_png, dpi=120)
                    plt.close(fig)
                    row["plot_path"] = str(out_png)
                except Exception as exc:  # noqa: BLE001
                    row["plot_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)
            ecg_errors.append(row)

    # ---------- 3. 汇总 ----------
    hr_coverages = [c["hr_coverage"] for c in ecg_checks if c.get("hr_coverage") is not None]
    hr_means = [c["hr_segment_bpm"] for c in ecg_checks if c.get("hr_segment_bpm") is not None]
    summary: Dict[str, Any] = {
        "mmwave_root": str(mmwave_root),
        "n_samples": len(samples),
        "n_sessions": len(session_stats),
        "mmwave_frames_distribution": {str(k): v for k, v in sorted(frame_counter.items())},
        "mmwave_tail_distribution": {str(k): v for k, v in sorted(tail_counter.items())},
        "mat_format_distribution": {str(k): v for k, v in format_counter.items()},
        "ecg_len_distribution": {str(k): v for k, v in sorted(ecg_len_counter.items())},
        "resp_len_distribution": {str(k): v for k, v in sorted(resp_len_counter.items())},
        "n_anomalies": len(anomaly_rows),
        "anomalies": anomaly_rows[:50],
        "session_stats": [
            {
                "session_id": st["session_id"],
                "n_samples": st["n_samples"],
                "frames": {str(k): v for k, v in st["frames"].items()},
                "formats": {str(k): v for k, v in st["formats"].items()},
            }
            for st in session_stats.values()
        ],
        "ecg_hr_check": {
            "n_checked": len(ecg_checks),
            "n_failed": len(ecg_errors),
            "failures": ecg_errors,
            "coverage_mean": float(np.mean(hr_coverages)) if hr_coverages else None,
            "coverage_min": float(np.min(hr_coverages)) if hr_coverages else None,
            "coverage_below_0_8": int(sum(1 for c in hr_coverages if c < 0.8)),
            "hr_mean_bpm": float(np.mean(hr_means)) if hr_means else None,
            "hr_std_bpm": float(np.std(hr_means)) if hr_means else None,
            "hr_min_bpm": float(np.min(hr_means)) if hr_means else None,
            "hr_max_bpm": float(np.max(hr_means)) if hr_means else None,
            "rows": ecg_checks,
        },
        "config": {
            "expected_frames": EXPECTED_FRAMES,
            "expected_tail": list(EXPECTED_TAIL),
            "frame_rate_hz": PHYSDRIVE_FRAME_RATE_HZ,
        },
    }
    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nwrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
