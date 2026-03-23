"""统一导出三种数据集脚本。

功能：
1. 调用 FTUDataLoader / BGT60TR13CDataLoader / PhysDriveDataLoader
2. 批量加载三种数据集
3. 保存到指定输出目录

示例：
python src/data/loaders/export_all_datasets.py \
  --ftu

默认运行（所有参数使用 default）：
python src/data/loaders/export_all_datasets.py

可在文件头部修改全局变量（例如 FTU_PARTICIPANT_ID / FTU_SCENARIO / FTU_DISTANCE）来筛选数据。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys

import numpy as np

# 允许直接以脚本方式运行
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.bgt60_loader import BGT60TR13CDataLoader
from src.data.loaders.ftu_loader import FTUDataLoader
from src.data.loaders.physdrive_loader import PhysDriveDataLoader

# ==================== Global Config (Manual Edit) ====================
# 数据路径与输出路径
DATASET_ROOT = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "exports"

# 导出控制
COMPRESS_OUTPUT = True
BGT_INCLUDE_LONG = False

# FTU 筛选（None 表示不筛选）
# 例如：
# FTU_PARTICIPANT_ID = 1
# FTU_SCENARIO = "Distance"
# FTU_DISTANCE = "40 cm"
FTU_PARTICIPANT_ID: Optional[int] = 1
FTU_SCENARIO: Optional[str] = "Distance"
FTU_DISTANCE: Optional[str] = "40 cm"
# ====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FTU/BGT60/PhysDrive datasets")
    parser.add_argument(
        "--ftu",
        action="store_true",
        help="仅导出 FTU（若与其它数据集参数同时出现，则按出现的数据集导出）",
    )
    parser.add_argument(
        "--bgt60",
        action="store_true",
        help="仅导出 BGT60（若与其它数据集参数同时出现，则按出现的数据集导出）",
    )
    parser.add_argument(
        "--physdrive",
        action="store_true",
        help="仅导出 PhysDrive（若与其它数据集参数同时出现，则按出现的数据集导出）",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def save_npz(
    path: Path,
    arrays: Dict[str, Any],
    compress: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def export_ftu(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
    participant_id: Optional[int] = None,
    scenario: Optional[str] = None,
    distance: Optional[str] = None,
) -> Tuple[int, int]:
    loader = FTUDataLoader(str(dataset_root / "4TU.ResearchD"))
    dataset_out = output_dir / "FTU"
    manifest_rows: List[Dict[str, Any]] = []

    samples = loader.list_available_samples(
        participant_id=participant_id,
        scenario=scenario,
        limit=None
    )
    if distance is not None:
        samples = [s for s in samples if s["distance"] == distance]

    if not samples:
        print("[FTU] 没有匹配样本，跳过导出。")
        write_manifest(dataset_out / "manifest.csv", [])
        return 0, 0
    success = 0
    failure = 0

    for i, sample in enumerate(samples, start=1):
        pid = sample["participant_id"]
        scenario = sample["scenario"]
        distance = sample["distance"]
        repeat = sample["repeat"]
        sample_tag = (
            f"p{pid:02d}_{safe_name(scenario)}_{safe_name(distance)}_r{repeat}"
        )

        try:
            radar = loader.load_radar_data(
                participant_id=pid,
                scenario=scenario,
                distance=distance,
                repeat=repeat,
            )
            ref = loader.load_reference_data(
                participant_id=pid,
                scenario=scenario,
                distance=distance,
                repeat=repeat,
            )

            npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
            save_npz(
                npz_path,
                {
                    "radar": radar,
                    "heart_rate": ref["heart_rate"],
                    "timestamps": ref["timestamps"],
                },
                compress=compress,
            )

            meta = {
                "dataset": "FTU",
                "participant_id": pid,
                "scenario": scenario,
                "distance": distance,
                "repeat": repeat,
                "radar_shape": list(radar.shape),
                "radar_dtype": str(radar.dtype),
                "reference_len": int(len(ref["heart_rate"])),
            }
            save_json(dataset_out / "meta" / f"{sample_tag}.json", meta)

            manifest_rows.append(
                {
                    "sample_tag": sample_tag,
                    "participant_id": pid,
                    "scenario": scenario,
                    "distance": distance,
                    "repeat": repeat,
                    "npz_path": str(npz_path),
                    "radar_shape": str(tuple(radar.shape)),
                    "reference_len": len(ref["heart_rate"]),
                    "status": "ok",
                }
            )
            success += 1
        except Exception as e:  # noqa: BLE001
            manifest_rows.append(
                {
                    "sample_tag": sample_tag,
                    "participant_id": pid,
                    "scenario": scenario,
                    "distance": distance,
                    "repeat": repeat,
                    "npz_path": "",
                    "radar_shape": "",
                    "reference_len": "",
                    "status": f"failed: {e}",
                }
            )
            failure += 1

        print(f"[FTU] {i}/{len(samples)} {sample_tag} done")

    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    return success, failure


def iter_bgt_samples(loader: BGT60TR13CDataLoader, include_long: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pid in loader.list_participants():
        info = loader.list_available_measurements(pid)
        for distance in info["short"]:
            rows.append(
                {
                    "participant_id": pid,
                    "distance": distance,
                    "measurement_type": "short",
                }
            )
    if include_long:
        rows.append(
            {
                "participant_id": 1,
                "distance": "long_duration",
                "measurement_type": "long",
            }
        )
    return rows


def export_bgt60(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
    include_long: bool,
) -> Tuple[int, int]:
    loader = BGT60TR13CDataLoader(str(dataset_root / "BGT60TR13C"))
    dataset_out = output_dir / "BGT60TR13C"
    manifest_rows: List[Dict[str, Any]] = []

    samples = iter_bgt_samples(loader, include_long=include_long)
    success = 0
    failure = 0

    for i, sample in enumerate(samples, start=1):
        pid = sample["participant_id"]
        distance = sample["distance"]
        measurement_type = sample["measurement_type"]
        sample_tag = f"p{pid:02d}_{safe_name(distance)}_{measurement_type}"

        try:
            radar = loader.load_radar_data(
                participant_id=pid,
                distance=distance if measurement_type == "short" else "0.3m",
                measurement_type=measurement_type,
                apply_dc_correction=False,
            )
            ref = loader.load_reference_data(
                participant_id=pid,
                distance=distance if measurement_type == "short" else "0.3m",
                measurement_type=measurement_type,
            )

            npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
            save_npz(
                npz_path,
                {
                    "radar": radar,
                    "time": ref["time"],
                    "heart_rate": ref["heart_rate"],
                    "respiration_rate": ref["respiration_rate"],
                },
                compress=compress,
            )

            meta = {
                "dataset": "BGT60TR13C",
                "participant_id": pid,
                "distance": distance,
                "measurement_type": measurement_type,
                "radar_shape": list(radar.shape),
                "radar_dtype": str(radar.dtype),
                "reference_len": int(len(ref["time"])),
            }
            save_json(dataset_out / "meta" / f"{sample_tag}.json", meta)

            manifest_rows.append(
                {
                    "sample_tag": sample_tag,
                    "participant_id": pid,
                    "distance": distance,
                    "measurement_type": measurement_type,
                    "npz_path": str(npz_path),
                    "radar_shape": str(tuple(radar.shape)),
                    "reference_len": len(ref["time"]),
                    "status": "ok",
                }
            )
            success += 1
        except Exception as e:  # noqa: BLE001
            manifest_rows.append(
                {
                    "sample_tag": sample_tag,
                    "participant_id": pid,
                    "distance": distance,
                    "measurement_type": measurement_type,
                    "npz_path": "",
                    "radar_shape": "",
                    "reference_len": "",
                    "status": f"failed: {e}",
                }
            )
            failure += 1

        print(f"[BGT60] {i}/{len(samples)} {sample_tag} done")

    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    return success, failure


def iter_phys_samples(loader: PhysDriveDataLoader) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for session_id in loader.list_sessions():
        for sample_id in loader.list_samples(session_id):
            rows.append({"session_id": session_id, "sample_id": sample_id})
    return rows


def export_physdrive(
    dataset_root: Path,
    output_dir: Path,
    compress: bool,
) -> Tuple[int, int]:
    loader = PhysDriveDataLoader(str(dataset_root / "PhysDrive"))
    dataset_out = output_dir / "PhysDrive"
    manifest_rows: List[Dict[str, Any]] = []

    samples = iter_phys_samples(loader)
    success = 0
    failure = 0

    for i, sample in enumerate(samples, start=1):
        session_id = sample["session_id"]
        sample_id = sample["sample_id"]
        sample_tag = f"{session_id}_{sample_id:03d}"

        try:
            radar = loader.load_radar_data(
                session_id=session_id,
                sample_id=sample_id,
                return_complex=True,
            )
            ref = loader.load_reference_data(
                session_id=session_id,
                sample_id=sample_id,
            )

            npz_path = dataset_out / "samples" / f"{sample_tag}.npz"
            save_npz(
                npz_path,
                {
                    "radar": radar,
                    "ecg": ref["ecg"],
                    "respiration": ref["respiration"],
                },
                compress=compress,
            )

            meta = {
                "dataset": "PhysDrive",
                "session_id": session_id,
                "sample_id": sample_id,
                "radar_shape": list(radar.shape),
                "radar_dtype": str(radar.dtype),
                "reference_len": int(len(ref["ecg"])),
            }
            save_json(dataset_out / "meta" / f"{sample_tag}.json", meta)

            manifest_rows.append(
                {
                    "sample_tag": sample_tag,
                    "session_id": session_id,
                    "sample_id": sample_id,
                    "npz_path": str(npz_path),
                    "radar_shape": str(tuple(radar.shape)),
                    "reference_len": len(ref["ecg"]),
                    "status": "ok",
                }
            )
            success += 1
        except Exception as e:  # noqa: BLE001
            manifest_rows.append(
                {
                    "sample_tag": sample_tag,
                    "session_id": session_id,
                    "sample_id": sample_id,
                    "npz_path": "",
                    "radar_shape": "",
                    "reference_len": "",
                    "status": f"failed: {e}",
                }
            )
            failure += 1

        print(f"[PhysDrive] {i}/{len(samples)} {sample_tag} done")

    write_manifest(dataset_out / "manifest.csv", manifest_rows)
    return success, failure


def main() -> None:
    args = parse_args()
    dataset_root = Path(DATASET_ROOT).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    selected_datasets = []
    if args.ftu:
        selected_datasets.append("FTU")
    if args.bgt60:
        selected_datasets.append("BGT60TR13C")
    if args.physdrive:
        selected_datasets.append("PhysDrive")
    if not selected_datasets:
        selected_datasets = ["FTU", "BGT60TR13C", "PhysDrive"]

    if "FTU" in selected_datasets:
        ok, fail = export_ftu(
            dataset_root=dataset_root,
            output_dir=output_dir,
            compress=COMPRESS_OUTPUT,
            participant_id=FTU_PARTICIPANT_ID,
            scenario=FTU_SCENARIO,
            distance=FTU_DISTANCE,
        )
        summary["FTU"] = {"success": ok, "failed": fail}

    if "BGT60TR13C" in selected_datasets:
        ok, fail = export_bgt60(
            dataset_root=dataset_root,
            output_dir=output_dir,
            compress=COMPRESS_OUTPUT,
            include_long=BGT_INCLUDE_LONG,
        )
        summary["BGT60TR13C"] = {"success": ok, "failed": fail}

    if "PhysDrive" in selected_datasets:
        ok, fail = export_physdrive(
            dataset_root=dataset_root,
            output_dir=output_dir,
            compress=COMPRESS_OUTPUT,
        )
        summary["PhysDrive"] = {"success": ok, "failed": fail}

    save_json(output_dir / "export_summary.json", summary)
    print("\n=== Export Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
