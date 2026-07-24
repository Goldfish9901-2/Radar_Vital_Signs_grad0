#!/usr/bin/env python3
"""Collect experiment JSON files into flat CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "record_type",
    "method",
    "model",
    "setting",
    "source_dataset",
    "target_dataset",
    "split",
    "count",
    "mae_bpm",
    "rmse_bpm",
    "pearson_r",
    "within_3bpm_percent",
    "within_5bpm_percent",
    "within_10bpm_percent",
    "label_mean_bpm",
    "pred_mean_bpm",
    "json_path",
]


def infer_dataset(value: Any) -> str:
    if isinstance(value, list):
        return "+".join(str(item) for item in value)
    return str(value)


def source_from_model_dir(path: str) -> str:
    lower = path.lower()
    names = {"ftu": "FTU", "physdrive": "PhysDrive", "bgt60tr13c": "BGT60TR13C"}
    for dataset, name in names.items():
        if f"_{dataset}_" in lower or lower.endswith(f"_{dataset}_source"):
            return name
    return ""


def target_from_path(path: Path) -> str:
    lower = str(path).lower()
    names = {"ftu": "FTU", "physdrive": "PhysDrive", "bgt60tr13c": "BGT60TR13C"}
    for dataset, name in names.items():
        if f"to_{dataset}" in lower or f"_{dataset}_test" in lower:
            return name
    return ""


def row_from_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "overall" not in data and "before_adaptation" not in data:
        raise ValueError("skip non-evaluation json")
    overall = data.get("overall") or data.get("after_adaptation", {})
    if "overall" in overall:
        overall = overall["overall"]
    target = data.get("target", {})
    method = data.get("method") or data.get("model") or ""
    model = data.get("model") or method
    path_text = str(path).lower()
    if "cycleformer_v3_small" in path_text:
        model = "cycleformer_v3_small"
    elif "cycleformer_v3" in path_text:
        model = "cycleformer_v3"
    elif "cycleformer_v2" in path_text:
        model = "cycleformer_v2"
    setting = "signal_baseline"
    source_dataset = ""
    if "before_adaptation" in data and "after_adaptation" in data:
        setting = "pseudo_label_adaptation"
        target_dataset = infer_dataset(data.get("target_datasets", "")) or target_from_path(path)
        source_dataset = source_from_model_dir(str(data.get("source_checkpoint", "")))
    else:
        target_dataset = infer_dataset(target.get("datasets", ""))
        checkpoint = str(data.get("checkpoint", ""))
        if checkpoint:
            setting = "source_only_or_within"
            source_dataset = source_from_model_dir(checkpoint)
    if method in {"fft", "stft"}:
        model = method.upper()
        method = model

    return {
        "record_type": "overall",
        "method": method,
        "model": model,
        "setting": setting,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "split": target.get("split", ""),
        "count": overall.get("count", ""),
        "mae_bpm": overall.get("mae_bpm", ""),
        "rmse_bpm": overall.get("rmse_bpm", ""),
        "pearson_r": overall.get("pearson_r", ""),
        "within_3bpm_percent": overall.get("within_3bpm_percent", ""),
        "within_5bpm_percent": overall.get("within_5bpm_percent", ""),
        "within_10bpm_percent": overall.get("within_10bpm_percent", ""),
        "label_mean_bpm": overall.get("label_mean_bpm", ""),
        "pred_mean_bpm": overall.get("pred_mean_bpm", ""),
        "json_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.rglob("*.json")):
        if any(part.startswith("smoke_") for part in path.parts):
            continue
        if path.name in {"run_config.json", "history.json"}:
            continue
        try:
            rows.append(row_from_json(path))
        except Exception as exc:  # noqa: BLE001
            if str(exc) == "skip non-evaluation json":
                continue
            rows.append({"record_type": "parse_error", "json_path": str(path), "method": str(exc)})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
