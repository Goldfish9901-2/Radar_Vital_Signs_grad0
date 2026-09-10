"""Evaluate trained heart-rate models on selected exported splits/datasets."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import MODEL_CHOICES, count_parameters, create_model
from src.training.common.checkpoints import load_run_config, resolve_checkpoint as resolve_checkpoint_path
from src.training.common.metrics import aggregate, append_prediction_rows
from src.training.datasets import LabelStats, RadarWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on target exported data.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Training output directory containing run_config.json and best.pt/final.pt.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to <model-dir>/best.pt.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="Model architecture. Usually inferred from <model-dir>/run_config.json.",
    )
    parser.add_argument("--export-dir", type=Path, default=Path("training_exports"))
    parser.add_argument(
        "--target-datasets",
        nargs="+",
        default=None,
        choices=["FTU", "BGT60TR13C", "PhysDrive"],
        help="Target dataset(s) to evaluate. Omit for all datasets in the selected split.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for writing evaluation metrics. Defaults to <model-dir>/eval_<split>_<datasets>.json.",
    )
    return parser.parse_args()


def default_output_path(args: argparse.Namespace, target_datasets: set[str] | None) -> Path | None:
    if args.output_json is not None:
        return args.output_json
    if args.model_dir is None:
        return None
    suffix = "all" if target_datasets is None else "_".join(sorted(target_datasets))
    return args.model_dir / f"eval_{args.split}_{suffix}.json"


def main() -> None:
    args = parse_args()
    run_config = load_run_config(args.model_dir)
    checkpoint_path = resolve_checkpoint_path(args.model_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = args.model or run_config.get("model")
    if model_name not in MODEL_CHOICES:
        raise ValueError("Cannot infer model architecture. Pass --model or provide --model-dir with run_config.json.")

    label_stats = LabelStats(**checkpoint["label_stats"])
    model = create_model(model_name, checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    target_datasets = set(args.target_datasets) if args.target_datasets else None
    dataset = RadarWindowDataset(args.export_dir, args.split, label_stats=label_stats, datasets=target_datasets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.SmoothL1Loss(beta=0.5, reduction="sum")
    rows: list[Dict[str, Any]] = []
    total_loss = 0.0
    seen = 0

    with torch.inference_mode():
        for batch in loader:
            x_time = batch["x_time"].to(device, non_blocking=True)
            x_freq = batch["x_freq"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x_time, x_freq)
            total_loss += float(criterion(pred, y))
            seen += int(y.numel())
            append_prediction_rows(rows, batch, pred, label_stats)

    result = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "source_label_stats": asdict(label_stats),
        "target": {
            "export_dir": str(args.export_dir),
            "split": args.split,
            "datasets": sorted(target_datasets) if target_datasets else "all",
            "size": len(dataset),
        },
        "parameters": count_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "loss": total_loss / seen if seen else math.nan,
        "overall": aggregate(rows)["overall"],
        "by_dataset": aggregate(rows, "dataset"),
        "by_participant": aggregate(rows, "participant_id"),
        "by_group": aggregate(rows, "group_key"),
    }

    output_path = default_output_path(args, target_datasets)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["output_json"] = str(output_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
