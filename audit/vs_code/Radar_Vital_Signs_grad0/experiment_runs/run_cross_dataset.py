"""True cross-dataset generalization evaluation.

The headline benchmark (make_benchmark_v3.py) reports each model trained AND
tested on the same dataset. That answers "how good is model X on dataset D",
but it does NOT answer the harder question the plan cares about: does a model
trained on one dataset (FTU, the largest/primary) generalize to the others?

This script answers that directly: for every model we take the checkpoint that
was trained on FTU (model_outputs/<model>/FTU/best.pt) and evaluate it on the
TEST split of every dataset (FTU, BGT60TR13C, PhysDrive). Because
evaluate_model.py restores the model's normalization from the CHECKPOINT's
label_stats, the predictions are denormalized back to BPM with FTU's stats and
are directly comparable to each target's BPM labels -- this is a genuine
domain-shift measurement, not a (un)lucky normalization accident.

Output (under model_outputs_cross/):
    <model>/eval_<TARGET>.json      per-target metrics (overall + by participant)
    cross_dataset_matrix.csv        model x target MAE + Pearson + degradation
    cross_dataset_matrix.md         human-readable table
    cross_dataset_matrix.json       raw aggregation (for downstream tooling)

Usage:
    uv run python experiment_runs/run_cross_dataset.py
    uv run python experiment_runs/run_cross_dataset.py --models tcn transformer
    CUDA_VISIBLE_DEVICES="" uv run python experiment_runs/run_cross_dataset.py

The FTU-trained checkpoints must already exist (produced by queue_all.sh).
Missing checkpoints are skipped with a warning so the script runs incrementally.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import MODEL_CHOICES, count_parameters, create_model
from src.training.common.checkpoints import load_run_config, resolve_checkpoint as resolve_checkpoint_path
from src.training.common.metrics import aggregate, append_prediction_rows
from src.training.datasets import LabelStats, RadarWindowDataset

EXPORT = Path("/home/agent-dev-radar/radar/work/run/upstream/training_exports")
SOURCES = ["FTU"]  # canonical protocol = train on FTU, generalize outward
TARGETS = ["FTU", "BGT60TR13C", "PhysDrive"]


def evaluate_checkpoint_on(model_dir: Path, target: str, batch_size: int, num_workers: int, device: torch.device):
    """Evaluate a single checkpoint (trained on its source) on one target's test split."""
    run_config = load_run_config(model_dir)
    ckpt_path = resolve_checkpoint_path(model_dir, None)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = run_config.get("model")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unknown model in {model_dir}: {model_name}")
    label_stats = LabelStats(**ckpt["label_stats"])
    model = create_model(model_name, ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    dataset = RadarWindowDataset(EXPORT, "test", label_stats=label_stats, datasets={target})
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    criterion = nn.SmoothL1Loss(beta=0.5, reduction="sum")
    rows: List[Dict[str, Any]] = []
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

    return {
        "model": model_name,
        "source": model_dir.parent.name,  # dataset the checkpoint was trained on
        "target": target,
        "checkpoint": str(ckpt_path),
        "n_windows": len(dataset),
        "loss": total_loss / seen if seen else math.nan,
        "overall": aggregate(rows)["overall"],
        "by_participant": aggregate(rows, "participant_id"),
        "by_group": aggregate(rows, "group_key"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_CHOICES),
                    help="Subset of backbones (default: all MODEL_CHOICES).")
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--model-outputs-root", type=Path, default=ROOT / "model_outputs")
    ap.add_argument("--out-root", type=Path, default=ROOT / "model_outputs_cross")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None, help="Force device; default cuda if available else cpu.")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)

    matrix: List[Dict[str, Any]] = []
    for model in args.models:
        for source in args.sources:
            src_dir = args.model_outputs_root / model / source
            ckpt = src_dir / "best.pt"
            if not ckpt.exists():
                print(f"[skip] {model}/{source}: no best.pt at {ckpt}")
                continue
            for target in args.targets:
                out_dir = args.out_root / model
                out_dir.mkdir(parents=True, exist_ok=True)
                out_json = out_dir / f"eval_{source}_to_{target}.json"
                if out_json.exists():
                    print(f"[skip] {model}: {source}->{target} (cached)")
                    res = json.loads(out_json.read_text())
                else:
                    print(f"[run] {model}: {source} -> {target}  ({device})")
                    res = evaluate_checkpoint_on(src_dir, target, args.batch_size, args.num_workers, device)
                    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
                ov = res["overall"]
                matrix.append({
                    "model": model,
                    "source": source,
                    "target": target,
                    "n": ov["count"],
                    "mae_bpm": round(ov["mae_bpm"], 3),
                    "rmse_bpm": round(ov["rmse_bpm"], 3),
                    "pearson_r": round(ov["pearson_r"], 4),
                    "within_5bpm_pct": round(ov["within_5bpm_percent"], 2),
                    "within_10bpm_pct": round(ov["within_10bpm_percent"], 2),
                })

    # Aggregate into a model x target matrix (one row per model, columns per target).
    by_model: Dict[str, Dict[str, Any]] = {}
    for r in matrix:
        by_model.setdefault(r["model"], {"model": r["model"]})
        rec = by_model[r["model"]]
        rec[f"mae_{r['target']}"] = r["mae_bpm"]
        rec[f"r_{r['target']}"] = r["pearson_r"]
        rec[f"n_{r['target']}"] = r["n"]

    rows_out = list(by_model.values())
    # degradation vs the in-domain (source==target) MAE, if present
    for rec in rows_out:
        base = rec.get("mae_FTU") if rec.get("mae_FTU") is not None else None
        for t in args.targets:
            key = f"mae_{t}"
            if key in rec and rec[key] is not None and base is not None:
                rec[f"degrad_{t}_vs_FTU"] = round(rec[key] - base, 3)
            else:
                rec[f"degrad_{t}_vs_FTU"] = None

    cols = ["model"]
    for t in args.targets:
        cols += [f"mae_{t}", f"r_{t}", f"n_{t}", f"degrad_{t}_vs_FTU"]
    import csv
    csv_path = args.out_root / "cross_dataset_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in rows_out:
            w.writerow({c: rec.get(c, "") for c in cols})
    (args.out_root / "cross_dataset_matrix.json").write_text(json.dumps(rows_out, indent=2), encoding="utf-8")

    md = ["# Cross-dataset generalization (train FTU -> test all)\n",
          "",
          "MAE in BPM. `degrad_*` = MAE(target) - MAE(FTU): positive = worse than in-domain.\n",
          "",
          "| model | FTU MAE | BGT60 MAE | PhysDrive MAE | ΔBGT60 | ΔPhysDrive |",
          "|---|---|---|---|---|---|"]
    for rec in rows_out:
        md.append(
            f"| {rec['model']} | {rec.get('mae_FTU','')} | {rec.get('mae_BGT60TR13C','')} | "
            f"{rec.get('mae_PhysDrive','')} | {rec.get('degrad_BGT60TR13C_vs_FTU','')} | "
            f"{rec.get('degrad_PhysDrive_vs_FTU','')} |"
        )
    (args.out_root / "cross_dataset_matrix.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {csv_path} ({len(rows_out)} models x {len(args.targets)} targets)")


if __name__ == "__main__":
    main()
