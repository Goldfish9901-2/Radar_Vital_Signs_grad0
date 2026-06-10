"""Training loop for heart-rate regression models."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
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

from src.models import (
    HeartTimeMixer,
    HeartTimeMixerConfig,
    TCNConfig,
    TCNHeartRateModel,
    TransformerConfig,
    TransformerHeartRateModel,
)
from src.models.heart_timemixer import count_parameters
from src.training.datasets import LabelStats, build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HeartTimeMixer on training_exports.")
    parser.add_argument(
        "--model",
        choices=["heart_timemixer", "tcn", "transformer"],
        default="heart_timemixer",
        help="Model architecture to train.",
    )
    parser.add_argument("--export-dir", type=Path, default=Path("training_exports"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/heart_timemixer"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["FTU", "BGT60TR13C", "PhysDrive"],
        help="Optional dataset subset, for example: --datasets FTU",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=48, help="TCN hidden channels.")
    parser.add_argument("--num-blocks", type=int, default=4, help="TCN temporal blocks.")
    parser.add_argument("--kernel-size", type=int, default=7, help="TCN convolution kernel size.")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--decomp-method", choices=["moving_avg", "dft"], default="moving_avg")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--down-sampling-layers", type=int, default=3)
    parser.add_argument("--time-only", action="store_true", help="Disable the frequency branch.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug only: cap batches per epoch.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae_bpm(pred_norm: torch.Tensor, y_norm: torch.Tensor, stats: LabelStats) -> torch.Tensor:
    pred = pred_norm * stats.std + stats.mean
    y = y_norm * stats.std + stats.mean
    return torch.mean(torch.abs(pred - y))


def participant_id(dataset: str, group_key: str, sample_tag: str) -> str:
    if "/participant/" in group_key:
        return group_key.rsplit("/", 1)[-1]
    if "/session/" in group_key:
        return group_key.rsplit("/", 1)[-1]
    if dataset in {"FTU", "BGT60TR13C"}:
        match = re.match(r"p0?(\d+)", sample_tag)
        if match:
            return match.group(1)
    if dataset == "PhysDrive":
        return sample_tag.split("_", 1)[0]
    return group_key or sample_tag or "unknown"


def aggregate_errors(rows: list[Dict[str, float | str]], key: str) -> Dict[str, Dict[str, float | int]]:
    buckets: Dict[str, list[Dict[str, float | str]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)

    metrics: Dict[str, Dict[str, float | int]] = {}
    for name, values in buckets.items():
        errors = [float(row["abs_error_bpm"]) for row in values]
        labels = [float(row["label_bpm"]) for row in values]
        preds = [float(row["pred_bpm"]) for row in values]
        metrics[name] = {
            "count": len(values),
            "mae_bpm": sum(errors) / len(errors),
            "within_3bpm_percent": 100.0 * sum(error <= 3.0 for error in errors) / len(errors),
            "within_5bpm_percent": 100.0 * sum(error <= 5.0 for error in errors) / len(errors),
            "label_mean_bpm": sum(labels) / len(labels),
            "pred_mean_bpm": sum(preds) / len(preds),
        }
    return metrics


def tolerance_metrics(rows: list[Dict[str, float | str]]) -> Dict[str, float]:
    if not rows:
        return {"within_3bpm_percent": math.nan, "within_5bpm_percent": math.nan}
    errors = [float(row["abs_error_bpm"]) for row in rows]
    return {
        "within_3bpm_percent": 100.0 * sum(error <= 3.0 for error in errors) / len(errors),
        "within_5bpm_percent": 100.0 * sum(error <= 5.0 for error in errors) / len(errors),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: LabelStats,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int | None = None,
    collect_participant_metrics: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    seen = 0
    detail_rows: list[Dict[str, float | str]] = []

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x_time = batch["x_time"].to(device, non_blocking=True)
        x_freq = batch["x_freq"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(x_time, x_freq)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = int(y.numel())
        total_loss += float(loss.detach()) * batch_size
        total_mae += float(mae_bpm(pred.detach(), y.detach(), stats)) * batch_size
        seen += batch_size
        if collect_participant_metrics:
            pred_bpm = stats.denormalize(pred.detach().cpu())
            y_bpm = batch["y_bpm"].detach().cpu()
            for idx in range(batch_size):
                dataset = batch["dataset"][idx]
                group_key = batch["group_key"][idx]
                sample_tag = batch["sample_tag"][idx]
                label = float(y_bpm[idx])
                prediction = float(pred_bpm[idx])
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "group_key": group_key,
                        "participant_id": participant_id(dataset, group_key, sample_tag),
                        "label_bpm": label,
                        "pred_bpm": prediction,
                        "abs_error_bpm": abs(prediction - label),
                    }
                )

    if seen == 0:
        return {"loss": math.nan, "mae_bpm": math.nan}
    metrics: Dict[str, float | Dict[str, Dict[str, float | int]]] = {
        "loss": total_loss / seen,
        "mae_bpm": total_mae / seen,
    }
    if collect_participant_metrics:
        metrics.update(tolerance_metrics(detail_rows))
        metrics["by_participant"] = aggregate_errors(detail_rows, "participant_id")
        metrics["by_group"] = aggregate_errors(detail_rows, "group_key")
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: HeartTimeMixerConfig | TCNConfig | TransformerConfig,
    stats: LabelStats,
    metrics: Dict[str, Any],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "label_stats": asdict(stats),
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_filter = set(args.datasets) if args.datasets else None
    train_ds, val_ds, test_ds, stats = build_datasets(args.export_dir, datasets=dataset_filter)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    model, cfg = create_model_and_config(args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.SmoothL1Loss(beta=0.5)

    metadata = {
        "config": asdict(cfg),
        "model": args.model,
        "label_stats": asdict(stats),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "datasets": sorted(dataset_filter) if dataset_filter else "all",
        "parameters": count_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "args": vars(args) | {"export_dir": str(args.export_dir), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_val = float("inf")
    stale_epochs = 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, stats, optimizer=optimizer, limit_batches=args.limit_batches
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, stats, limit_batches=args.limit_batches)
        scheduler.step(val_metrics["mae_bpm"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(
            f"epoch {epoch:03d} | train MAE {train_metrics['mae_bpm']:.3f} BPM "
            f"| val MAE {val_metrics['mae_bpm']:.3f} BPM | lr {row['lr']:.2e}",
            flush=True,
        )

        if val_metrics["mae_bpm"] < best_val:
            best_val = val_metrics["mae_bpm"]
            stale_epochs = 0
            save_checkpoint(args.output_dir / "best.pt", model, cfg, stats, val_metrics, epoch)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        stats,
        limit_batches=args.limit_batches,
        collect_participant_metrics=True,
    )
    save_checkpoint(args.output_dir / "final.pt", model, cfg, stats, test_metrics, int(checkpoint["epoch"]))
    summary = {"best_val_mae_bpm": best_val, "test": test_metrics, "elapsed_sec": time.time() - start}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def create_model_and_config(args: argparse.Namespace) -> tuple[nn.Module, HeartTimeMixerConfig | TCNConfig | TransformerConfig]:
    if args.model == "heart_timemixer":
        cfg = HeartTimeMixerConfig(
            d_model=args.d_model,
            d_ff=args.d_ff,
            e_layers=args.e_layers,
            dropout=args.dropout,
            decomp_method=args.decomp_method,
            top_k=args.top_k,
            moving_avg=args.moving_avg,
            down_sampling_layers=args.down_sampling_layers,
            use_frequency_domain=not args.time_only,
        )
        return HeartTimeMixer(cfg), cfg
    if args.model == "tcn":
        cfg = TCNConfig(
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        return TCNHeartRateModel(cfg), cfg
    if args.model == "transformer":
        cfg = TransformerConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.d_ff,
            dropout=args.dropout,
            use_frequency_domain=not args.time_only,
        )
        return TransformerHeartRateModel(cfg), cfg
    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
