"""Training loop for radar heart-rate regression models."""

from __future__ import annotations

import argparse
import json
import math
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

from src.models.factory import (
    MODEL_CHOICES,
    ModelConfig,
    config_to_dict,
    count_parameters,
    create_model_and_config,
)
from src.training.common.metrics import aggregate, append_prediction_rows, mae_bpm, rmse_bpm, tolerance_metrics
from src.training.datasets import LabelStats, build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train radar HR regression models on training_exports.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="cycleformer",
        help="Model architecture to train.",
    )
    parser.add_argument("--export-dir", type=Path, default=Path("training_exports"))
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/cycleformer"))
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
    parser.add_argument("--patch-len", type=int, default=16, help="PatchTST patch length.")
    parser.add_argument("--patch-stride", type=int, default=8, help="PatchTST patch stride.")
    # TSLANet-specific hyperparameters (kept separate from shared args so other
    # backbones are unaffected). None of these enable the frequency branch (R8
    # is deferred); use_frequency_domain stays False for tslanet.
    parser.add_argument("--emb-dim", type=int, default=64, help="TSLANet embedding dimension.")
    parser.add_argument("--tslanet-depth", type=int, default=3, help="TSLANet encoder depth.")
    parser.add_argument("--tslanet-patch-len", type=int, default=16, help="TSLANet patch length.")
    parser.add_argument("--tslanet-patch-stride", type=int, default=8, help="TSLANet patch stride.")
    parser.add_argument(
        "--tslanet-dropout",
        type=float,
        default=0.5,
        help="TSLANet dropout (official default, higher than the shared --dropout).",
    )
    parser.add_argument(
        "--use-asb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TSLANet adaptive spectral block.",
    )
    parser.add_argument(
        "--use-icb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TSLANet interactive convolution block.",
    )
    parser.add_argument(
        "--adaptive-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TSLANet adaptive high-frequency mask.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RevIN-style input normalization in TSLANet.",
    )
    parser.add_argument(
        "--channel-mode",
        choices=["joint", "official"],
        default="joint",
        help="TSLANet channel handling: 'joint' mixes channels in the patch "
        "projection; 'official' is channel-independent (F1 experimental factor).",
    )
    parser.add_argument(
        "--heart-periods",
        type=int,
        nargs="+",
        default=None,
        help="CycleFormer heart-period candidates in samples.",
    )
    parser.add_argument(
        "--respiration-periods",
        type=int,
        nargs="+",
        default=None,
        help="CycleFormer respiration-period candidates in samples.",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--decomp-method", choices=["moving_avg", "dft"], default="moving_avg")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--down-sampling-layers", type=int, default=3)
    parser.add_argument("--time-only", action="store_true", help="Disable the frequency branch.")
    parser.add_argument(
        "--time-channels", type=int, default=7,
        help="Input time-branch channel count. Normally auto-derived from the "
        "representation baked into --export-dir/build_config.json; override only "
        "for ad-hoc experiments.",
    )
    parser.add_argument(
        "--freq-channels", type=int, default=7,
        help="Input frequency-branch channel count (see --time-channels).",
    )
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug only: cap batches per epoch.")
    parser.add_argument(
        "--dump-predictions",
        action="store_true",
        help="Write per-sample (label, prediction) rows for the test split to "
        "predictions_test.csv in the output dir (needed for scatter/histogram "
        "and overall Pearson plots in the unified benchmark).",
    )
    resume_grp = parser.add_mutually_exclusive_group()
    resume_grp.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from <output-dir>/best.pt if it exists (restores "
        "model, optimizer, scheduler, epoch, best_val, stale_epochs, history).",
    )
    resume_grp.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from an explicit checkpoint path instead of "
        "<output-dir>/best.pt.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    total_rmse = 0.0
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
        total_rmse += float(rmse_bpm(pred.detach(), y.detach(), stats)) * batch_size
        seen += batch_size
        if collect_participant_metrics:
            append_prediction_rows(detail_rows, batch, pred, stats)

    if seen == 0:
        return {"loss": math.nan, "mae_bpm": math.nan, "rmse_bpm": math.nan}
    metrics: Dict[str, float | Dict[str, Dict[str, float | int]]] = {
        "loss": total_loss / seen,
        "mae_bpm": total_mae / seen,
        "rmse_bpm": total_rmse / seen,
    }
    if collect_participant_metrics:
        metrics.update(tolerance_metrics(detail_rows))
        metrics["by_participant"] = aggregate(detail_rows, "participant_id")
        metrics["by_group"] = aggregate(detail_rows, "group_key")
        # Keep the raw per-sample rows so callers can dump them for plots / overall Pearson.
        metrics["detail_rows"] = detail_rows
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: ModelConfig,
    stats: LabelStats,
    metrics: Dict[str, Any],
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    best_val: float | None = None,
    stale_epochs: int | None = None,
    history: list | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config_to_dict(cfg),
            "label_stats": asdict(stats),
            "metrics": metrics,
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val": best_val,
            "stale_epochs": stale_epochs,
            "history": history,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Derive input channel counts from the representation baked into the export,
    # so backbones are built to match the data (a new method emitting a different
    # channel count works with no code change). Prefer the data-derived geometry
    # written by build_training_dataset.py; fall back to the name->shape registry
    # for legacy exports that lack it; finally fall back to CLI defaults (7, 7).
    try:
        bc_path = Path(args.export_dir) / "build_config.json"
        if bc_path.exists():
            bc = json.loads(bc_path.read_text())
            t_ch = bc.get("time_channels")
            f_ch = bc.get("freq_channels")
            rep = bc.get("representation")
            if t_ch is not None and f_ch is not None:
                args.time_channels = int(t_ch)
                args.freq_channels = int(f_ch)
                print(f"[channels] build_config -> time={t_ch} freq={f_ch}", flush=True)
            elif rep:
                from src.features.representations import representation_input_channels
                t_ch, f_ch = representation_input_channels(rep)
                args.time_channels = t_ch
                args.freq_channels = f_ch
                print(f"[channels] registry({rep}) -> time={t_ch} freq={f_ch}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[channels] keep default (7,7): {exc}", flush=True)

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
        "config": config_to_dict(cfg),
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

    # --- optional checkpoint resume (for long / interruptible runs) ---
    start_epoch = 1
    best_val = float("inf")
    stale_epochs = 0
    history = []
    resume_path = args.resume_from or (args.output_dir / "best.pt")
    if (args.resume or args.resume_from is not None) and resume_path.exists():
        print(f"resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        history = list(ckpt.get("history", []))
        print(f"resumed: start_epoch={start_epoch}, best_val={best_val:.4f}, "
              f"stale_epochs={stale_epochs}, history_len={len(history)}", flush=True)

    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, stats, optimizer=optimizer, limit_batches=args.limit_batches
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, stats, limit_batches=args.limit_batches)
        scheduler.step(val_metrics["mae_bpm"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        # Flush history every epoch (including the early-stop epoch) so the on-disk
        # record is never one epoch behind the actual stop point.
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch:03d} | train MAE {train_metrics['mae_bpm']:.3f} RMSE {train_metrics['rmse_bpm']:.3f} BPM "
            f"| val MAE {val_metrics['mae_bpm']:.3f} RMSE {val_metrics['rmse_bpm']:.3f} BPM | lr {row['lr']:.2e}",
            flush=True,
        )

        if val_metrics["mae_bpm"] < best_val:
            best_val = val_metrics["mae_bpm"]
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "best.pt", model, cfg, stats, val_metrics, epoch,
                optimizer=optimizer, scheduler=scheduler,
                best_val=best_val, stale_epochs=stale_epochs, history=history,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

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
    # Per the experiment results-spec, also emit a standalone eval artifact.
    (args.output_dir / "eval_test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    if args.dump_predictions:
        detail_rows = test_metrics.get("detail_rows", [])
        pred_path = args.output_dir / "predictions_test.csv"
        pred_cols = ["dataset", "group_key", "sample_tag", "participant_id",
                     "label_bpm", "pred_bpm", "abs_error_bpm"]
        with open(pred_path, "w", newline="", encoding="utf-8") as pf:
            import csv as _csv
            w = _csv.DictWriter(pf, fieldnames=pred_cols)
            w.writeheader()
            for row in detail_rows:
                w.writerow({c: row.get(c, "") for c in pred_cols})
        print(f"wrote {len(detail_rows)} per-sample predictions -> {pred_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)



if __name__ == "__main__":
    main()
