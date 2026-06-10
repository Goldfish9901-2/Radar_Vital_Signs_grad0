"""Source-Free Unsupervised Domain Adaptation for radar heart-rate estimation.

Implements two adaptation strategies:
  - tent (E05): Prediction consistency under input perturbation.
    Freezes all parameters except ScaleNormalize affine weights.
  - wpl  (E06): Weighted Pseudo-Labels using rda_spatial_confidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import HeartTimeMixer, HeartTimeMixerConfig
from src.models.heart_timemixer import ScaleNormalize, count_parameters
from src.training.datasets import LabelStats, RadarWindowDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Source-Free UDA for radar HR models.")
    p.add_argument("--source-model", type=Path, required=True)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-datasets", nargs="+", default=["PhysDrive", "BGT60TR13C"])
    p.add_argument("--method", choices=["tent", "wpl"], default="tent")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--confidence-threshold", type=float, default=0.4)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-batches", type=int, default=None)
    return p.parse_args()


def _resolve_path(row_path: str, export_dir: Path) -> Path:
    p = Path(row_path)
    if p.exists():
        return p
    marker = "training_exports/"
    text = row_path.replace("\\", "/")
    if marker in text:
        rel = text.split(marker, 1)[1]
        candidate = export_dir / rel
        if candidate.exists():
            return candidate
    return export_dir / p.name


def cache_dataset(
    dataset: RadarWindowDataset,
    load_confidence: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    """Pre-load all .npz files into memory tensors."""
    n = len(dataset)
    x_time_list, x_freq_list, y_bpm_list = [], [], []
    confidences: list[float] = [] if load_confidence else None
    print(f"[CACHE] Loading {n} windows...", flush=True)
    for i in range(n):
        p = _resolve_path(dataset.rows[i]["window_path"], dataset.export_dir)
        with np.load(p, allow_pickle=True) as data:
            x_time_list.append(data["x_time"].astype(np.float32))
            x_freq_list.append(data["x_freq"].astype(np.float32))
            if confidences is not None and "meta_json" in data:
                meta = json.loads(str(data["meta_json"]))
                confidences.append(float(meta.get("feature_meta", {}).get("rda_spatial_confidence", 0.5)))
            elif confidences is not None:
                confidences.append(0.5)
        y_bpm_list.append(float(dataset.rows[i]["label_heart_rate"]))
        if (i + 1) % 2000 == 0:
            print(f"[CACHE] {i+1}/{n}", flush=True)
    x_time = torch.from_numpy(np.stack(x_time_list))
    x_freq = torch.from_numpy(np.stack(x_freq_list))
    y_bpm = torch.tensor(y_bpm_list, dtype=torch.float32)
    conf_arr = np.array(confidences, dtype=np.float32) if confidences is not None else None
    print(f"[CACHE] Done: {x_time.shape}", flush=True)
    return x_time, x_freq, y_bpm, conf_arr


def load_source_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = HeartTimeMixerConfig(**ckpt["config"])
    model = HeartTimeMixer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    stats = LabelStats(**ckpt["label_stats"])
    return model, cfg, stats


def freeze_all_except_scale_norm(model: HeartTimeMixer) -> List[nn.Parameter]:
    adapt_params = []
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, ScaleNormalize):
            if module.weight is not None:
                module.weight.requires_grad = True
                adapt_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad = True
                adapt_params.append(module.bias)
    n_adapt = sum(p.numel() for p in adapt_params)
    print(f"[SFUDA] Adapting {n_adapt} ScaleNormalize params", flush=True)
    return adapt_params


def evaluate(model, x_time, x_freq, y_bpm, device, stats, batch_size=256):
    model.eval()
    errors = []
    n = len(y_bpm)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            pred = model(x_time[s:e].to(device), x_freq[s:e].to(device))
            pred_bpm = pred * stats.std + stats.mean
            abs_err = (pred_bpm - y_bpm[s:e].to(device)).abs()
            errors.extend(abs_err.cpu().numpy().tolist())
    if not errors:
        return {"mae_bpm": float("nan"), "count": 0}
    return {
        "mae_bpm": sum(errors) / n,
        "count": n,
        "within_3bpm_percent": 100.0 * sum(e <= 3.0 for e in errors) / n,
        "within_5bpm_percent": 100.0 * sum(e <= 5.0 for e in errors) / n,
    }


def run_tent(model, x_time, x_freq, device, adapt_params, args):
    optimizer = torch.optim.Adam(adapt_params, lr=args.lr, weight_decay=1e-4)
    history = []
    model.train()
    n = len(x_time)
    idx = np.arange(n)
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        total_loss, seen = 0.0, 0
        for s in range(0, n, args.batch_size):
            if args.limit_batches is not None and s // args.batch_size >= args.limit_batches:
                break
            e = min(s + args.batch_size, n)
            bi = idx[s:e]
            xt = x_time[bi].to(device)
            xf = x_freq[bi].to(device)
            with torch.no_grad():
                pred_clean = model(xt, xf)
            noise_t = torch.randn_like(xt) * args.noise_std
            noise_f = torch.randn_like(xf) * args.noise_std
            pred_noisy = model(xt + noise_t, xf + noise_f)
            loss = nn.functional.mse_loss(pred_noisy, pred_clean.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = e - s
            total_loss += float(loss.detach()) * bs
            seen += bs
        avg = total_loss / max(seen, 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"[TENT] epoch {epoch:03d} | loss {avg:.6f}", flush=True)
    return history


def run_wpl(model, x_time, x_freq, confidences, device, stats, args):
    print("[WPL] Generating pseudo-labels...", flush=True)
    model.eval()
    n = len(x_time)
    preds = []
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            p = model(x_time[s:e].to(device), x_freq[s:e].to(device))
            preds.append(p.cpu().numpy())
    pseudo_norm = np.concatenate(preds)

    mask = confidences >= args.confidence_threshold
    n_kept = int(mask.sum())
    print(f"[WPL] threshold={args.confidence_threshold}: kept {n_kept}/{n}", flush=True)
    if n_kept == 0:
        print("[WPL] WARNING: no samples passed, using all", flush=True)
        mask = np.ones(n, dtype=bool)

    sel = np.where(mask)[0]
    xt = x_time[sel]
    xf = x_freq[sel]
    pt = torch.tensor(pseudo_norm[sel], dtype=torch.float32)
    wt = torch.tensor(confidences[sel], dtype=torch.float32)
    wt = wt / wt.mean()

    print(f"[WPL] Training on {len(sel)} samples", flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5, reduction="none")
    history = []
    m = len(sel)
    idx = np.arange(m)
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        total_loss, seen = 0.0, 0
        for s in range(0, m, args.batch_size):
            if args.limit_batches is not None and s // args.batch_size >= args.limit_batches:
                break
            e = min(s + args.batch_size, m)
            bi = idx[s:e]
            pred = model(xt[bi].to(device), xf[bi].to(device))
            loss = (criterion(pred, pt[bi].to(device)) * wt[bi].to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = e - s
            total_loss += float(loss.detach()) * bs
            seen += bs
        avg = total_loss / max(seen, 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"[WPL] epoch {epoch:03d} | loss {avg:.6f}", flush=True)
    return history


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SFUDA] method={args.method}, source={args.source_model}", flush=True)
    print(f"[SFUDA] targets={args.target_datasets}", flush=True)

    model, cfg, stats = load_source_model(args.source_model, device)
    print(f"[SFUDA] Model loaded ({count_parameters(model)} params)", flush=True)

    target_set = set(args.target_datasets)
    want_conf = args.method == "wpl"

    train_ds = RadarWindowDataset(args.export_dir, "train", label_stats=stats, datasets=target_set)
    tr_xt, tr_xf, tr_y, tr_conf = cache_dataset(train_ds, load_confidence=want_conf)

    test_ds = RadarWindowDataset(args.export_dir, "test", label_stats=stats, datasets=target_set)
    te_xt, te_xf, te_y, _ = cache_dataset(test_ds)

    print(f"[SFUDA] train={len(tr_y)}, test={len(te_y)}", flush=True)

    src = evaluate(model, te_xt, te_xf, te_y, device, stats)
    print(f"[SFUDA] Source: MAE={src['mae_bpm']:.2f}, 5BPM={src.get('within_5bpm_percent',0):.1f}%", flush=True)

    t0 = time.time()
    if args.method == "tent":
        params = freeze_all_except_scale_norm(model)
        hist = run_tent(model, tr_xt, tr_xf, device, params, args)
    else:
        hist = run_wpl(model, tr_xt, tr_xf, tr_conf, device, stats, args)

    adapted = evaluate(model, te_xt, te_xf, te_y, device, stats)
    elapsed = time.time() - t0

    delta = src["mae_bpm"] - adapted["mae_bpm"]
    print(f"\n[SFUDA] Source MAE: {src['mae_bpm']:.2f}", flush=True)
    print(f"[SFUDA] Adapted MAE: {adapted['mae_bpm']:.2f} (delta {delta:+.2f})", flush=True)
    print(f"[SFUDA] 5BPM: {src.get('within_5bpm_percent',0):.1f}% -> {adapted.get('within_5bpm_percent',0):.1f}%", flush=True)

    torch.save({
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "label_stats": asdict(stats),
        "method": args.method,
    }, args.output_dir / "adapted.pt")

    summary = {
        "method": args.method,
        "source_model": str(args.source_model),
        "target_datasets": args.target_datasets,
        "source_test": src,
        "adapted_test": adapted,
        "delta_mae": delta,
        "history": hist,
        "elapsed_sec": elapsed,
        "confidence_threshold": args.confidence_threshold if args.method == "wpl" else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
