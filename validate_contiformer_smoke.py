"""Smoke validation for ContiFormer training (no architecture changes intended).

Runs a capped number of batches on the real exported dataset and reports:
  1) batches complete
  2) training loss decreases
  3) no NaN / Inf in loss or gradients
  4) GPU memory footprint
  5) parameter count
  6) forward latency (ms / batch)
  7) output prediction range (normalized + BPM)

Usage:
  uv run python validate_contiformer_smoke.py

Note: ContiFormer is a Transformer, so the (B, nhead, L, L) attention matrix at
L=256 drives memory; batch size is kept at 32 (not 64) to stay safe on the 4 GB
laptop GPU.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from src.models.contiformer import ContiFormerConfig, ContiFormerHeartRateModel, count_parameters
from src.training.common.metrics import mae_bpm
from src.training.datasets import build_datasets

EXPORT = "/home/agent-dev-radar/radar/work/run/upstream/training_exports"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 3e-4
MAX_BATCHES = 300  # full train epoch is ~141 batches; this only caps runaway runs


def main() -> None:
    print(f"device: {DEVICE}  (cuda available: {torch.cuda.is_available()})")
    train_ds, val_ds, test_ds, stats = build_datasets(EXPORT, datasets=None)
    print(f"dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    cfg = ContiFormerConfig()  # use_frequency_domain defaults True
    model = ContiFormerHeartRateModel(cfg).to(DEVICE)
    n_params = count_parameters(model)
    print(f"[5] params: {n_params:,}")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        pin_memory=DEVICE.type == "cuda")
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- [6] forward latency (eval, no grad) ---
    model.eval()
    xb = next(iter(loader))
    xt = xb["x_time"].to(DEVICE)
    xf = xb["x_freq"].to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            model(xt, xf)
        t0 = time.perf_counter()
        N = 30
        for _ in range(N):
            model(xt, xf)
        lat_ms = (time.perf_counter() - t0) / N * 1000
    print(f"[6] forward latency: {lat_ms:.2f} ms/batch (B={BATCH_SIZE}, T=256)")

    # --- [1] batches + [2] loss decrease + [3] NaN ---
    model.train()
    losses, maes = [], []
    nan_seen = False
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n_batches = 0
    for i, batch in enumerate(loader):
        if i >= MAX_BATCHES:
            break
        x_time = batch["x_time"].to(DEVICE, non_blocking=True)
        x_freq = batch["x_freq"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)

        pred = model(x_time, x_freq)
        loss = criterion(pred, y)
        if not torch.isfinite(loss):
            nan_seen = True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not grad_ok:
            nan_seen = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach()))
        with torch.no_grad():
            maes.append(float(mae_bpm(pred.detach(), y.detach(), stats)))
        n_batches += 1

    elapsed = time.perf_counter() - t0
    print(f"[1] batches: {n_batches}  time: {elapsed:.1f}s")
    print(f"[2] loss  first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}")
    print(f"    MAE(BPM) first={maes[0]:.2f}  last={maes[-1]:.2f}  (delta {maes[0]-maes[-1]:+.2f})")
    print(f"[3] NaN/Inf seen: {nan_seen}")

    # --- [4] GPU footprint ---
    if DEVICE.type == "cuda":
        print(f"[4] GPU peak mem: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    else:
        print("[4] GPU: not used (CPU run)")

    # --- [7] prediction range ---
    model.eval()
    vdl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    vb = next(iter(vdl))
    with torch.no_grad():
        p = model(vb["x_time"].to(DEVICE), vb["x_freq"].to(DEVICE))
    p_bpm = stats.denormalize(p)
    print(f"[7] raw output  min={p.min().item():.3f} max={p.max().item():.3f} mean={p.mean().item():.3f}")
    print(f"    BPM pred   min={p_bpm.min().item():.1f} max={p_bpm.max().item():.1f} "
          f"mean={p_bpm.mean().item():.1f}  (label mean={stats.mean:.1f})")
    print(f"    all finite: {torch.isfinite(p).all().item()}")


if __name__ == "__main__":
    main()
