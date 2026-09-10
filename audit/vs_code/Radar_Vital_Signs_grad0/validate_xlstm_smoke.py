"""Smoke validation for the xLSTM backbone (CPU-only, no GPU contention).

Confirms:
  1) the model builds and parameter count is reasonable
  2) a forward pass over the dual time+freq contract yields shape (B,)
  3) loss + backward produce finite gradients (no NaN)
  4) prediction range is sane after denormalization

Usage:
  uv run python validate_xlstm_smoke.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.models.factory import create_model_and_config
from src.training.common.metrics import mae_bpm
from src.training.datasets import build_datasets

EXPORT = "/home/agent-dev-radar/radar/work/run/upstream/training_exports"
DEVICE = torch.device("cpu")  # force CPU so we never touch the GPU queue
BATCH_SIZE = 8
LR = 3e-4


def main() -> None:
    print(f"device: {DEVICE}")
    train_ds, val_ds, test_ds, stats = build_datasets(EXPORT, datasets=None)
    print(f"dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Build via the factory so CLI wiring is exercised exactly like train_model.py.
    class _Args:
        model = "xlstm"
        d_model = 64
        num_layers = 2
        d_ff = 128
        dropout = 0.2
        time_only = False
    model, cfg = create_model_and_config(_Args())
    from src.models.xlstm import count_parameters
    print(f"[1] params: {count_parameters(model):,}")
    print(f"    config: {cfg}")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # [2] forward shape
    model.eval()
    xb = next(iter(loader))
    with torch.no_grad():
        p = model(xb["x_time"], xb["x_freq"])
    print(f"[2] forward output shape: {tuple(p.shape)} (expect (B,))")

    # [3] backward + NaN guard
    model.train()
    nan_seen = False
    losses, maes = [], []
    for i, batch in enumerate(loader):
        if i >= 50:
            break
        x_time = batch["x_time"]
        x_freq = batch["x_freq"]
        y = batch["y"]
        pred = model(x_time, x_freq)
        loss = criterion(pred, y)
        if not torch.isfinite(loss):
            nan_seen = True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            nan_seen = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            maes.append(float(mae_bpm(pred.detach(), y.detach(), stats)))
    print(f"[3] NaN/Inf seen: {nan_seen}")
    print(f"    loss  first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}")
    print(f"    MAE(BPM) first={maes[0]:.2f}  last={maes[-1]:.2f}  (delta {maes[0]-maes[-1]:+.2f})")

    # [4] prediction range
    model.eval()
    vb = next(iter(DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)))
    with torch.no_grad():
        p = model(vb["x_time"], vb["x_freq"])
    p_bpm = stats.denormalize(p)
    print(f"[4] raw output min={p.min().item():.3f} max={p.max().item():.3f} mean={p.mean().item():.3f}")
    print(f"    BPM pred   min={p_bpm.min().item():.1f} max={p_bpm.max().item():.1f} "
          f"mean={p_bpm.mean().item():.1f}  (label mean={stats.mean:.1f})")
    print(f"    all finite: {torch.isfinite(p).all().item()}")


if __name__ == "__main__":
    main()
