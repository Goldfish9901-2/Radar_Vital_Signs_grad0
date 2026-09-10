"""Inference benchmark for the newly trained backbones.

Loads each model's best.pt (rebuilt from the stored config), runs the FTU test
loader, and measures:
  * parameter count (from run_config.json)
  * average inference latency per batch (ms/batch)
  * peak GPU memory during a full test forward pass (MB)

This produces the latency / memory / params columns used by the unified
benchmark comparative analysis. It is run AFTER training completes.

Usage:
  uv run python experiment_runs/benchmark_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]          # .../work/upstream_rw
sys.path.insert(0, str(ROOT))                       # make the `src` package importable
EXPORT = Path("/home/agent-dev-radar/radar/work/run/upstream/training_exports")
MODEL_OUTPUTS = ROOT / "model_outputs"
OUT = ROOT / "experiment_runs" / "benchmark_2026-07-26" / "inference_metrics.json"

MODELS = ["dlinear", "nlinear", "tsmixer", "contiformer", "mamba"]
INFERENCE_BATCH = 64
WARMUP = 10
N_ITERS = 60


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from src.models.factory import create_model  # noqa: E402
    from src.training.datasets import build_datasets  # noqa: E402

    _, _, test_ds, stats = build_datasets(EXPORT, datasets={"FTU"})
    loader = DataLoader(test_ds, batch_size=INFERENCE_BATCH, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")

    results = {}
    for name in MODELS:
        odir = MODEL_OUTPUTS / name
        best = odir / "best.pt"
        rc_path = odir / "run_config.json"
        if not (best.exists() and rc_path.exists()):
            print(f"[skip] {name}: no best.pt / run_config.json yet")
            continue
        rc = json.loads(rc_path.read_text())
        ckpt = torch.load(best, map_location=device, weights_only=False)
        model = create_model(rc["model"], ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        n_params = rc.get("parameters", -1)
        xb = next(iter(loader))
        xt = xb["x_time"].to(device)
        xf = xb["x_freq"].to(device)

        # warmup
        with torch.no_grad():
            for _ in range(WARMUP):
                model(xt, xf)
        # latency
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(N_ITERS):
                    model(xt, xf)
                torch.cuda.synchronize()
            else:
                t0 = time.perf_counter()
                for _ in range(N_ITERS):
                    model(xt, xf)
            lat_ms = (time.perf_counter() - t0) / N_ITERS * 1000

        # peak GPU memory over a full test forward pass
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for batch in loader:
                model(batch["x_time"].to(device), batch["x_freq"].to(device))
        gpu_peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

        results[name] = {
            "model": rc["model"],
            "parameters": n_params,
            "latency_ms_batch": round(lat_ms, 3),
            "inference_batch": INFERENCE_BATCH,
            "gpu_peak_mb": round(gpu_peak_mb, 1),
            "device": str(device),
            "test_count": len(test_ds),
        }
        print(f"{name:12} params={n_params:>8,}  lat={lat_ms:6.2f} ms/batch  "
              f"gpu_peak={gpu_peak_mb:7.1f} MB")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
