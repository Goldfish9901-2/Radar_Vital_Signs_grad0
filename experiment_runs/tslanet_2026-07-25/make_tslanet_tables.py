"""Generate unified-format experiment results for the TSLANet FTU benchmark.

Reads each run's eval artifact (produced by evaluate_model.py) and per-epoch
history (produced by train_model.py) and writes two CSVs in the exact schema
used by experiment_runs/cycleformer_2026-06-23/tables/overall_metrics.csv:

  * tables/overall_metrics.csv   — one row per channel_mode (test-split overall)
  * tables/per_epoch_metrics.csv — one row per (mode, epoch, split)

Run AFTER training + evaluation have produced:
  model_outputs/tslanet_joint_ftu_source/{history.json,eval_test_FTU.json}
  model_outputs/tslanet_official_ftu_source/{history.json,eval_test_FTU.json}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../work/upstream_rw
EXP = Path(__file__).resolve().parent       # .../experiment_runs/tslanet_2026-07-25
MODES = {
    "joint": "tslanet_joint_ftu_source",
    "official": "tslanet_official_ftu_source",
}

OVERALL_COLS = [
    "record_type", "method", "model", "setting", "source_dataset", "target_dataset",
    "split", "count", "mae_bpm", "rmse_bpm", "pearson_r", "within_3bpm_percent",
    "within_5bpm_percent", "within_10bpm_percent", "label_mean_bpm", "pred_mean_bpm",
    "json_path",
]
PER_EPOCH_COLS = ["mode", "epoch", "split", "loss", "mae_bpm", "rmse_bpm", "lr"]


def main() -> None:
    overall_rows = []
    per_epoch_rows = []
    for mode, out_dir in MODES.items():
        out = ROOT / "model_outputs" / out_dir
        eval_json = out / "eval_test_FTU.json"
        history_json = out / "history.json"
        if not eval_json.exists() or not history_json.exists():
            raise SystemExit(f"missing artifacts for mode={mode}: {eval_json} / {history_json}")

        ev = json.loads(eval_json.read_text())
        o = ev["overall"]
        overall_rows.append({
            "record_type": "overall",
            "method": "TSLANet",
            "model": "tslanet",
            "setting": mode,
            "source_dataset": "FTU",
            "target_dataset": "FTU",
            "split": "test",
            "count": o["count"],
            "mae_bpm": o["mae_bpm"],
            "rmse_bpm": o["rmse_bpm"],
            "pearson_r": o["pearson_r"],
            "within_3bpm_percent": o["within_3bpm_percent"],
            "within_5bpm_percent": o["within_5bpm_percent"],
            "within_10bpm_percent": o["within_10bpm_percent"],
            "label_mean_bpm": o["label_mean_bpm"],
            "pred_mean_bpm": o["pred_mean_bpm"],
            "json_path": str(eval_json.resolve()),
        })

        hist = json.loads(history_json.read_text())
        for row in hist:
            for split in ("train", "val"):
                m = row[split]
                per_epoch_rows.append({
                    "mode": mode,
                    "epoch": row["epoch"],
                    "split": split,
                    "loss": m["loss"],
                    "mae_bpm": m["mae_bpm"],
                    "rmse_bpm": m["rmse_bpm"],
                    "lr": row["lr"],
                })

    tables = EXP / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "overall_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OVERALL_COLS)
        w.writeheader()
        w.writerows(overall_rows)
    with open(tables / "per_epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_EPOCH_COLS)
        w.writeheader()
        w.writerows(per_epoch_rows)
    print(f"wrote {tables / 'overall_metrics.csv'} ({len(overall_rows)} rows)")
    print(f"wrote {tables / 'per_epoch_metrics.csv'} ({len(per_epoch_rows)} rows)")


if __name__ == "__main__":
    main()
