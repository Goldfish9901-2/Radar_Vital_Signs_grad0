#!/usr/bin/env python3
"""Create paper-friendly CSV tables from overall_metrics.csv."""

from __future__ import annotations

import csv
from pathlib import Path


KEEP = [
    "setting",
    "model",
    "source_dataset",
    "target_dataset",
    "mae_bpm",
    "rmse_bpm",
    "pearson_r",
    "within_5bpm_percent",
    "within_10bpm_percent",
]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in KEEP} for row in rows])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "tables"
    rows = list(csv.DictReader((table_dir / "overall_metrics.csv").open(encoding="utf-8")))

    signal = [row for row in rows if row["setting"] == "signal_baseline"]
    source_only = [
        row
        for row in rows
        if row["setting"] == "source_only_or_within" and row["source_dataset"] == "FTU"
    ]
    adaptation = [row for row in rows if row["setting"] == "pseudo_label_adaptation"]
    deep_comparison = source_only + adaptation

    sort_key = lambda row: (row["target_dataset"], row["setting"], row["model"])
    write_rows(table_dir / "signal_baselines.csv", sorted(signal, key=sort_key))
    write_rows(table_dir / "ftu_source_deep_comparison.csv", sorted(deep_comparison, key=sort_key))
    print(f"wrote {table_dir / 'signal_baselines.csv'}")
    print(f"wrote {table_dir / 'ftu_source_deep_comparison.csv'}")


if __name__ == "__main__":
    main()
