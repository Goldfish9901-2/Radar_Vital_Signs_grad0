"""Unified 12x3 benchmark aggregator (v3).

Reads per-sample predictions from the uniform layout written by queue_all.sh:

    model_outputs/<model>/<DATASET>/predictions_test.csv

and produces, for the full 12-backbone x 3-dataset matrix:

  * overall metrics per (model, dataset): MAE / RMSE / Pearson r / within-x% /
    label vs pred mean, from per-sample rows (via metrics.aggregate).
  * per-subject MAE/RMSE summary per (model, dataset) (stats.per_subject_summary),
    so the "subject-level" question is answered directly.
  * significance: paired Wilcoxon across all models within a dataset +
    Benjamini-Hochberg q-values (stats.wilcoxon_matrix).
  * bootstrap 95% CI on MAE per (model, dataset) (stats.bootstrap_mae_ci).
  * cross-dataset table: MAE per model across FTU / BGT60TR13C / PhysDrive and
    the FTU->target degradation (answers the cross-dataset generalization question;
    the source-only vs target *training* matrix is produced separately by
    run_cross_dataset.py).

Missing (model, dataset) cells are skipped, so this runs incrementally as the
queue fills in.

Usage:
  uv run python experiment_runs/make_benchmark_v3.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.factory import MODEL_CHOICES  # noqa: E402
from src.training.common import stats as S  # noqa: E402

MODEL_OUTPUTS = ROOT / "model_outputs"
BENCH = ROOT / "experiment_runs" / "benchmark_v3"
TABLES = BENCH / "tables"
DISPLAY = {
    "contiformer": "ContiFormer", "cycleformer": "CycleFormer", "dlinear": "DLinear",
    "heart_timemixer": "HeartTimeMixer", "mamba": "Mamba", "nlinear": "NLinear",
    "patchtst": "PatchTST", "tcn": "TCN", "timesnet": "TimesNet",
    "transformer": "Transformer", "tslanet": "TSLANet", "tsmixer": "TSMixer",
    "xlstm": "xLSTM",
}
DATASETS = ["FTU", "BGT60TR13C", "PhysDrive"]


def _load(model: str, ds: str):
    p = MODEL_OUTPUTS / model / ds / "predictions_test.csv"
    if not p.exists():
        return None
    return S.load_predictions(p, model)


def overall_row(model: str, ds: str, rows) -> dict:
    ov = S.aggregate(rows)["overall"]
    subj = S.per_subject_summary(rows)
    ci = S.bootstrap_mae_ci(rows, n_boot=1000, seed=42)
    abs_err = [r["abs_error_bpm"] for r in rows]
    return {
        "model": model,
        "method": DISPLAY.get(model, model),
        "dataset": ds,
        "n": int(ov["count"]),
        "mae_bpm": round(ov["mae_bpm"], 3),
        "rmse_bpm": round(ov["rmse_bpm"], 3),
        "pearson_r": round(ov["pearson_r"], 4),
        "within_3bpm_pct": round(ov["within_3bpm_percent"], 2),
        "within_5bpm_pct": round(ov["within_5bpm_percent"], 2),
        "within_10bpm_pct": round(ov["within_10bpm_percent"], 2),
        "label_mean_bpm": round(ov["label_mean_bpm"], 2),
        "pred_mean_bpm": round(ov["pred_mean_bpm"], 2),
        "subjects": subj["across"]["n_subjects"],
        "subject_mean_mae": round(subj["across"]["mean_mae"], 3),
        "subject_std_mae": round(subj["across"]["std_mae"], 3),
        "mae_ci_low": round(ci[0], 3),
        "mae_ci_high": round(ci[1], 3),
    }


def write_csv(path: Path, rows: list, cols: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    print(f"scanning {MODEL_OUTPUTS} for 12x3 predictions ...")

    # (model, dataset) -> rows
    cells = {}
    for model in MODEL_CHOICES:
        for ds in DATASETS:
            rows = _load(model, ds)
            if rows:
                cells[(model, ds)] = rows
    print(f"found {len(cells)} (model, dataset) prediction sets")

    # --- overall + per-subject + bootstrap ---
    overall_rows = []
    per_subject_paths = []
    for (model, ds), rows in cells.items():
        overall_rows.append(overall_row(model, ds, rows))
        # per-subject table per cell
        subj = S.per_subject_summary(rows)
        spath = TABLES / "per_subject" / f"{model}__{ds}.csv"
        write_csv(
            spath,
            [
                {"subject": pid, "mae_bpm": round(v["mae_bpm"], 3),
                 "rmse_bpm": round(v["rmse_bpm"], 3), "n": int(v["n"])}
                for pid, v in sorted(subj["subjects"].items())
            ],
            ["subject", "mae_bpm", "rmse_bpm", "n"],
        )
        per_subject_paths.append(spath)
    overall_rows.sort(key=lambda r: (r["dataset"], r["mae_bpm"]))
    write_csv(
        TABLES / "overall_by_model_dataset.csv",
        overall_rows,
        ["model", "method", "dataset", "n", "subjects", "mae_bpm", "rmse_bpm",
         "pearson_r", "within_3bpm_pct", "within_5bpm_pct", "within_10bpm_pct",
         "label_mean_bpm", "pred_mean_bpm", "subject_mean_mae", "subject_std_mae",
         "mae_ci_low", "mae_ci_high"],
    )

    # --- significance per dataset (paired Wilcoxon + BH) ---
    for ds in DATASETS:
        present = [(m, cells[(m, ds)]) for m in MODEL_CHOICES if (m, ds) in cells]
        if len(present) < 2:
            continue
        models = [m for m, _ in present]
        _models, pairs, pvals, qvals = S.wilcoxon_matrix({m: rows for m, rows in present})
        pmap = {(a, b): p for (a, b), p in zip(pairs, pvals)}
        qmap = {(a, b): q for (a, b), q in zip(pairs, qvals)}
        sig_rows = []
        for (a, b), _p in zip(pairs, pvals):
            sig_rows.append({
                "dataset": ds,
                "model_a": a, "method_a": DISPLAY.get(a, a),
                "model_b": b, "method_b": DISPLAY.get(b, b),
                "p_value": round(_p, 6),
                "q_bh": round(qmap.get((a, b), float("nan")), 6),
                "significant_0.05": bool(_p < 0.05),
            })
        sig_rows.sort(key=lambda r: r["p_value"])
        write_csv(
            TABLES / f"significance_{ds}.csv",
            sig_rows,
            ["dataset", "model_a", "method_a", "model_b", "method_b",
             "p_value", "q_bh", "significant_0.05"],
        )

    # --- cross-dataset MAE matrix + FTU->target degradation ---
    cross_rows = []
    for model in MODEL_CHOICES:
        if not any((model, ds) in cells for ds in DATASETS):
            continue
        row = {"model": model, "method": DISPLAY.get(model, model)}
        ftu_mae = None
        for ds in DATASETS:
            if (model, ds) in cells:
                ov = S.aggregate(cells[(model, ds)])["overall"]
                row[f"mae_{ds}"] = round(ov["mae_bpm"], 3)
                row[f"r_{ds}"] = round(ov["pearson_r"], 4)
                if ds == "FTU":
                    ftu_mae = ov["mae_bpm"]
            else:
                row[f"mae_{ds}"] = ""
                row[f"r_{ds}"] = ""
        # degradation relative to FTU (within-dataset training proxy)
        for ds in DATASETS:
            key = f"mae_{ds}"
            if key in row and row[key] != "" and ftu_mae:
                row[f"degrad_{ds}_vs_FTU"] = round(row[key] - ftu_mae, 3)
            else:
                row[f"degrad_{ds}_vs_FTU"] = ""
        cross_rows.append(row)
    write_csv(
        TABLES / "cross_dataset.csv",
        cross_rows,
        ["model", "method",
         "mae_FTU", "r_FTU", "mae_BGT60TR13C", "r_BGT60TR13C",
         "mae_PhysDrive", "r_PhysDrive",
         "degrad_BGT60TR13C_vs_FTU", "degrad_PhysDrive_vs_FTU"],
    )

    # --- report ---
    lines = ["# Unified Benchmark v3 — 12 backbones × 3 datasets\n"]
    lines.append(f"Cells populated: {len(cells)} / {len(MODEL_CHOICES)*len(DATASETS)} "
                 f"(missing cells are still training).\n")
    lines.append("## Overall (sorted by dataset, then MAE)\n")
    lines.append("| model | dataset | n | MAE | RMSE | r | ≤5% | ≤10% | subj mean MAE | MAE 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in overall_rows:
        lines.append(
            f"| {r['method']} | {r['dataset']} | {r['n']} | {r['mae_bpm']} | "
            f"{r['rmse_bpm']} | {r['pearson_r']} | {r['within_5bpm_pct']} | "
            f"{r['within_10bpm_pct']} | {r['subject_mean_mae']} | "
            f"[{r['mae_ci_low']}, {r['mae_ci_high']}] |"
        )
    lines.append("\n## Cross-dataset MAE (FTU → BGT60 / PhysDrive)\n")
    lines.append("| model | FTU | BGT60 | PhysDrive | ΔBGT60 | ΔPhysDrive |")
    lines.append("|---|---|---|---|---|---|")
    for r in cross_rows:
        lines.append(
            f"| {r['method']} | {r.get('mae_FTU','')} | {r.get('mae_BGT60TR13C','')} | "
            f"{r.get('mae_PhysDrive','')} | {r.get('degrad_BGT60TR13C_vs_FTU','')} | "
            f"{r.get('degrad_PhysDrive_vs_FTU','')} |"
        )
    lines.append("\n## Significance\n")
    lines.append("Per-dataset paired Wilcoxon across models + Benjamini-Hochberg q-values "
                 "are in `tables/significance_<dataset>.csv`.\n")
    lines.append("## Per-subject\n")
    lines.append(f"Per-subject MAE/RMSE tables: `tables/per_subject/*.csv` "
                 f"({len(per_subject_paths)} cells).\n")
    (BENCH / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {len(overall_rows)} overall rows, "
          f"{len(cross_rows)} cross-dataset rows, per-subject x {len(per_subject_paths)}")
    print(f"report: {BENCH / 'REPORT.md'}")


if __name__ == "__main__":
    main()
