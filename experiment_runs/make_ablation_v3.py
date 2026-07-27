"""Factorial analysis of the representation ablation.

Reads model_outputs_ablation/<model>/<rep>/FTU/predictions_test.csv for every
(backbone, representation) cell and answers the paper's core question:

    In cross-dataset radar HR regression, does performance come from the
    BACKBONE or the REPRESENTATION (input features)?

It does this two ways:
  1. A backbone x representation MAE table (the factorial grid).
  2. A two-way ANOVA / variance decomposition on the cell MAE means:
         performance = backbone effect + representation effect + interaction
     reported as percentage of total sum-of-squares. This is the quantitative
     "backbone vs representation" answer a reviewer will ask for.

Backbones in the ablation are 4 paradigm archetypes (linear / conv / attention /
recurrent); representations are the 5 ablation choices. Missing cells are skipped
so the script runs incrementally as the ablation queue progresses.

Output (model_outputs_ablation/analysis/):
    ablation_mae_matrix.csv
    ablation_variance_decomposition.csv
    ablation_report.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.common import stats as S

ABLATION_ROOT = ROOT / "model_outputs_ablation"
OUT = ABLATION_ROOT / "analysis"
# canonical order for readable tables
BACKBONES = ["dlinear", "tcn", "transformer", "xlstm"]
REPS = ["proposed", "edacm_only", "raw_logmag", "raw_real_imag", "edacm_vmd_fixed"]
REP_LABEL = {
    "proposed": "EDACM+HR-AdaVMD (7)",
    "edacm_only": "EDACM-only (1)",
    "raw_logmag": "Raw log-mag (1)",
    "raw_real_imag": "Raw real/imag (2)",
    "edacm_vmd_fixed": "EDACM+VMD-fixed (7)",
}


def load_cell(model: str, rep: str):
    p = ABLATION_ROOT / model / rep / "FTU" / "predictions_test.csv"
    if not p.exists():
        return None
    rows = S.load_predictions(p, model)
    ov = S.aggregate(rows)["overall"]
    return {
        "mae_bpm": ov["mae_bpm"],
        "rmse_bpm": ov["rmse_bpm"],
        "pearson_r": ov["pearson_r"],
        "within_5bpm_pct": ov["within_5bpm_percent"],
        "n": ov["count"],
    }


def two_way_anova(cells: Dict[str, Dict[str, float]]) -> Dict:
    """Variance decomposition of MAE over backbone x representation.

    cells: {(model, rep): mae}. Assumes (near-)balanced grid.
    Returns SS and percentage for backbone / representation / interaction.
    """
    models = sorted({m for (m, _r) in cells})
    reps = sorted({_r for (m, _r) in cells})
    vals = {(m, r): cells[(m, r)] for m in models for r in reps if (m, r) in cells}
    if len(vals) < 2:
        return {}
    ys = list(vals.values())
    grand = mean(ys)
    b_means = {m: mean([vals[(m, r)] for r in reps if (m, r) in vals]) for m in models}
    r_means = {r: mean([vals[(m, r)] for m in models if (m, r) in vals]) for r in reps}
    n_r = len(reps)
    n_b = len(models)
    ssb = n_r * sum((b_means[m] - grand) ** 2 for m in models)
    ssr = n_b * sum((r_means[r] - grand) ** 2 for r in reps)
    ssi = sum((vals[(m, r)] - b_means[m] - r_means[r] + grand) ** 2 for (m, r) in vals)
    sst = sum((y - grand) ** 2 for y in ys)
    # for a balanced grid sst == ssb+ssr+ssi; report proportions of sst
    denom = sst if sst > 0 else 1.0
    return {
        "n_cells": len(vals),
        "grand_mean_mae": round(grand, 3),
        "ss_backbone": round(ssb, 3),
        "ss_representation": round(ssr, 3),
        "ss_interaction": round(ssi, 3),
        "ss_total": round(sst, 3),
        "pct_backbone": round(100 * ssb / denom, 1),
        "pct_representation": round(100 * ssr / denom, 1),
        "pct_interaction": round(100 * ssi / denom, 1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # collect available cells
    grid: Dict[str, Dict[str, float]] = {}
    detailed: List[Dict] = []
    for model in BACKBONES:
        for rep in REPS:
            c = load_cell(model, rep)
            if c is None:
                continue
            grid[(model, rep)] = c["mae_bpm"]
            detailed.append({
                "backbone": model, "representation": rep,
                "rep_label": REP_LABEL.get(rep, rep),
                "mae_bpm": round(c["mae_bpm"], 3),
                "rmse_bpm": round(c["rmse_bpm"], 3),
                "pearson_r": round(c["pearson_r"], 4),
                "within_5bpm_pct": round(c["within_5bpm_pct"], 2),
                "n": c["n"],
            })

    if not grid:
        print("[ablation] no cells found yet; run the ablation queue first.")
        return

    # MAE matrix csv
    matrix_cols = ["backbone"] + [f"{r}" for r in REPS]
    with (OUT / "ablation_mae_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=matrix_cols)
        w.writeheader()
        for model in BACKBONES:
            row = {"backbone": model}
            for r in REPS:
                row[r] = grid.get((model, r), "")
            w.writerow(row)

    decomp = two_way_anova(grid)
    with (OUT / "ablation_variance_decomposition.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["effect", "sum_sq", "pct_of_total"])
        w.writerow(["backbone", decomp.get("ss_backbone", ""), decomp.get("pct_backbone", "")])
        w.writerow(["representation", decomp.get("ss_representation", ""), decomp.get("pct_representation", "")])
        w.writerow(["interaction", decomp.get("ss_interaction", ""), decomp.get("pct_interaction", "")])
        w.writerow(["total", decomp.get("ss_total", ""), 100.0])

    # detailed per-cell csv
    with (OUT / "ablation_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "representation", "rep_label",
                                          "mae_bpm", "rmse_bpm", "pearson_r", "within_5bpm_pct", "n"])
        w.writeheader()
        for d in detailed:
            w.writerow(d)

    # markdown report
    md = ["# Representation Ablation — factorial analysis (FTU)\n",
          f"- cells populated: {len(grid)} / {len(BACKBONES)*len(REPS)}\n",
          "\n## MAE (BPM): backbone x representation\n",
          "| backbone | " + " | ".join(REP_LABEL.get(r, r) for r in REPS) + " |",
          "|---|" + "|".join(["---"] * len(REPS)) + "|"]
    for model in BACKBONES:
        cells_row = []
        for r in REPS:
            v = grid.get((model, r))
            cells_row.append(f"{v:.3f}" if v is not None else "-")
        md.append(f"| {model} | " + " | ".join(cells_row) + " |")

    if decomp:
        md.append("\n## Variance decomposition (two-way ANOVA on cell MAE)\n")
        md.append(f"- grand mean MAE: **{decomp['grand_mean_mae']} BPM** over {decomp['n_cells']} cells\n")
        md.append("| effect | sum of squares | % of total |",
                  "|---|---|---|")
        md.append(f"| backbone | {decomp['ss_backbone']} | **{decomp['pct_backbone']}%** |")
        md.append(f"| representation | {decomp['ss_representation']} | **{decomp['pct_representation']}%** |")
        md.append(f"| interaction | {decomp['ss_interaction']} | {decomp['pct_interaction']}% |")
        dominant = max(
            ("backbone", decomp["pct_backbone"]),
            ("representation", decomp["pct_representation"]),
            ("interaction", decomp["pct_interaction"]),
            key=lambda x: x[1],
        )
        md.append(f"\n**Dominant factor: {dominant[0]} ({dominant[1]}% of variance).**")

    (OUT / "ablation_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[ablation] wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
