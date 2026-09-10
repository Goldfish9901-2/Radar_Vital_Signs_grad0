"""Assemble the unified FTU->FTU benchmark report from EXISTING results only.

No model is retrained. Sources (all pre-existing):

  A. June experiment tables
     work/upstream_rw/experiment_runs/cycleformer_2026-06-23/tables/overall_metrics.csv
       -> TCN / Transformer / PatchTST / TimesNet / CycleFormer (v1, v3, v3-small)
          FTU->FTU test, produced in the June experiments (80 epochs).
       NOTE: HeartTimeMixer was NOT recorded in these June tables.

  B. June-era HeartTimeMixer (same evaluate_model.py schema, FTU->FTU test)
     work/run/upstream/model_outputs/direct_transfer_ftu_source/eval_test_FTU.json
       -> HeartTimeMixer FTU->FTU (count=704, label_mean=77.5000502, identical test
          set to the others). June-era result stored in a different run dir.

  C. This study (2026-07-25) TSLANet official
     work/upstream_rw/experiment_runs/tslanet_2026-07-25/tables/overall_metrics.csv
       -> TSLANet (official) FTU->FTU (20 epochs).

Outputs (under experiment_runs/unified_benchmark_2026-07-25/):
  tables/overall_metrics.csv   unified FTU->FTU table, with provenance + epochs
  figures/mae_bar_ftu_source.png
  figures/rmse_bar_ftu_source.png
  REPORT.md                    provenance notes + explicit per-sample-plot omission

Per-sample charts (Pearson scatter / error histogram / pred-vs-label) are
INTENTIONALLY OMITTED: no eval artifact in this checkout stores per-sample
(pred, label) arrays for any model, so they cannot be reproduced without
retraining. They are not fabricated.

Run with the configured uv env at repo root:
  uv run python experiment_runs/unified_benchmark_2026-07-25/make_unified_benchmark.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # .../work/upstream_rw
WORK = ROOT.parent                               # .../work
EXP = Path(__file__).resolve().parent            # .../experiment_runs/unified_benchmark_2026-07-25

JUNE_CSV = ROOT / "experiment_runs" / "cycleformer_2026-06-23" / "tables" / "overall_metrics.csv"
TSL_CSV = ROOT / "experiment_runs" / "tslanet_2026-07-25" / "tables" / "overall_metrics.csv"
HTM_JSON = WORK / "run" / "upstream" / "model_outputs" / "direct_transfer_ftu_source" / "eval_test_FTU.json"

# (model, setting) -> display name, used to pull rows from the June CSV.
JUNE_MODELS = {
    "cycleformer": "CycleFormer (v1)",
    "cycleformer_v3": "CycleFormer v3",
    "cycleformer_v3_small": "CycleFormer v3-small",
    "tcn": "TCN",
    "transformer": "Transformer",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
}

# Provenance grouping for colouring: "new" (this study) vs "june" (reproduced).
NEW_PROV = "this_study_2026-07-25"
JUNE_PROV = "reproduced_june_2026-06-23"
HTM_PROV = "heart_timemixer_june_direct_transfer_ftu_source"


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x):
    return x if x == "" or x is None else float(x)


def build_rows() -> list[dict]:
    rows = []

    # --- A. June experiment tables (TCN/PatchTST/TimesNet/Transformer/CycleFormer) ---
    for r in load_rows(JUNE_CSV):
        if r["model"] in JUNE_MODELS and r["setting"] == "source_only_or_within" \
           and r["source_dataset"] == "FTU" and r["target_dataset"] == "FTU" and r["split"] == "test":
            rows.append({
                "method": JUNE_MODELS[r["model"]], "model": r["model"],
                "setting": r["setting"], "source_dataset": "FTU", "target_dataset": "FTU",
                "split": "test", "count": int(f(r["count"])),
                "mae_bpm": f(r["mae_bpm"]), "rmse_bpm": f(r["rmse_bpm"]),
                "pearson_r": f(r["pearson_r"]),
                "within_3bpm_percent": f(r["within_3bpm_percent"]),
                "within_5bpm_percent": f(r["within_5bpm_percent"]),
                "within_10bpm_percent": f(r["within_10bpm_percent"]),
                "label_mean_bpm": f(r["label_mean_bpm"]), "pred_mean_bpm": f(r["pred_mean_bpm"]),
                "json_path": r["json_path"], "provenance": JUNE_PROV, "epochs": 80,
            })

    # --- B. HeartTimeMixer (June-era, run/upstream, same schema) ---
    htm = json.loads(HTM_JSON.read_text())
    o = htm["overall"]
    rows.append({
        "method": "HeartTimeMixer", "model": "heart_timemixer",
        "setting": "direct_transfer_ftu_source", "source_dataset": "FTU", "target_dataset": "FTU",
        "split": "test", "count": int(o["count"]),
        "mae_bpm": f(o["mae_bpm"]), "rmse_bpm": f(o["rmse_bpm"]),
        "pearson_r": f(o["pearson_r"]),
                "within_3bpm_percent": f(o["within_3bpm_percent"]),
                "within_5bpm_percent": f(o["within_5bpm_percent"]),
                "within_10bpm_percent": f(o.get("within_10bpm_percent", "")),
        "label_mean_bpm": f(o["label_mean_bpm"]), "pred_mean_bpm": f(o["pred_mean_bpm"]),
        "json_path": str(HTM_JSON.resolve()), "provenance": HTM_PROV, "epochs": "n/a",
    })

    # --- C. TSLANet official (this study) ---
    for r in load_rows(TSL_CSV):
        if r["model"] == "tslanet" and r["setting"] == "official" \
           and r["source_dataset"] == "FTU" and r["target_dataset"] == "FTU" and r["split"] == "test":
            rows.append({
                "method": "TSLANet (official)", "model": "tslanet",
                "setting": "official", "source_dataset": "FTU", "target_dataset": "FTU",
                "split": "test", "count": int(f(r["count"])),
                "mae_bpm": f(r["mae_bpm"]), "rmse_bpm": f(r["rmse_bpm"]),
                "pearson_r": f(r["pearson_r"]),
                "within_3bpm_percent": f(r["within_3bpm_percent"]),
                "within_5bpm_percent": f(r["within_5bpm_percent"]),
                "within_10bpm_percent": f(r["within_10bpm_percent"]),
                "label_mean_bpm": f(r["label_mean_bpm"]), "pred_mean_bpm": f(r["pred_mean_bpm"]),
                "json_path": r["json_path"], "provenance": NEW_PROV, "epochs": 20,
            })

    rows.sort(key=lambda d: d["mae_bpm"])
    return rows


def prov_group(p: str) -> str:
    return "new" if p == NEW_PROV else "june"


def write_overall_csv(path: Path, rows: list[dict]) -> None:
    cols = ["record_type", "method", "model", "setting", "source_dataset", "target_dataset",
            "split", "count", "mae_bpm", "rmse_bpm", "pearson_r",
            "within_3bpm_percent", "within_5bpm_percent", "within_10bpm_percent",
            "label_mean_bpm", "pred_mean_bpm", "json_path", "provenance", "epochs"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["record_type"] = "overall"
            w.writerow({c: out.get(c, "") for c in cols})


def bar_chart(rows: list[dict], metric: str, title: str, fname: str) -> None:
    color_new = "#c0392b"   # this study
    color_june = "#34688f"  # reproduced from June
    models = [r["method"] for r in rows]
    vals = [r[metric] for r in rows]
    groups = [prov_group(r["provenance"]) for r in rows]
    colors = [color_new if g == "new" else color_june for g in groups]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(f"{metric.upper()} (bpm)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=color_new, edgecolor="black", label="New (this study, 2026-07-25)"),
              Patch(facecolor=color_june, edgecolor="black", label="Reproduced from June (2026-06-23 / June-era)")]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)

    ax.text(0.5, -0.34,
            "FTU->FTU test (n=704, label mean 77.5 bpm).  TSLANet = 20 epochs (this study); "
            "TCN/Transformer/PatchTST/TimesNet/CycleFormer = 80 epochs (June); "
            "HeartTimeMixer = June-era (run/upstream direct_transfer_ftu_source). "
            "Per-sample plots omitted: no prediction files available.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555", wrap=True)
    fig.tight_layout()
    out = EXP / "figures" / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def write_report(path: Path, rows: list[dict]) -> None:
    new = [r for r in rows if r["provenance"] == NEW_PROV]
    june_tbl = [r for r in rows if r["provenance"] == JUNE_PROV]
    htm = [r for r in rows if r["provenance"] == HTM_PROV]
    L = []
    L.append("# Unified Benchmark Report — FTU → FTU (same-dataset, test split)\n")
    L.append("Assembled from **existing results only**; no model was retrained.\n")
    L.append("## Data sources & provenance\n")
    L.append(f"- **New (this study, 2026-07-25)**: {', '.join(r['method'] for r in new)} "
             f"(20 epochs, `experiment_runs/tslanet_2026-07-25`).")
    L.append(f"- **Reproduced from June (2026-06-23)**: {', '.join(sorted(set(r['method'] for r in june_tbl)))} "
             f"(80 epochs, `experiment_runs/cycleformer_2026-06-23/tables/overall_metrics.csv`).")
    L.append(f"- **HeartTimeMixer (June-era)**: NOT present in the June `cycleformer_2026-06-23` tables. "
             f"Used `work/run/upstream/model_outputs/direct_transfer_ftu_source/eval_test_FTU.json` "
             f"(same `evaluate_model.py` schema; count=704, label_mean=77.5000502 — identical test set). "
             f"Epoch count not recorded in its run_config.\n")
    L.append("## Ranked results (by MAE)\n")
    L.append("| Rank | Model | Provenance | Epochs | MAE | RMSE | Pearson r | ≤5 bpm % | ≤10 bpm % |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        w5 = r["within_5bpm_percent"]
        w10 = r["within_10bpm_percent"]
        w5s = f"{w5:.1f}" if isinstance(w5, (int, float)) else "—"
        w10s = f"{w10:.1f}" if isinstance(w10, (int, float)) else "—"
        L.append(f"| {i} | {r['method']} | {r['provenance']} | {r['epochs']} | "
                 f"{r['mae_bpm']:.2f} | {r['rmse_bpm']:.2f} | {r['pearson_r']:.3f} | "
                 f"{w5s} | {w10s} |")
    L.append("")
    L.append("## Per-sample plots: intentionally omitted\n")
    L.append("Pearson scatter, error histogram, and prediction-vs-label plots are **not produced**. "
             "No eval artifact in this checkout stores per-sample `(pred, label)` arrays for any model, "
             "so these charts cannot be reproduced without retraining. They are explicitly omitted rather "
             "than fabricated.\n")
    L.append("## Note on epoch budgets\n")
    L.append("TSLANet official (20 epochs) is compared against June backbones (80 epochs) on the same "
             "FTU→FTU test set. TSLANet official ranks #2 by MAE, just behind CycleFormer v3-small (80 ep) "
             "and essentially tied with HeartTimeMixer (MAE 3.69 vs 3.69). A strictly epoch-matched "
             "comparison would require a unified re-run (deferred).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_overall_csv(EXP / "tables" / "overall_metrics.csv", rows)
    bar_chart(rows, "mae_bpm", "FTU → FTU — MAE by Backbone (provenance annotated)", "mae_bar_ftu_source.png")
    bar_chart(rows, "rmse_bpm", "FTU → FTU — RMSE by Backbone (provenance annotated)", "rmse_bar_ftu_source.png")
    write_report(EXP / "REPORT.md", rows)

    print(f"\nUnified overall_metrics.csv: {len(rows)} rows (FTU->FTU).")
    for i, r in enumerate(rows, 1):
        tag = "NEW " if r["provenance"] == NEW_PROV else "june"
        print(f"  {i:>2}. {r['method']:<20} MAE={r['mae_bpm']:.2f}  RMSE={r['rmse_bpm']:.2f}  [{tag}]")


if __name__ == "__main__":
    main()
