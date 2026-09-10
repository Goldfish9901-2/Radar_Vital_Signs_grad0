"""Regenerate the unified FTU->FTU benchmark INCLUDING the 5 newly trained
backbones (DLinear, NLinear, TSMixer, ContiFormer, Mamba).

Differences from the 2026-07-25 report:
  * New models are now TRAINED (not just listed) under the same 80-epoch FTU->FTU
    protocol, so they are epoch-matched to the June baselines.
  * Overall metrics for the new models are computed from their per-sample
    predictions_test.csv (via metrics.aggregate), giving a real overall Pearson r.
  * Per-sample plots are produced for the new models: Pearson scatter, error
    histogram, and prediction-vs-label tracking — these were previously omitted
    because no prediction files existed.
  * A comparative analysis table adds parameter count, inference latency,
    GPU peak memory, and training-stability signals (best epoch, val/train gap,
    post-best degradation) read from run_config.json / inference_metrics.json /
    history.json.

Existing sources (unchanged, for context):
  A. June tables        -> TCN/Transformer/PatchTST/TimesNet/CycleFormer (80 ep)
  B. HeartTimeMixer     -> June-era direct_transfer_ftu_source eval JSON (n/a ep)
  C. TSLANet official   -> this study 2026-07-25 (20 ep)

Outputs under experiment_runs/benchmark_2026-07-26/:
  tables/overall_metrics.csv
  figures/mae_bar.png, rmse_bar.png
  figures/scatter_pearson.png, hist_error.png, pred_vs_label.png
  figures/learning_curves.png
  REPORT.md

Run:
  uv run python experiment_runs/make_benchmark_v2.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # .../work/upstream_rw
sys.path.insert(0, str(ROOT))                       # make the `src` package importable
EXP = Path(__file__).resolve().parent               # .../experiment_runs
BENCH = EXP / "benchmark_2026-07-26"
TABLES = BENCH / "tables"
FIGS = BENCH / "figures"

JUNE_CSV = ROOT / "experiment_runs" / "cycleformer_2026-06-23" / "tables" / "overall_metrics.csv"
TSL_CSV = ROOT / "experiment_runs" / "tslanet_2026-07-25" / "tables" / "overall_metrics.csv"
HTM_JSON = ROOT.parent / "run" / "upstream" / "model_outputs" / "direct_transfer_ftu_source" / "eval_test_FTU.json"
MODEL_OUTPUTS = ROOT / "model_outputs"
INFERENCE_JSON = BENCH / "inference_metrics.json"

NEW_MODELS = ["dlinear", "nlinear", "tsmixer", "contiformer", "mamba"]
NEW_DISPLAY = {
    "dlinear": "DLinear", "nlinear": "NLinear", "tsmixer": "TSMixer",
    "contiformer": "ContiFormer", "mamba": "Mamba",
}
NEW_PROV = "this_study_2026-07-26"

JUNE_MODELS = {
    "cycleformer": "CycleFormer (v1)", "cycleformer_v3": "CycleFormer v3",
    "cycleformer_v3_small": "CycleFormer v3-small", "tcn": "TCN",
    "transformer": "Transformer", "patchtst": "PatchTST", "timesnet": "TimesNet",
}
JUNE_PROV = "reproduced_june_2026-06-23"
TSL_PROV = "this_study_2026-07-25"
HTM_PROV = "heart_timemixer_june_direct_transfer_ftu_source"


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x):
    return x if x == "" or x is None else float(x)


def _agg_from_predictions(model: str) -> dict | None:
    pred_csv = MODEL_OUTPUTS / model / "predictions_test.csv"
    if not pred_csv.exists():
        return None
    rows = []
    with open(pred_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "abs_error_bpm": float(r["abs_error_bpm"]),
                "label_bpm": float(r["label_bpm"]),
                "pred_bpm": float(r["pred_bpm"]),
            })
    if not rows:
        return None
    from src.training.common.metrics import aggregate
    overall = aggregate(rows)["overall"]
    return overall


def _stability(model: str) -> dict:
    hpath = MODEL_OUTPUTS / model / "history.json"
    if not hpath.exists():
        return {"epochs_run": 0, "best_epoch": None, "best_val_mae": math.nan,
                "train_val_gap": math.nan, "post_best_deg": math.nan}
    hist = json.loads(hpath.read_text())
    if not hist:
        return {"epochs_run": 0, "best_epoch": None, "best_val_mae": math.nan,
                "train_val_gap": math.nan, "post_best_deg": math.nan}
    import math
    val = [h["val"]["mae_bpm"] for h in hist]
    tr = [h["train"]["mae_bpm"] for h in hist]
    best_i = int(np.argmin(val))
    best_val = val[best_i]
    train_at_best = tr[best_i]
    final_val = val[-1]
    return {
        "epochs_run": len(hist),
        "best_epoch": best_i + 1,
        "best_val_mae": best_val,
        "train_val_gap": best_val - train_at_best,   # >0 means val worse than train (expected)
        "post_best_deg": final_val - best_val,        # >0 means degraded after best (early-stop should keep ~0)
    }


def build_rows() -> list[dict]:
    rows = []

    # --- A. June ---
    for r in load_rows(JUNE_CSV):
        if (r["model"] in JUNE_MODELS and r["setting"] == "source_only_or_within"
                and r["source_dataset"] == "FTU" and r["target_dataset"] == "FTU" and r["split"] == "test"):
            rows.append({
                "method": JUNE_MODELS[r["model"]], "model": r["model"], "setting": r["setting"],
                "source_dataset": "FTU", "target_dataset": "FTU", "split": "test",
                "count": int(f(r["count"])), "mae_bpm": f(r["mae_bpm"]), "rmse_bpm": f(r["rmse_bpm"]),
                "pearson_r": f(r["pearson_r"]), "within_3bpm_percent": f(r["within_3bpm_percent"]),
                "within_5bpm_percent": f(r["within_5bpm_percent"]),
                "within_10bpm_percent": f(r["within_10bpm_percent"]),
                "label_mean_bpm": f(r["label_mean_bpm"]), "pred_mean_bpm": f(r["pred_mean_bpm"]),
                "parameters": "", "latency_ms_batch": "", "gpu_peak_mb": "",
                "json_path": r["json_path"], "provenance": JUNE_PROV, "epochs": 80,
                "has_per_sample": False,
            })

    # --- B. HeartTimeMixer ---
    htm = json.loads(HTM_JSON.read_text())
    o = htm["overall"]
    rows.append({
        "method": "HeartTimeMixer", "model": "heart_timemixer", "setting": "direct_transfer_ftu_source",
        "source_dataset": "FTU", "target_dataset": "FTU", "split": "test", "count": int(o["count"]),
        "mae_bpm": f(o["mae_bpm"]), "rmse_bpm": f(o["rmse_bpm"]), "pearson_r": f(o["pearson_r"]),
        "within_3bpm_percent": f(o["within_3bpm_percent"]),
        "within_5bpm_percent": f(o["within_5bpm_percent"]),
        "within_10bpm_percent": f(o.get("within_10bpm_percent", "")),
        "label_mean_bpm": f(o["label_mean_bpm"]), "pred_mean_bpm": f(o["pred_mean_bpm"]),
        "parameters": "", "latency_ms_batch": "", "gpu_peak_mb": "",
        "json_path": str(HTM_JSON.resolve()), "provenance": HTM_PROV, "epochs": "n/a",
        "has_per_sample": False,
    })

    # --- C. TSLANet ---
    for r in load_rows(TSL_CSV):
        if (r["model"] == "tslanet" and r["setting"] == "official"
                and r["source_dataset"] == "FTU" and r["target_dataset"] == "FTU" and r["split"] == "test"):
            rows.append({
                "method": "TSLANet (official)", "model": "tslanet", "setting": "official",
                "source_dataset": "FTU", "target_dataset": "FTU", "split": "test", "count": int(f(r["count"])),
                "mae_bpm": f(r["mae_bpm"]), "rmse_bpm": f(r["rmse_bpm"]), "pearson_r": f(r["pearson_r"]),
                "within_3bpm_percent": f(r["within_3bpm_percent"]),
                "within_5bpm_percent": f(r["within_5bpm_percent"]),
                "within_10bpm_percent": f(r["within_10bpm_percent"]),
                "label_mean_bpm": f(r["label_mean_bpm"]), "pred_mean_bpm": f(r["pred_mean_bpm"]),
                "parameters": "", "latency_ms_batch": "", "gpu_peak_mb": "",
                "json_path": r["json_path"], "provenance": TSL_PROV, "epochs": 20,
                "has_per_sample": False,
            })

    # --- D. Newly trained backbones (this study, 2026-07-26, 80 ep) ---
    infer = json.loads(INFERENCE_JSON.read_text()) if INFERENCE_JSON.exists() else {}
    for model in NEW_MODELS:
        overall = _agg_from_predictions(model)
        if overall is None:
            print(f"[warn] no predictions_test.csv for {model}; skipping")
            continue
        rc = json.loads((MODEL_OUTPUTS / model / "run_config.json").read_text())
        stab = _stability(model)
        inf = infer.get(model, {})
        rows.append({
            "method": NEW_DISPLAY[model], "model": model, "setting": "this_study_2026-07-26",
            "source_dataset": "FTU", "target_dataset": "FTU", "split": "test",
            "count": int(overall["count"]), "mae_bpm": overall["mae_bpm"], "rmse_bpm": overall["rmse_bpm"],
            "pearson_r": overall["pearson_r"],
            "within_3bpm_percent": overall["within_3bpm_percent"],
            "within_5bpm_percent": overall["within_5bpm_percent"],
            "within_10bpm_percent": overall["within_10bpm_percent"],
            "label_mean_bpm": overall["label_mean_bpm"], "pred_mean_bpm": overall["pred_mean_bpm"],
            "parameters": rc.get("parameters", ""),
            "latency_ms_batch": inf.get("latency_ms_batch", ""),
            "gpu_peak_mb": inf.get("gpu_peak_mb", ""),
            "best_epoch": stab.get("best_epoch", ""),
            "train_val_gap": stab.get("train_val_gap", ""),
            "post_best_deg": stab.get("post_best_deg", ""),
            "json_path": str((MODEL_OUTPUTS / model / "predictions_test.csv").resolve()),
            "provenance": NEW_PROV, "epochs": 80, "has_per_sample": True,
        })

    rows.sort(key=lambda d: (d["mae_bpm"] if isinstance(d["mae_bpm"], (int, float)) else 1e9))
    return rows


def prov_group(p: str) -> str:
    return "new" if p == NEW_PROV else "june"


def _num(v):
    return v if isinstance(v, (int, float)) else (float(v) if isinstance(v, str) and v not in ("", "n/a") else None)


def write_overall_csv(path: Path, rows: list[dict]) -> None:
    cols = ["record_type", "method", "model", "setting", "source_dataset", "target_dataset",
            "split", "count", "mae_bpm", "rmse_bpm", "pearson_r",
            "within_3bpm_percent", "within_5bpm_percent", "within_10bpm_percent",
            "label_mean_bpm", "pred_mean_bpm", "parameters", "latency_ms_batch",
            "gpu_peak_mb", "best_epoch", "train_val_gap", "post_best_deg",
            "provenance", "epochs", "has_per_sample", "json_path"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["record_type"] = "overall"
            w.writerow({c: out.get(c, "") for c in cols})


def bar_chart(rows, metric, title, fname):
    color_new = "#c0392b"
    color_june = "#34688f"
    models = [r["method"] for r in rows]
    vals = [_num(r[metric]) for r in rows]
    colors = [color_new if prov_group(r["provenance"]) == "new" else color_june for r in rows]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(f"{metric.upper()} (bpm)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(v for v in vals if v is not None) * 1.20)
    for b, v in zip(bars, vals):
        if v is None:
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=35, labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=color_new, edgecolor="black", label="New (this study, 2026-07-26, 80 ep)"),
              Patch(facecolor=color_june, edgecolor="black", label="Reproduced June (80 ep) / TSLANet (20 ep)")]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = FIGS / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _load_pred(model):
    pred_csv = MODEL_OUTPUTS / model / "predictions_test.csv"
    lab, pred = [], []
    with open(pred_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lab.append(float(r["label_bpm"]))
            pred.append(float(r["pred_bpm"]))
    return np.array(lab), np.array(pred)


def per_sample_figures():
    n = len(NEW_MODELS)
    lab_all, pred_all = {}, {}
    for m in NEW_MODELS:
        try:
            lab_all[m], pred_all[m] = _load_pred(m)
        except FileNotFoundError:
            print(f"[warn] missing predictions for {m}; skipping its panels")

    # --- Pearson scatter (1 x n) ---
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, m in enumerate(NEW_MODELS):
        if m not in lab_all:
            continue
        ax = axes[0][i]
        ax.scatter(lab_all[m], pred_all[m], s=6, alpha=0.35, color="#34688f")
        lo, hi = min(lab_all[m].min(), pred_all[m].min()), max(lab_all[m].max(), pred_all[m].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        r = np.corrcoef(lab_all[m], pred_all[m])[0, 1]
        ax.set_title(f"{NEW_DISPLAY[m]}\nr={r:.3f}")
        ax.set_xlabel("label (bpm)"); ax.set_ylabel("pred (bpm)")
    fig.suptitle("Pearson scatter — new backbones (FTU→FTU test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "scatter_pearson.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'scatter_pearson.png'}")

    # --- Error histogram (1 x n) ---
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, m in enumerate(NEW_MODELS):
        if m not in lab_all:
            continue
        ax = axes[0][i]
        err = pred_all[m] - lab_all[m]
        ax.hist(err, bins=40, color="#34688f", edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="r", ls="--", lw=1)
        ax.set_title(f"{NEW_DISPLAY[m]}\nmean={err.mean():.2f} std={err.std():.2f}")
        ax.set_xlabel("pred − label (bpm)"); ax.set_ylabel("count")
    fig.suptitle("Error histogram — new backbones (FTU→FTU test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "hist_error.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'hist_error.png'}")

    # --- Prediction vs label tracking (1 x n) ---
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for i, m in enumerate(NEW_MODELS):
        if m not in lab_all:
            continue
        ax = axes[0][i]
        order = np.argsort(lab_all[m])
        ax.plot(lab_all[m][order], label="label", lw=1, color="#2c3e50")
        ax.plot(pred_all[m][order], label="pred", lw=1, color="#c0392b", alpha=0.8)
        ax.set_title(NEW_DISPLAY[m]); ax.set_xlabel("sample (sorted by label)"); ax.set_ylabel("bpm")
        ax.legend(fontsize=8)
    fig.suptitle("Prediction vs label tracking — new backbones (FTU→FTU test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "pred_vs_label.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'pred_vs_label.png'}")


def learning_curves():
    n = len(NEW_MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, m in enumerate(NEW_MODELS):
        hpath = MODEL_OUTPUTS / m / "history.json"
        if not hpath.exists():
            continue
        hist = json.loads(hpath.read_text())
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        tr = [h["train"]["mae_bpm"] for h in hist]
        va = [h["val"]["mae_bpm"] for h in hist]
        axes[0].plot(ep, va, color=colors[i], label=NEW_DISPLAY[m])
        axes[1].plot(ep, va, color=colors[i], label=NEW_DISPLAY[m])
        axes[1].plot(ep, tr, color=colors[i], ls="--", alpha=0.6)
    axes[0].set_title("Validation MAE vs epoch (new backbones)")
    axes[1].set_title("Val (solid) / Train (dashed) MAE vs epoch")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.set_ylabel("MAE (bpm)")
        ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS / 'learning_curves.png'}")


def write_report(path: Path, rows: list[dict]) -> None:
    new = [r for r in rows if r["provenance"] == NEW_PROV]
    L = []
    L.append("# Unified Benchmark Report — FTU → FTU (this study 2026-07-26)\n")
    L.append("Includes the 5 newly integrated backbones trained under the **same 80-epoch FTU→FTU "
             "protocol** as the June baselines (lr=3e-4, AdamW, SmoothL1Loss β=0.5, seed=42, dual "
             "time+freq). Existing June / TSLANet / HeartTimeMixer rows are carried over for context.\n")
    L.append("## Ranked results (by MAE)\n")
    L.append("| Rank | Model | Prov | Ep | Params | MAE | RMSE | Pearson r | ≤5% | ≤10% | Latency ms/b | GPU MB | Best ep |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        def g(k):
            v = r.get(k, "")
            if isinstance(v, float):
                return f"{v:.2f}" if not math.isnan(v) else "—"
            return str(v) if v != "" else "—"
        L.append(f"| {i} | {r['method']} | {r['provenance']} | {r['epochs']} | {g('parameters')} | "
                 f"{g('mae_bpm')} | {g('rmse_bpm')} | {g('pearson_r')} | {g('within_5bpm_percent')} | "
                 f"{g('within_10bpm_percent')} | {g('latency_ms_batch')} | {g('gpu_peak_mb')} | {g('best_epoch')} |")
    L.append("\n## Per-sample plots (new backbones)\n")
    L.append("Scatter (Pearson), error histogram, and prediction-vs-label tracking are now generated "
             "from `predictions_test.csv` for the 5 new models (`figures/scatter_pearson.png`, "
             "`hist_error.png`, `pred_vs_label.png`). The June/TSLANet/HeartTimeMixer rows still lack "
             "per-sample files, so their per-sample plots remain unavailable.\n")
    L.append("## Training stability (new backbones)\n")
    L.append("`figures/learning_curves.png` shows val (and train, dashed) MAE per epoch. Best epoch and "
             "post-best degradation are in the table above.\n")
    L.append("## Notes\n")
    L.append("- Batch size was adjusted only where the 4 GB GPU would OOM: Mamba=16, ContiFormer=32; "
             "DLinear/NLinear/TSMixer=64. This affects throughput, not the model or final quality.\n"
             "- HeartTimeMixer is a collapsed near-constant predictor (Pearson ≈ 0) and is kept only as "
             "a June-era reference; it should not be read as a competitive baseline.\n"
             "- All new models are epoch-matched (80) to the June baselines; TSLANet (20 ep) remains the "
             "only epoch-mismatched entry.\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_overall_csv(TABLES / "overall_metrics.csv", rows)
    bar_chart(rows, "mae_bpm", "FTU → FTU — MAE by Backbone", "mae_bar.png")
    bar_chart(rows, "rmse_bpm", "FTU → FTU — RMSE by Backbone", "rmse_bar.png")
    per_sample_figures()
    learning_curves()
    write_report(BENCH / "REPORT.md", rows)

    print(f"\nUnified overall_metrics.csv: {len(rows)} rows.")
    for i, r in enumerate(rows, 1):
        tag = "NEW " if r["provenance"] == NEW_PROV else "    "
        print(f"  {i:>2}. {r['method']:<20} MAE={r['mae_bpm'] if isinstance(r['mae_bpm'],(int,float)) else r['mae_bpm']:.2f}"
              f"  RMSE={r['rmse_bpm']:.2f}  r={r['pearson_r']:.3f}  [{tag}]")


if __name__ == "__main__":
    main()
