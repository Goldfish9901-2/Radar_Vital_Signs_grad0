"""Rigorous statistical analysis for heart-rate experiments.

Builds on ``src.training.common.metrics`` (``aggregate``, ``participant_id``) to
add the statistics needed for a defensible backbone comparison:

* per-subject summaries (mean +/- std of MAE across participants),
* paired significance testing across backbone pairs (Wilcoxon signed-rank,
  Benjamini-Hochberg corrected across all pairs),
* bootstrap 95% confidence intervals on MAE.

The functions operate on the same flat ``list[dict]`` prediction rows that
``metrics.aggregate`` consumes (one dict per test window). A small ``__main__``
driver can analyse every ``model_outputs/<model>/predictions_test.csv`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable whether this file is run as a script or imported.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import math
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from src.training.common.metrics import aggregate, participant_id

DEFAULT_KEY: Tuple[str, ...] = ("dataset", "sample_tag")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_predictions(path: str, model: str | None = None) -> List[Dict[str, Any]]:
    """Load a ``predictions_test.csv`` into a list of row dicts.

    Numeric columns (``label_bpm``, ``pred_bpm``, ``abs_error_bpm``) are coerced
    to ``float``. If ``model`` is given it is attached as ``row["model"]``.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row: Dict[str, Any] = dict(r)
            for k in ("label_bpm", "pred_bpm", "abs_error_bpm"):
                try:
                    row[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
            if model is not None:
                row["model"] = model
            rows.append(row)
    return rows


def load_all_predictions(model_outputs_dir: str) -> List[Dict[str, Any]]:
    """Load every ``<model>/predictions_test.csv`` directly under ``model_outputs_dir``.

    Returns rows tagged with ``row["model"]`` taken from the immediate sub-directory
    name. Used by the ``__main__`` driver for an ad-hoc analysis.
    """
    rows: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(model_outputs_dir)):
        csv_path = os.path.join(model_outputs_dir, entry, "predictions_test.csv")
        if os.path.isfile(csv_path):
            rows.extend(load_predictions(csv_path, model=entry))
    return rows


# --------------------------------------------------------------------------- #
# Per-subject statistics
# --------------------------------------------------------------------------- #
def per_subject_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate by ``participant_id`` and report across-subject mean +/- std.

    Returns ``{"subjects": {pid: {mae_bpm, rmse_bpm, n}}, "across": {...}}``.
    The ``across`` block holds ``mean_mae``, ``std_mae``, ``mean_rmse``,
    ``std_rmse`` (sample std, ddof=1) and ``n_subjects``.
    """
    buckets = aggregate(rows, "participant_id")
    per: Dict[str, Any] = {}
    maes: List[float] = []
    rmses: List[float] = []
    for pid, m in buckets.items():
        if pid == "overall":
            continue
        per[pid] = {"mae_bpm": m["mae_bpm"], "rmse_bpm": m["rmse_bpm"], "n": m["count"]}
        maes.append(m["mae_bpm"])
        rmses.append(m["rmse_bpm"])
    out: Dict[str, Any] = {"subjects": per}
    if maes:
        out["across"] = {
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            "n_subjects": len(maes),
        }
    return out


# --------------------------------------------------------------------------- #
# Paired significance testing
# --------------------------------------------------------------------------- #
def align_pairs(
    model_rows: Dict[str, List[Dict[str, Any]]],
    key: Sequence[str] = DEFAULT_KEY,
) -> Dict[str, Dict[Tuple, float]]:
    """Outer-join per-sample absolute errors across models on a stable key.

    Returns ``{model: {(key_tuple): abs_error}}`` so two models can be compared
    only on samples present in both.
    """
    key = tuple(key)
    out: Dict[str, Dict[Tuple, float]] = {}
    for model, rows in model_rows.items():
        d: Dict[Tuple, float] = {}
        for row in rows:
            k = tuple(str(row.get(kk, "")) for kk in key)
            d[k] = float(row["abs_error_bpm"])
        out[model] = d
    return out


def paired_wilcoxon(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    key: Sequence[str] = DEFAULT_KEY,
) -> float:
    """Paired Wilcoxon signed-rank p-value on absolute errors of two row sets."""
    aligned = align_pairs({"a": rows_a, "b": rows_b}, key)
    da, db = aligned["a"], aligned["b"]
    common = [k for k in da if k in db]
    if len(common) < 2:
        return math.nan
    x = np.array([da[k] for k in common], dtype=float)
    y = np.array([db[k] for k in common], dtype=float)
    if np.all(x == y):
        return 1.0
    try:
        _, p = scipy_stats.wilcoxon(x, y)
    except ValueError:
        return math.nan
    return float(p)


def benjamin_hochberg(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg step-up FDR correction (NaN p-values are skipped)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    q = np.full(n, math.nan)
    finite = ~np.isnan(p)
    fp = p[finite]
    if len(fp) == 0:
        return q.tolist()
    order = np.argsort(fp)
    sorted_p = fp[order]
    # q_i = p_(i) * m / i  with monotonic enforcement from the largest p upward
    ranked = sorted_p * len(fp) / np.arange(1, len(fp) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(fp))
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[finite] = out
    return q.tolist()


def wilcoxon_matrix(
    model_rows: Dict[str, List[Dict[str, Any]]],
    key: Sequence[str] = DEFAULT_KEY,
) -> Tuple[List[str], List[Tuple[str, str]], List[float], List[float]]:
    """All backbone pairs: raw p-values and BH-corrected q-values.

    Returns ``(models, pairs, pvals, qvals)``. Pairs are only tested on the
    samples shared by both models (paired); unpaired/identical pairs yield NaN.
    """
    models = sorted(model_rows.keys())
    aligned = align_pairs(model_rows, key)
    pairs: List[Tuple[str, str]] = []
    pvals: List[float] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            common = [k for k in aligned[a] if k in aligned[b]]
            if len(common) < 2:
                p = math.nan
            else:
                x = np.array([aligned[a][k] for k in common], dtype=float)
                y = np.array([aligned[b][k] for k in common], dtype=float)
                if np.all(x == y):
                    p = 1.0
                else:
                    try:
                        _, p = scipy_stats.wilcoxon(x, y)
                    except ValueError:
                        p = math.nan
            pairs.append((a, b))
            pvals.append(float(p) if not math.isnan(p) else math.nan)
    qvals = benjamin_hochberg(pvals)
    return models, pairs, pvals, qvals


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval on MAE
# --------------------------------------------------------------------------- #
def bootstrap_mae_ci(
    rows: Iterable[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI on MAE from per-sample absolute errors."""
    errs = np.array([float(r["abs_error_bpm"]) for r in rows], dtype=float)
    if errs.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    n = errs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = errs[idx].mean(axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# --------------------------------------------------------------------------- #
# Ad-hoc driver
# --------------------------------------------------------------------------- #
def _run_on_dir(model_outputs_dir: str) -> None:
    all_rows = load_all_predictions(model_outputs_dir)
    if not all_rows:
        print(f"no predictions_test.csv found under {model_outputs_dir}")
        return
    datasets = sorted({r["dataset"] for r in all_rows})
    for ds in datasets:
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        models = sorted({r["model"] for r in ds_rows})
        print(f"\n==================== dataset={ds} | models={models} ====================")

        # Per-subject + bootstrap per model
        print("\n-- per-subject (mean MAE across subjects) & bootstrap MAE 95% CI --")
        for m in models:
            mrows = [r for r in ds_rows if r["model"] == m]
            ps = per_subject_summary(mrows)
            lo, hi = bootstrap_mae_ci(mrows)
            across = ps.get("across", {})
            print(f"  {m:14s} subj_MAE={across.get('mean_mae', float('nan')):6.2f}"
                  f" +/-{across.get('std_mae', 0):5.2f} (n={across.get('n_subjects','?')})"
                  f" | MAE 95% CI=[{lo:.2f}, {hi:.2f}]")

        # Wilcoxon matrix
        model_rows = {m: [r for r in ds_rows if r["model"] == m] for m in models}
        _, pairs, pvals, qvals = wilcoxon_matrix(model_rows)
        print("\n-- paired Wilcoxon (p) with Benjamini-Hochberg q (rows sig at q<0.05) --")
        for (a, b), p, q in sorted(zip(pairs, pvals, qvals), key=lambda t: (math.nan if math.isnan(t[1]) else t[1])):
            ps = "NaN" if math.isnan(p) else f"{p:.4f}"
            qs = "NaN" if math.isnan(q) else f"{q:.4f}"
            flag = " *" if (not math.isnan(q) and q < 0.05) else ""
            print(f"  {a:14s} vs {b:14s}  p={ps:>8}  q={qs:>8}{flag}")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "model_outputs"
    _run_on_dir(target)
