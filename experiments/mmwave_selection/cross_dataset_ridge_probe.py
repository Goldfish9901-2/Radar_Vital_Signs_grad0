"""Canonical cross-dataset Ridge probe (FTU / BGT60 / PhysDrive).

Why this exists
---------------
On mmwave-897 every axis landed on the same plateau: backbone 11.66, RDA front-end
pool 12.35, oracle range-bin selection 12.28, PC1 12.59, harmonic removal 12.26,
against a Ridge reference of 12.23 (constant 14.28, FFT peak 15.7). The open
question is whether ~12.2 is a property of *that dataset* or a general property of
this whole problem setup.

This script answers it by moving the **identical probe protocol** to the other
three datasets — same representation, same groups, same folds, same Ridge, same
metrics, same clipping — without adding any new algorithm.

Protocol (copied verbatim from `src/data/mmwave_probe.evaluate`)
----------------------------------------------------------------
    features -> subject-level GroupKFold(5)
             -> inner GroupKFold(5) alpha CV over ALPHAS
             -> standardize on the training folds only
             -> center y on the training folds (equivalent to an intercept)
             -> predict, clip to [30, 200] BPM
             -> MAE / RMSE / R^2 / Pearson

Two reported numbers per dataset
--------------------------------
    window-level   one sample per 12.8 s window (mmwave-897 has one sample per session)
    session-level  window predictions averaged within a session, then scored
                   -- the closest analogue to mmwave-897's session-level probe

Data
----
`training_exports/` (12994 windows): FTU 3520, BGT60 4448, PhysDrive 5026. Every
window carries `x_freq` (7, 129) and `x_time` (7, 256) from the
EDACM + HR-AdaVMD representation, `x_rda` (256, 8, 16, 8) log-magnitude cube,
`label_heart_rate`, and `group_key` (participant for FTU/BGT60, session for
PhysDrive). fs = 20 Hz, HR band 0.75-2.5 Hz.

Caveat that must be stated next to every number: `x_rda` is stored as
**log-magnitude**, so the raw complex phase is not recoverable from the exports and
the mmwave-897-style phase spectrum cannot be rebuilt here. The canonical
representation for these three datasets is therefore the EDACM/VMD `x_freq`
(903-d), which is exactly the feature the NN baselines were trained on.
mmwave-897's own row uses its 71-d log1p HR-band spectrum. Absolute MAEs are not
comparable across that boundary; the *gains* (vs constant, vs FFT) are.

Usage
-----
    python experiments/mmwave_selection/cross_dataset_ridge_probe.py \
        --exports training_exports --out-dir experiments/mmwave_selection/cross_dataset_output
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
for parent in [_HERE.parent, *_HERE.parents]:
    if (parent / "src" / "data" / "mmwave_probe.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

sys.path.append(str(_HERE.parent))


def _import(dotted: str):
    candidates = [dotted]
    if dotted.startswith("src."):
        candidates.append(dotted[len("src."):])
    last: BaseException | None = None
    for cand in candidates:
        try:
            return __import__(cand, fromlist=["__name__"])
        except ModuleNotFoundError as exc:
            last = exc
            continue
    raise ModuleNotFoundError(f"cannot import {dotted}: {last}") from last


_p = _import("src.data.mmwave_probe")
ALPHAS = _p.ALPHAS
_group_folds = _p.group_folds
_ridge_fit = _p.ridge_fit
_sanitize = _p._sanitize

_b = _import("src.data.mmwave_baseline")
_bp = _b.butter_bandpass

DATASETS = ("FTU", "BGT60TR13C", "PhysDrive")
FEATURE_SETS = ("phase71", "x_freq")
CLIP = (30.0, 200.0)

# The exports store `x_rda` as log-magnitude, so the complex phase is gone and
# mmwave-897's phase spectrum cannot be rebuilt from it directly. But VMD is a
# reconstruction: sum of the 7 modes in `x_time` ~= the (detrended, z-scored)
# EDACM phase. Feeding that through mmwave-897's own recipe gives a genuinely
# comparable 71-d representation, which closes the representation confound.
FS_EXPORT_HZ = 20.0
HR_BAND_EXPORT = (0.75, 2.5)
F_GRID = _p.F_GRID          # 0.5 .. 4.0 Hz, 71 points (mmwave-897's grid)


def phase71_from_xtime(x_time: np.ndarray) -> np.ndarray:
    """mmwave-897's canonical representation applied to the EDACM phase trace."""
    from scipy.signal import get_window

    trace = np.asarray(x_time, dtype=np.float64).sum(axis=0)
    if not np.isfinite(trace).all() or np.var(trace) < 1e-12:
        return np.zeros(F_GRID.size)
    filt = _bp(trace, FS_EXPORT_HZ, HR_BAND_EXPORT)
    n = filt.size
    if n < 8:
        return np.zeros(F_GRID.size)
    windowed = filt * get_window("hann", n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.abs(np.fft.rfft(windowed, nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / FS_EXPORT_HZ)
    amp = np.log1p(np.interp(F_GRID, freqs, spec))
    nrm = np.linalg.norm(amp)
    if not np.isfinite(nrm) or nrm < 1e-12:
        return np.zeros(F_GRID.size)
    return _sanitize(amp / nrm)


def _r2(pred: np.ndarray, yt: np.ndarray) -> float:
    v = np.var(yt)
    if v <= 1e-9:
        return 0.0
    return float(1 - np.mean((pred - yt) ** 2) / v)


def ridge_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  alphas: Sequence[float] = ALPHAS) -> np.ndarray:
    """Out-of-fold predictions under the mmwave_probe CV. Same code path as
    `phase0_rda_screen.ridge_eval` / `mmwave_probe.evaluate`, but returns preds."""
    n = len(y)
    preds = np.full(n, np.nan)
    for tr_m, te_m in _group_folds(groups, 5):
        trX, try_ = X[tr_m], y[tr_m]
        ytr_mean = try_.mean()
        yb = try_ - ytr_mean
        mu = trX.mean(axis=0)
        sd = trX.std(axis=0) + 1e-12
        trXs = (trX - mu) / sd
        best_alpha, best_cv = alphas[0], -np.inf
        for a in alphas:
            scores = []
            for tr2, te2 in _group_folds(groups[tr_m], 5):
                if tr2.sum() < 2 or te2.sum() < 1:
                    continue
                b = _ridge_fit(trXs[tr2], yb[tr2], a)
                p = trXs[te2] @ b + ytr_mean
                scores.append(_r2(p, try_[te2]))
            if not scores:
                continue
            m = float(np.mean(scores))
            if m > best_cv:
                best_cv, best_alpha = m, a
        b = _ridge_fit(trXs, yb, best_alpha)
        teXs = (X[te_m] - mu) / sd
        preds[te_m] = np.clip(teXs @ b + ytr_mean, CLIP[0], CLIP[1])
    return preds


def constant_predict(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Train-fold mean, i.e. the honest 'predict the mean' baseline."""
    preds = np.full(len(y), np.nan)
    for tr_m, te_m in _group_folds(groups, 5):
        preds[te_m] = y[tr_m].mean()
    return preds


def metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - y)))
    pr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-9 else 0.0
    return {"mae": mae,
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "r2": _r2(pred, y),
            "pearson": pr,
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred)),
            "label_mean": float(np.mean(y)),
            "label_std": float(np.std(y)),
            "pct_within3": float(100.0 * np.mean(np.abs(pred - y) < 3.0)),
            "pct_within5": float(100.0 * np.mean(np.abs(pred - y) < 5.0))}


def load_dataset(exports: Path, dataset: str,
                 feature_set: str) -> Dict[str, Any]:
    rows = list(csv.DictReader(open(exports / "manifest.csv")))
    rows = [r for r in rows if r["dataset"] == dataset]
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    groups: List[str] = []
    sessions: List[str] = []
    missing = 0
    for r in rows:
        rel = os.path.join("windows", dataset, r["split"],
                           os.path.basename(r["window_path"]))
        path = exports / rel
        if not path.exists():
            missing += 1
            continue
        with np.load(path) as d:
            if feature_set == "phase71":
                X_list.append(phase71_from_xtime(d["x_time"]))
            else:
                parts = []
                if "x_freq" in feature_set:
                    parts.append(d["x_freq"].reshape(-1))
                if "x_time" in feature_set:
                    parts.append(d["x_time"].reshape(-1))
                X_list.append(np.concatenate(parts).astype(np.float32))
            y_list.append(float(d["label_heart_rate"]))
        groups.append(r["group_key"])
        sessions.append(r["sample_tag"])
    if missing:
        print(f"  [{dataset}] WARNING: {missing} windows missing on disk", flush=True)
    X = _sanitize(np.stack(X_list).astype(np.float64))
    return {"X": X, "y": np.asarray(y_list, dtype=float),
            "groups": np.asarray(groups), "sessions": np.asarray(sessions)}


def session_aggregate(pred: np.ndarray, sessions: np.ndarray,
                      y: np.ndarray) -> Dict[str, float]:
    """Average window predictions inside each session, then score per session."""
    uniq = np.unique(sessions)
    p_s, y_s = [], []
    for s in uniq:
        m = sessions == s
        p_s.append(float(np.mean(pred[m])))
        y_s.append(float(np.mean(y[m])))
    return metrics(np.asarray(p_s), np.asarray(y_s))


def random_groups(n: int, seed: int = 0) -> np.ndarray:
    """Random partition labels (5 folds) -- the 'ignore subject identity' control.

    Grouped CV asks "does the model transfer to unseen subjects?" A random-split
    control asks "is there signal at all?" Comparing the two separates
    *features carry no HR information* from *features work within-subject but do
    not transfer across subjects*.
    """
    rng = np.random.default_rng(seed)
    return (rng.permutation(n) % 5).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", type=Path,
                    default=Path(__file__).resolve().parents[2] / "training_exports")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--feature-sets", type=str, default=",".join(FEATURE_SETS))
    ap.add_argument("--permute-labels", action="store_true",
                    help="also run grouped CV with shuffled labels. Sanity control: "
                         "must give r ~ 0 and gain ~ 0; if it reproduces the negative "
                         "r of the real labels, the probe is buggy.")
    ap.add_argument("--cv-modes", type=str, default="grouped",
                    help="comma-separated: grouped (subject-level, canonical) and/or "
                         "random (ignore subject identity, signal-existence control)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "cross_dataset_output")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    cv_modes = [c.strip() for c in args.cv_modes.split(",") if c.strip()]

    out: Dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exports": str(args.exports),
        "protocol": {
            "cv": "subject-level GroupKFold(5) + inner GroupKFold(5) alpha CV",
            "alphas": list(ALPHAS),
            "standardize": "fit on training folds only",
            "intercept": "y centered on training folds",
            "clip_bpm": list(CLIP),
            "identical_to": "src/data/mmwave_probe.evaluate (mmwave-897 12.23 reference)",
        },
        "results": {},
    }

    for dataset in datasets:
        out["results"][dataset] = {}
        for fs_name in feature_sets:
            print(f"[load] {dataset} / {fs_name}", flush=True)
            data = load_dataset(args.exports, dataset, fs_name)
            X, y, groups, sessions = (data["X"], data["y"], data["groups"],
                                      data["sessions"])
            print(f"  X {X.shape}  n_groups={len(np.unique(groups))} "
                  f"n_sessions={len(np.unique(sessions))}", flush=True)

            pred_c = constant_predict(y, groups)
            res = {
                "n_windows": int(len(y)),
                "n_groups": int(len(np.unique(groups))),
                "n_sessions": int(len(np.unique(sessions))),
                "n_features": int(X.shape[1]),
                "constant": metrics(pred_c, y),
            }
            if args.permute_labels:
                rngp = np.random.default_rng(7)
                y_perm = rngp.permutation(y)
                pred_p = ridge_predict(X, y_perm, groups)
                mp = metrics(pred_p, y_perm)
                res["permuted"] = mp
                res["gain_permuted"] = (
                    metrics(constant_predict(y_perm, groups), y_perm)["mae"] - mp["mae"])
                print(f"  [permuted] ridge MAE={mp['mae']:.2f} r={mp['pearson']:+.3f} "
                      f"R²={mp['r2']:.3f} gain={res['gain_permuted']:+.2f} "
                      f"(sanity: expect r~0, gain~0)", flush=True)

            for cv in cv_modes:
                g = groups if cv == "grouped" else random_groups(len(y))
                pred_r = ridge_predict(X, y, g)
                w = metrics(pred_r, y)
                s = session_aggregate(pred_r, sessions, y)
                res[f"ridge_window_{cv}"] = w
                res[f"ridge_session_{cv}"] = s
                res[f"gain_vs_constant_{cv}"] = res["constant"]["mae"] - w["mae"]
                print(f"  [{cv}] constant MAE={res['constant']['mae']:.2f} | "
                      f"ridge window MAE={w['mae']:.2f} r={w['pearson']:.3f} "
                      f"R²={w['r2']:.3f} pred_std={w['pred_std']:.2f} "
                      f"label_std={w['label_std']:.2f} | "
                      f"session MAE={s['mae']:.2f} r={s['pearson']:.3f} | "
                      f"gain={res[f'gain_vs_constant_{cv}']:+.2f}", flush=True)
            out["results"][dataset][fs_name] = res

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cross_dataset_ridge.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["# Canonical cross-dataset Ridge probe\n",
             f"- cv modes: {', '.join(cv_modes)}  (`grouped` = subject-level, "
             f"`random` = folds ignore subject identity)",
             f"- exports: `{args.exports}`",
             "- protocol: " + out["protocol"]["cv"],
             f"- alphas: {list(ALPHAS)}; standardize on train folds; y centered on "
             f"train folds; predictions clipped to [{CLIP[0]:.0f}, {CLIP[1]:.0f}] BPM",
             "- identical to `src/data/mmwave_probe.evaluate`, which produces the "
             "mmwave-897 12.23 reference\n"]
    for cv in cv_modes:
        lines += ["", f"## {cv} folds (window level)\n",
                  "| dataset | features | constant | Ridge | gain | r | R² | "
                  "pred σ | label σ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for dataset in datasets:
            for fs_name in feature_sets:
                r = out["results"][dataset][fs_name]
                w = r[f"ridge_window_{cv}"]
                lines.append(
                    f"| {dataset} | {fs_name} | {r['constant']['mae']:.2f} | "
                    f"{w['mae']:.2f} | {r[f'gain_vs_constant_{cv}']:+.2f} | "
                    f"{w['pearson']:.3f} | {w['r2']:.3f} | {w['pred_std']:.2f} | "
                    f"{w['label_std']:.2f} |")

    lines += ["",
              "## Reading it\n",
              "- `gain` = constant MAE − Ridge MAE. That is the quantity to compare "
              "across datasets, not the absolute MAE.",
              "- `pred σ` vs `label σ` is the collapse check: if pred σ is tiny "
              "relative to label σ, the model is predicting the mean regardless of "
              "how low its MAE looks.",
              "- mmwave-897's row (not computed here) comes from a *different* "
              "representation (71-d log1p HR-band phase spectrum) because the exports "
              "store `x_rda` as log-magnitude, so the complex phase is not "
              "recoverable. Reference: constant 14.28, FFT peak 15.7, Ridge 12.23 "
              "(r 0.420, R² 0.176, session-level)."]
    (args.out_dir / "cross_dataset_ridge.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nDone. Artifacts in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
