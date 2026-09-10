"""Does the Ridge probe's transfer depend on *how many subjects* are available?

Why this exists
---------------
The cross-dataset probe (`cross_dataset_ridge_probe.py`) found that on
FTU / BGT60 / PhysDrive the canonical subject-level Ridge barely beats predicting
the mean, and its out-of-fold Pearson r is *negative* (−0.59 / −0.77 / −0.26),
while on mmwave-897 the same protocol gives r = +0.42 and a +2.05 BPM gain over
the constant baseline.

Before concluding "the datasets differ", the obvious confound has to be ruled
out: **the number of subjects**. mmwave-897 has 110 participants; FTU has 10,
BGT60 has 8, PhysDrive has 48 group keys. With 5-fold group CV, 8 subjects means
test folds of 1–2 unseen people, so the probe is forced to extrapolate.

This script subsamples mmwave-897 down to those same group counts and re-runs the
identical probe, so the only thing that changes is N.

Design
------
    features: 71-d log1p HR-band spectrum, `single_bin` (the 12.23 reference)
    protocol: subject-level GroupKFold(5) + inner GroupKFold(5) alpha CV,
              standardize on train folds, y centered on train folds,
              predictions clipped to [30, 200] BPM
    for N in {8, 10, 20, 48, 110} x R=5 random participant subsets:
        grouped folds  -> MAE / r / gain vs constant
        random folds   -> same (signal-existence control)

If MAE gain collapses and r goes negative at N = 8..48 even on mmwave-897, the
cross-dataset comparison is dominated by subject count, not by dataset physics.

Usage
-----
    python experiments/mmwave_selection/phase0_groupcount_ablation.py --all \
        --out-dir experiments/mmwave_selection/groupcount_output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
_HERE = Path(__file__).resolve()
for parent in [_HERE.parent, *_HERE.parents]:
    if (parent / "src" / "data" / "mmwave_probe.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break


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
_build_features = _p.build_features
_group_folds = _p.group_folds
_sanitize = _p._sanitize

import phase0_rda_screen as _screen  # noqa: E402

ridge_eval = _screen.ridge_eval

_working = Path("/kaggle/working")
DEFAULT_OUT_DIR = (_working / "groupcount_output") if _working.is_dir() else (
    Path(__file__).resolve().parent / "groupcount_output")

N_LIST = (8, 10, 20, 48, 110)
N_REPEAT = 5
CLIP = (30.0, 200.0)


def _r2(pred: np.ndarray, yt: np.ndarray) -> float:
    v = np.var(yt)
    return 0.0 if v <= 1e-9 else float(1 - np.mean((pred - yt) ** 2) / v)


def _metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-9 else 0.0
    return {"mae": float(np.mean(np.abs(pred - y))),
            "r2": _r2(pred, y), "pearson": pr,
            "pred_std": float(np.std(pred)), "label_std": float(np.std(y))}


def _constant_pred(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    preds = np.full(len(y), np.nan)
    for tr_m, te_m in _group_folds(groups, 5):
        preds[te_m] = y[tr_m].mean()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--participants", type=int, nargs="+", default=None)
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument("--n-list", type=str, default=",".join(map(str, N_LIST)))
    ap.add_argument("--repeat", type=int, default=N_REPEAT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if args.dataset is None:
        for c in [Path("/kaggle/input/mmwave-897-2026"),
                  Path("/kaggle/input/goldfish9901/mmwave-897-2026")]:
            if (c / "P001").exists():
                args.dataset = c
                break
    if args.dataset is None:
        base = Path("/kaggle/input")
        if base.is_dir():
            for p in sorted(base.glob("**/P001")):
                args.dataset = p.parent
                break
    if args.dataset is None:
        ap.error("could not locate mmwave-897-2026; pass --dataset")
    print(f"[dataset] {args.dataset}", flush=True)

    loader = _screen.MMWaveDataLoader(str(args.dataset))
    if not (args.all or args.participants or args.max_participants):
        args.all = True
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]
    if args.max_participants:
        participants = participants[: args.max_participants]

    # extract once for every session, then subsample participants per trial
    feats: List[np.ndarray] = []
    ys: List[float] = []
    pids: List[int] = []
    for pid, raw, hr_band, hr_ecg in _screen.iter_sessions(loader, participants):
        fs = loader.FRAME_RATE_HZ
        feats.append(_build_features(raw, fs, hr_band, "single_bin"))
        ys.append(hr_ecg)
        pids.append(pid)
    X_all = _sanitize(np.stack(feats))
    y_all = np.asarray(ys, dtype=float)
    pid_all = np.asarray(pids)
    print(f"[extract] {X_all.shape[0]} sessions, "
          f"{len(np.unique(pid_all))} participants", flush=True)

    n_list = [int(s) for s in args.n_list.split(",") if s.strip()]
    out: Dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_sessions_total": int(X_all.shape[0]),
        "n_participants_total": int(len(np.unique(pid_all))),
        "repeat": args.repeat,
        "protocol": ("subject-level GroupKFold(5) + inner GroupKFold(5) alpha CV, "
                     "single_bin 71-d log1p HR-band spectrum"),
        "results": {},
    }

    for n_groups in n_list:
        if n_groups > len(np.unique(pid_all)):
            continue
        trials: List[Dict[str, Any]] = []
        for rep in range(args.repeat):
            rng = np.random.default_rng(1000 + rep)
            chosen = rng.choice(np.unique(pid_all), size=n_groups, replace=False)
            mask = np.isin(pid_all, chosen)
            X, y, g = X_all[mask], y_all[mask], pid_all[mask]
            m = ridge_eval(X, y, g, ALPHAS)
            c = _constant_pred(y, g)
            cm = _metrics(c, y)
            # random-fold control: same data, folds ignore subject identity
            g_rand = (rng.permutation(len(y)) % 5).astype(int)
            m_rand = ridge_eval(X, y, g_rand, ALPHAS)
            c_rand = _constant_pred(y, g_rand)
            trials.append({
                "rep": rep,
                "n_sessions": int(len(y)),
                "grouped": {"mae": m["mae_bpm"], "r2": m["r2"],
                            "pearson": m["pearson_r"],
                            "constant_mae": cm["mae"],
                            "gain": cm["mae"] - m["mae_bpm"]},
                "random": {"mae": m_rand["mae_bpm"], "r2": m_rand["r2"],
                           "pearson": m_rand["pearson_r"],
                           "constant_mae": _metrics(c_rand, y)["mae"],
                           "gain": _metrics(c_rand, y)["mae"] - m_rand["mae_bpm"]},
            })
            print(f"[n={n_groups} rep={rep}] grouped MAE={m['mae_bpm']:.2f} "
                  f"r={m['pearson_r']:+.3f} gain={trials[-1]['grouped']['gain']:+.2f} | "
                  f"random MAE={m_rand['mae_bpm']:.2f} r={m_rand['pearson_r']:+.3f} "
                  f"gain={trials[-1]['random']['gain']:+.2f}", flush=True)

        agg: Dict[str, Any] = {}
        for mode in ("grouped", "random"):
            maes = [t[mode]["mae"] for t in trials]
            rs = [t[mode]["pearson"] for t in trials]
            gains = [t[mode]["gain"] for t in trials]
            agg[mode] = {
                "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "pearson_mean": float(np.mean(rs)), "pearson_std": float(np.std(rs)),
                "gain_mean": float(np.mean(gains)), "gain_std": float(np.std(gains)),
            }
        out["results"][str(n_groups)] = {"trials": trials, "aggregate": agg}
        print(f"[n={n_groups}] MEAN grouped MAE={agg['grouped']['mae_mean']:.2f}"
              f"±{agg['grouped']['mae_std']:.2f} r={agg['grouped']['pearson_mean']:+.3f}"
              f" gain={agg['grouped']['gain_mean']:+.2f} | "
              f"random r={agg['random']['pearson_mean']:+.3f} "
              f"gain={agg['random']['gain_mean']:+.2f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "groupcount_ablation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["# mmwave-897: does the Ridge probe depend on the number of subjects?\n",
             "- features: 71-d log1p HR-band spectrum (`single_bin`)",
             "- protocol: subject-level GroupKFold(5) + inner alpha CV; "
             "5 random participant subsets per N\n",
             "| N subjects | sessions | grouped MAE | grouped r | gain vs constant | "
             "random r | random gain |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for n_groups in n_list:
        key = str(n_groups)
        if key not in out["results"]:
            continue
        a = out["results"][key]["aggregate"]
        nn = out["results"][key]["trials"][0]["n_sessions"]
        lines.append(f"| {n_groups} | {nn} | {a['grouped']['mae_mean']:.2f}"
                     f"±{a['grouped']['mae_std']:.2f} | "
                     f"{a['grouped']['pearson_mean']:+.3f} | "
                     f"{a['grouped']['gain_mean']:+.2f} | "
                     f"{a['random']['pearson_mean']:+.3f} | "
                     f"{a['random']['gain_mean']:+.2f} |")
    lines += ["",
              "Compare against the cross-dataset probe: FTU (10 groups) r=−0.59, "
              "BGT60 (8 groups) r=−0.77, PhysDrive (48 groups) r=−0.26."]
    (args.out_dir / "groupcount_ablation.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nDone. Artifacts in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
