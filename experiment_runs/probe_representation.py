"""Information / linear probe: is there usable HR signal in the representation?

Motivation (post head-line analysis): we showed PhysDrive is *representation-,
not backbone-limited* — deep models barely beat a constant predictor and their
per-window correlation with HR is near zero. The open question is *where the
ceiling is*: does a simple linear model on the frozen representation already
capture essentially everything the deep backbone does?

    frozen representation (x_time, x_freq)  ->  Ridge  ->  HR

Interpretation
--------------
If  probe MAE  ~=  best deep-model MAE   ->  the representation carries little
signal and the bottleneck is the INPUT (representation), not the model. Adding
backbones cannot help.
If  probe MAE  >> best deep-model MAE   ->  the backbone is extracting real
structure from the representation; deeper/recurrent models may still help.

This is the "model capacity upper bound" experiment and runs on CPU in seconds.
It needs no new signal-processing method and does not launch any GPU training.

Usage
-----
    python experiment_runs/probe_representation.py --representation proposed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.training.common import stats as S  # noqa: E402

DATASETS = ["FTU", "BGT60TR13C", "PhysDrive"]
EXPORT_DIR = Path("/home/agent-dev-radar/radar/work/run/upstream/training_exports")
MODEL_OUTPUTS = ROOT / "model_outputs"
LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


# --------------------------------------------------------------------------- #
def _load_split(export_dir: Path, dataset: str, split: str):
    feats, hrs = [], []
    d = export_dir / "windows" / dataset / split
    if not d.is_dir():
        return None, None
    for f in sorted(d.glob("*.npz")):
        data = np.load(f, allow_pickle=False)
        if "x_time" not in data:
            continue
        xt = data["x_time"].astype(np.float32)
        xf = data["x_freq"].astype(np.float32) if "x_freq" in data else None
        v = xt.reshape(1, -1)  # flatten the whole window into one row
        if xf is not None:
            v = np.concatenate([v, xf.reshape(1, -1)], axis=1)
        feats.append(v)
        hrs.append(float(np.asarray(data["label_heart_rate"])))
    if not feats:
        return None, None
    return np.concatenate(feats, axis=0), np.asarray(hrs, dtype=np.float32)


def _ridge(X, y, lam: float) -> np.ndarray:
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def _standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def _metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    if y.std() > 0 and pred.std() > 0:
        r = float(np.corrcoef(pred, y)[0, 1])
    else:
        r = 0.0
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "pearson_r": r, "r2": r2}


def _best_deep_mae(dataset: str, model_outputs: Path) -> float:
    best = None
    if not model_outputs.is_dir():
        return float("nan")
    for mdir in model_outputs.iterdir():
        p = mdir / dataset / "predictions_test.csv"
        if p.exists():
            rows = S.load_predictions(p, mdir.name)
            m = S.aggregate(rows)["overall"]["mae_bpm"]
            best = m if best is None else min(best, m)
    return best


# --------------------------------------------------------------------------- #
def probe_dataset(dataset: str, export_dir: Path, n_folds: int = 3, model_outputs: Path = MODEL_OUTPUTS):
    Xtr, ytr = _load_split(export_dir, dataset, "train")
    Xte, yte = _load_split(export_dir, dataset, "test")
    if Xtr is None:
        return None
    if Xte is None:  # fall back to val, else skip
        Xte, yte = _load_split(export_dir, dataset, "val")
    if Xte is None:
        return None

    Xtr_s, mu, sd = _standardize(Xtr)
    Xte_s, _, _ = _standardize(Xte, mu, sd)

    # Implicit intercept via target centering on the TRAIN mean: when the probe
    # has no signal (w->0) it predicts the train mean, i.e. MAE ~= MAD — the
    # correct "no information" baseline (NOT a collapse to 0).
    y_mean = float(ytr.mean())
    ytr_c = ytr - y_mean
    yte_c = yte - y_mean

    # lambda selection by k-fold CV MAE on train (centered targets)
    n = Xtr_s.shape[0]
    idx = np.array_split(np.arange(n), n_folds)
    best_lam, best_mae = None, 1e9
    for lam in LAMBDA_GRID:
        folds_mae = []
        for k in range(n_folds):
            te = idx[k]
            tr = np.concatenate([idx[j] for j in range(n_folds) if j != k])
            w = _ridge(Xtr_s[tr], ytr_c[tr], lam)
            pred = Xtr_s[te] @ w
            folds_mae.append(np.mean(np.abs(pred - ytr_c[te])))
        m = float(np.mean(folds_mae))
        if m < best_mae:
            best_mae, best_lam = m, lam
    w = _ridge(Xtr_s, ytr_c, best_lam)
    # The probe was trained on centered targets, so the raw prediction is also
    # centered. Shift it back by the TRAIN mean to recover predictions on the
    # original HR scale before scoring.
    pred = (Xte_s @ w) + y_mean
    met = _metrics(pred, yte)
    met["lambda"] = float(best_lam)
    met["n_train"] = int(n)
    met["n_test"] = int(Xte_s.shape[0])
    met["deep_mae"] = _best_deep_mae(dataset, model_outputs)
    return met


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--representation", default="proposed", help="label for reporting only")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--model-outputs", type=Path, default=MODEL_OUTPUTS)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None,
                    help="optional path to dump per-dataset probe metrics as JSON")
    args = ap.parse_args()

    cache: Dict[str, object] = {
        "representation": args.representation,
        "datasets": {},
    }
    print(f"# Information probe — representation: {args.representation}\n")
    print(f"{'dataset':<12} {'probe MAE':>9} {'deep MAE':>9} {'Δ(deep-probe)':>13} {'pearson_r':>10} {'R²':>7}  verdict")
    print("-" * 92)
    for ds in args.datasets:
        m = probe_dataset(ds, args.export_dir, args.folds, args.model_outputs)
        if m is None:
            print(f"{ds:<12} no windows")
            continue
        deep = m["deep_mae"]
        delta = (deep - m["mae"]) if deep == deep else float("nan")
        if deep == deep and delta < 1.0:
            verdict = "representation-limited (probe ~ deep)"
        elif deep == deep:
            verdict = "backbone helps (deep >> probe)"
        else:
            verdict = "deep ref unavailable"
        print(f"{ds:<12} {m['mae']:>9.2f} {deep:>9.2f} {delta:>+13.2f} {m['pearson_r']:>10.3f} "
              f"{m['r2']:>7.3f}  {verdict}")
        print(f"            (n_train={m['n_train']}, n_test={m['n_test']}, lambda={m['lambda']:.3g})")
        cache["datasets"][ds] = {
            "probe_mae": m["mae"], "deep_mae": deep, "delta_deep_probe": delta,
            "pearson_r": m["pearson_r"], "r2": m["r2"], "lambda": m["lambda"],
            "n_train": m["n_train"], "n_test": m["n_test"], "verdict": verdict,
        }

    if args.json is not None:
        import json as _json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_json.dumps(cache, indent=2), encoding="utf-8")
        print(f"\n[written] {args.json}")

    print("\nInterpretation: probe ≈ deep  => the representation is the ceiling (backbones cannot "
          "recover missing signal). deep >> probe => the backbone extracts real structure, so "
          "better architectures may still help. Either way, a near-constant probe (R²≈0) means the "
          "input carries almost no HR signal under the current representation.")


if __name__ == "__main__":
    main()
