"""Subject-wise (leave-one-subject-out) cross-validation.

The headline benchmark splits windows RANDOMLY, so every subject contributes
windows to both train and test. That means the "per-subject" numbers in v3 are
in-sample for those subjects. To make the subject-level claim rigorous we add
true leave-one-subject-out (LOSO) CV:

  For dataset D with subjects s1..sk:
    fold i: train/val on all subjects EXCEPT si, test on si (all of si's windows)
  After k folds, the concatenation of every fold's test predictions is a
  per-window evaluation in which NO test window's subject was ever seen in
  training -- an honest subject-generalization number.

Each fold is a normal training run (canonical FTU->FTU protocol) pointed at a
per-fold manifest. The per-fold manifest is generated here by re-labeling the
`split` column of the real export's manifest; window paths are rewritten to the
real absolute location so datasets.py resolves them without copying data.

Output (under model_outputs_cv/<model>/<dataset>/):
    folds/<subject>/manifest.csv           per-fold split
    folds/<subject>/{best.pt,predictions_test.csv,...}
    cv_summary.json                        pooled + per-fold metrics
    cv_summary.md

Execution is GPU-bound (k folds x training). Until the headline 12x3 queue frees
the GPU, run a plumbing smoke test on CPU:
    uv run python experiment_runs/run_subject_cv.py --model tcn --dataset FTU \\
        --smoke --folds 1

Usage (full sweep, once GPU is free):
    uv run python experiment_runs/run_subject_cv.py --model tcn --dataset FTU
    uv run python experiment_runs/run_subject_cv.py --model transformer --dataset FTU --folds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REAL_EXPORT = Path("/home/agent-dev-radar/radar/work/run/upstream/training_exports")

# Canonical FTU->FTU protocol (mirrors experiment_runs/queue_all.sh).
CANON_EPOCHS = 80
CANON_LR = 3e-4
CANON_WD = 1e-4
CANON_SEED = 42

# Per-model batch sizes (4 GB T600) -- same table as queue_all.sh.
BS = {
    "contiformer": 32, "dlinear": 64, "nlinear": 64, "tsmixer": 64,
    "mamba": 16, "cycleformer": 16, "heart_timemixer": 16, "patchtst": 32,
    "tcn": 32, "timesnet": 16, "transformer": 16, "tslanet": 32,
    "xlstm": 16, "frets": 32,
}


def _subject_id(group_key: str) -> str:
    """Extract the participant id from 'FTU/participant/1' -> '1' (dataset-relative)."""
    return group_key.split("/")[-1]


def _remap_window_path(raw: str, real_export: Path) -> str:
    """Rewrite a build-time window path to its real absolute location."""
    marker = "training_exports/"
    text = raw.replace("\\", "/")
    if marker in text:
        rel = text.split(marker, 1)[1]
        candidate = real_export / rel
        if candidate.exists():
            return str(candidate)
    p = Path(raw)
    if p.exists():
        return str(p)
    return raw


def list_subjects(manifest_rows: List[Dict[str, str]], dataset: str) -> List[str]:
    subs = sorted({_subject_id(r["group_key"]) for r in manifest_rows if r["dataset"] == dataset})
    return subs


def write_fold_manifest(real_export: Path, out_dir: Path, rows: List[Dict[str, str]],
                        dataset: str, held_out: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.csv"
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r["dataset"] != dataset:
                continue
            row = dict(r)
            row["window_path"] = _remap_window_path(r["window_path"], real_export)
            if _subject_id(r["group_key"]) == held_out:
                row["split"] = "test"
            # other subjects keep their original train/val split
            w.writerow(row)
    return out_path


def train_fold(model: str, dataset: str, fold_export: Path, out_dir: Path,
               epochs: int, batch_size: int, smoke: bool) -> int:
    """Run train_model.py for one fold. Returns process return code."""
    cmd = [
        "uv", "run", "python", "src/training/train_model.py",
        "--model", model, "--datasets", dataset,
        "--epochs", str(epochs), "--batch-size", str(batch_size),
        "--seed", str(CANON_SEED), "--lr", str(CANON_LR), "--weight-decay", str(CANON_WD),
        "--export-dir", str(fold_export), "--output-dir", str(out_dir),
        "--dump-predictions", "--resume",
    ]
    if smoke:
        cmd += ["--limit-batches", "4"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.log"
    with log.open("w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT).returncode
    return rc


def aggregate_cv(model: str, dataset: str, out_root: Path, subjects: List[str]) -> Dict:
    from src.training.common import stats as S
    pooled_rows: List[Dict] = []
    per_fold = []
    for sub in subjects:
        pred_csv = out_root / "folds" / sub / "predictions_test.csv"
        if not pred_csv.exists():
            per_fold.append({"subject": sub, "done": False})
            continue
        rows = S.load_predictions(pred_csv, model)
        per_fold.append({"subject": sub, "done": True,
                         "mae_bpm": round(S.aggregate(rows)["overall"]["mae_bpm"], 3),
                         "n": S.aggregate(rows)["overall"]["count"]})
        pooled_rows.extend(rows)
    if pooled_rows:
        ov = S.aggregate(pooled_rows)["overall"]
        ci = S.bootstrap_mae_ci(pooled_rows, n_boot=1000, seed=42)
        subj = S.per_subject_summary(pooled_rows)
        summary = {
            "model": model,
            "dataset": dataset,
            "n_subjects": len(subjects),
            "n_done": sum(1 for f in per_fold if f["done"]),
            "pooled": {
                "n_windows": ov["count"],
                "mae_bpm": round(ov["mae_bpm"], 3),
                "rmse_bpm": round(ov["rmse_bpm"], 3),
                "pearson_r": round(ov["pearson_r"], 4),
                "within_5bpm_pct": round(ov["within_5bpm_percent"], 2),
                "within_10bpm_pct": round(ov["within_10bpm_percent"], 2),
                "mae_ci_95": [round(ci[0], 3), round(ci[1], 3)],
            },
            "subject_mean_mae": round(subj["across"]["mean_mae"], 3),
            "subject_std_mae": round(subj["across"]["std_mae"], 3),
            "per_fold": per_fold,
        }
    else:
        summary = {"model": model, "dataset": dataset, "per_fold": per_fold}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["FTU", "BGT60TR13C", "PhysDrive"])
    ap.add_argument("--real-export", type=Path, default=REAL_EXPORT)
    ap.add_argument("--out-root", type=Path, default=ROOT / "model_outputs_cv")
    ap.add_argument("--epochs", type=int, default=CANON_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=None, help="Default: per-model table.")
    ap.add_argument("--folds", type=int, default=None, help="Limit number of folds (demo).")
    ap.add_argument("--smoke", action="store_true", help="1 epoch + limit-batches=4 (plumbing test).")
    ap.add_argument("--no-train", action="store_true", help="Only (re)generate manifests + aggregate.")
    args = ap.parse_args()

    batch_size = args.batch_size or BS.get(args.model, 32)
    epochs = 1 if args.smoke else args.epochs

    manifest_rows = list(csv.DictReader((args.real_export / "manifest.csv").open(encoding="utf-8")))
    subjects = list_subjects(manifest_rows, args.dataset)
    if args.folds:
        subjects = subjects[: args.folds]
    print(f"[cv] {args.model} / {args.dataset}: {len(subjects)} subject-fold(s) "
          f"(epochs={epochs}, bs={batch_size})")

    out_root = args.out_root / args.model / args.dataset
    for sub in subjects:
        fold_export = out_root / "folds" / sub / "export"
        fold_out = out_root / "folds" / sub
        pred_csv = fold_out / "predictions_test.csv"
        done = pred_csv.exists() and (fold_out / "best.pt").exists()
        # (re)generate manifest regardless (cheap, idempotent)
        write_fold_manifest(args.real_export, fold_export, manifest_rows, args.dataset, sub)
        if done:
            print(f"  [skip] fold {sub} (already trained)")
            continue
        if args.no_train:
            print(f"  [manifest-only] fold {sub}")
            continue
        print(f"  [train] fold {sub} -> {fold_out}")
        rc = train_fold(args.model, args.dataset, fold_export, fold_out, epochs, batch_size, args.smoke)
        if rc != 0:
            print(f"  [FAILED] fold {sub} rc={rc} (see {fold_out/'train.log'})")

    summary = aggregate_cv(args.model, args.dataset, out_root, subjects)
    (out_root / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [f"# Subject-wise LOSO CV: {args.model} on {args.dataset}\n",
          f"- subjects (folds): {summary.get('n_subjects')} (done: {summary.get('n_done')})\n"]
    if "pooled" in summary:
        p = summary["pooled"]
        md.append(f"- pooled held-out MAE: **{p['mae_bpm']} BPM** "
                  f"(95% CI [{p['mae_ci_95'][0]}, {p['mae_ci_95'][1]}]), "
                  f"RMSE {p['rmse_bpm']}, Pearson r {p['pearson_r']}\n")
        md.append(f"- subject mean MAE: {summary['subject_mean_mae']} ± {summary['subject_std_mae']}\n")
        md.append(f"- within 5/10 BPM: {p['within_5bpm_pct']}% / {p['within_10bpm_pct']}%\n")
    md.append("\n## Per fold\n| subject | done | MAE | n |\n|---|---|---|---|")
    for f in summary["per_fold"]:
        md.append(f"| {f['subject']} | {f['done']} | {f.get('mae_bpm','')} | {f.get('n','')} |")
    (out_root / "cv_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[cv] wrote {out_root/'cv_summary.json'} and cv_summary.md")


if __name__ == "__main__":
    main()
