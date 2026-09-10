"""Phase-0c: does a better *spatial combination* move the Ridge ceiling?

Context
-------
* The subject-level GroupKFold(5) Ridge on the 71-d log1p HR-band spectrum is
  the project's go/no-go metric: `single_bin` = 12.23, `spec_mean` = 12.46,
  gradient constant = 14.28.
* The bin-oracle diagnostic (`phase0_bin_oracle_diag.py`) showed the FFT-peak
  estimator is hopeless per-bin (oracle 1.64 vs best unsupervised 15.3-15.7),
  and that the canonical max-energy bin is #2 (0.63 m) while the oracle bin sits
  at 4-5 m (chest -> wall multipath). PC1 of the HR-band bin ensemble does not
  rescue the FFT-peak estimator either (15.31).
* BUT: the FFT-peak estimator and the Ridge-on-spectrum read the data
  differently. A spatial combination that is useless for picking one peak may
  still denoise the *spectrum* that the Ridge reads.

So this probe asks the two questions that actually decide the next step:

1. **Headroom**: if range-bin selection were perfect (oracle bin, chosen with
   the ECG HR, which is only legitimate as an upper bound), what would the Ridge
   MAE be? If oracle-bin Ridge is still ~12, bin selection is a dead end and the
   12.23 plateau is intrinsic to the representation. If it is much lower, bin
   selection has real headroom and is worth chasing.
2. **Unsupervised spatial combination**: what does the Ridge give on spectra of
   PC1 / PC2 / mean-of-bins traces -- i.e. data-driven spatial filters instead of
   picking one bin?

Reps evaluated (all fed through the identical 71-d spectrum + Ridge CV)
----------------------------------------------------------------------
    single_bin   canonical baseline (must reproduce 12.23)
    spec_mean    mean of per-bin normalized spectra (12.46 reference)
    pc1 / pc2    spectrum of PC1 / PC2 of the HR-band bin ensemble
    mean_bp      spectrum of the mean HR-band-passed trace over bins
    oracle_bin   spectrum of the min-error bin (supervised UPPER BOUND)

Usage
-----
    python experiments/mmwave_selection/phase0_rep_probe.py --all \
        --out-dir experiments/mmwave_selection/rep_probe_output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

_HERE = Path(__file__).resolve()
for parent in [_HERE.parent, *_HERE.parents]:
    if (parent / "src" / "data" / "mmwave_baseline.py").exists():
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
EXCLUDE_FIRST = _p.EXCLUDE_FIRST
ALPHAS = _p.ALPHAS
build_features = _p.build_features
bin_spectrum = _p.bin_spectrum
_sanitize = _p._sanitize

_b = _import("src.data.mmwave_baseline")
REST_HR_BAND_HZ = _b.REST_HR_BAND_HZ
POSTEX_HR_BAND_HZ = _b.POSTEX_HR_BAND_HZ
butter_bandpass = _b.butter_bandpass
extract_phase = _b.extract_phase
hr_from_phase = _b.hr_from_phase
hr_from_ecg = _b.hr_from_ecg

_l = _import("src.data.loaders.mmwave_loader")
MMWaveDataLoader = _l.MMWaveDataLoader

# reuse the screening harness's evaluation + feature code so the CV is identical
_screen = None
try:
    import phase0_rda_screen as _screen  # type: ignore
except ImportError:  # pragma: no cover - depends on how the kernel is laid out
    _screen = None
if _screen is None:
    raise ImportError("cannot import phase0_rda_screen (needed for ridge_eval)")

ridge_eval = _screen.ridge_eval
trace_to_features = _screen.trace_to_features
single_bin_phase_trace = _screen.single_bin_phase_trace

_working = Path("/kaggle/working")
DEFAULT_OUT_DIR = (_working / "rep_probe_output") if _working.is_dir() else (
    Path(__file__).resolve().parent / "rep_probe_output")

REPS = ["single_bin", "spec_mean", "pc1", "pc2", "mean_bp", "oracle_bin"]


def pc_traces(radar: np.ndarray, fs: float,
              hr_band: Tuple[float, float]) -> List[np.ndarray]:
    """HR-band-passed per-bin traces -> PC time series, strongest first."""
    traces = []
    for b in range(EXCLUDE_FIRST, radar.shape[2]):
        traces.append(butter_bandpass(extract_phase(radar, b), fs, hr_band))
    X = np.asarray(traces)
    X = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xs = X / np.where(norm > 1e-12, norm, 1.0)
    u, s, vt = np.linalg.svd(Xs, full_matrices=False)
    return [vt[i] * s[i] for i in range(min(2, len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--participants", type=int, nargs="+", default=None)
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument("--reps", type=str, default=",".join(REPS))
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

    loader = MMWaveDataLoader(str(args.dataset))
    if not (args.all or args.participants or args.max_participants):
        args.all = True
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]
    if args.max_participants:
        participants = participants[: args.max_participants]

    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    feats: Dict[str, List[np.ndarray]] = {r: [] for r in reps}
    y: List[float] = []
    groups: List[int] = []
    oracle_bins: List[int] = []

    n_sessions = 0
    for pid in participants:
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                if not (loader._participant_dir(pid) / posture / condition).exists():
                    continue
                radar = loader.load_radar_data(pid, posture, condition)
                ref = loader.load_reference_data(pid, posture, condition)
                fs = loader.FRAME_RATE_HZ
                hr_band = (POSTEX_HR_BAND_HZ if condition == "Post-exercise"
                           else REST_HR_BAND_HZ)
                hr_ecg, _, _ = hr_from_ecg(ref["ecg_mv"], ref["fs_ecg"])
                if not np.isfinite(hr_ecg):
                    continue

                pcs = pc_traces(radar, fs, hr_band) if any(
                    r in ("pc1", "pc2") for r in reps) else []
                mean_bp = (np.mean([butter_bandpass(extract_phase(radar, b), fs, hr_band)
                                    for b in range(EXCLUDE_FIRST, radar.shape[2])], axis=0)
                           if "mean_bp" in reps else None)

                for rep in reps:
                    if rep == "single_bin":
                        vec = build_features(radar, fs, hr_band, "single_bin")
                    elif rep == "spec_mean":
                        vec = build_features(radar, fs, hr_band, "spec_mean")
                    elif rep == "pc1":
                        vec = trace_to_features(pcs[0], fs, hr_band)
                    elif rep == "pc2":
                        vec = trace_to_features(pcs[1], fs, hr_band)
                    elif rep == "mean_bp":
                        vec = trace_to_features(mean_bp, fs, hr_band)
                    elif rep == "oracle_bin":
                        # supervised upper bound: the bin whose FFT peak is closest
                        # to the ECG HR. Diagnostic only -- never a real method.
                        best_b, best_err = None, np.inf
                        for b in range(EXCLUDE_FIRST, radar.shape[2]):
                            hr_b, _, _ = hr_from_phase(extract_phase(radar, b),
                                                       fs, hr_band)
                            err = abs(hr_b - hr_ecg)
                            if err < best_err:
                                best_err, best_b = err, b
                        vec = bin_spectrum(radar, int(best_b), fs, hr_band)
                        oracle_bins.append(int(best_b))
                    else:
                        raise ValueError(rep)
                    feats[rep].append(_sanitize(vec))

                y.append(hr_ecg)
                groups.append(pid)
                n_sessions += 1
                if n_sessions % 50 == 0:
                    print(f"[extract] {n_sessions} sessions", flush=True)

    print(f"[extract] {n_sessions} sessions total", flush=True)
    yv = np.asarray(y, dtype=float)
    gv = np.asarray(groups)

    results: Dict[str, Any] = {}
    for rep in reps:
        X = np.stack(feats[rep])
        m = ridge_eval(X, yv, gv, ALPHAS)
        results[rep] = m
        print(f"[ridge] {rep:12s} MAE={m['mae_bpm']:.2f} R²={m['r2']:.3f} "
              f"r={m['pearson_r']:.3f} alpha={m['best_alpha']}", flush=True)

    base = results.get("single_bin", {}).get("mae_bpm", float("nan"))
    for rep in reps:
        if rep in results:
            results[rep]["delta_vs_single_bin"] = results[rep]["mae_bpm"] - base

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_sessions": n_sessions,
        "n_participants": int(len(np.unique(gv))),
        "fs": loader.FRAME_RATE_HZ,
        "reference": {"single_bin": 12.228782637566697,
                      "spec_mean": 12.46292561924824,
                      "constant": 14.283008288284295},
        "results": results,
        "oracle_bin_hist": ({
            "median": float(np.median(oracle_bins)),
            "min": int(min(oracle_bins)), "max": int(max(oracle_bins)),
            "hist": {str(k): int(v) for k, v in
                     zip(*np.unique(oracle_bins, return_counts=True))},
        } if oracle_bins else None),
    }
    (args.out_dir / "rep_probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    lines = ["# Spatial-combination Ridge probe (mmwave-897)\n",
             f"- sessions: {n_sessions}   participants: {payload['n_participants']}"
             f"   fs: {loader.FRAME_RATE_HZ} Hz",
             "- identical subject-level GroupKFold(5) Ridge + inner alpha CV as "
             "`mmwave_probe`; the 71-d log1p HR-band spectrum is the feature.\n",
             "| representation | MAE | Δ vs single_bin | R² | r |",
             "|---|---:|---:|---:|---:|"]
    for rep in reps:
        m = results[rep]
        lines.append(f"| {rep} | {m['mae_bpm']:.2f} | "
                     f"{m.get('delta_vs_single_bin', float('nan')):+.2f} | "
                     f"{m['r2']:.3f} | {m['pearson_r']:.3f} |")
    lines.append("")
    lines.append("`oracle_bin` is a **supervised upper bound** (the bin whose FFT peak "
                 "is closest to the ECG HR). It is not a method -- it bounds how much "
                 "perfect range-bin selection could possibly buy.")
    (args.out_dir / "rep_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Artifacts in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
