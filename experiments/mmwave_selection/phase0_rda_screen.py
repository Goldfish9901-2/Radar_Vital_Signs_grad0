"""Phase-0 RDA candidate screening harness (CPU-only, no NN training).

Purpose
-------
Replace the ad-hoc "change a few lines in build_training_dataset.py and re-run an
NN" loop with a *controlled single-stage-at-a-time* screen of the RDA front-end
algorithm pool (``src.radar``). Every candidate runs **one stage swapped at a
time** against a fixed baseline, so a win/loss is attributable to exactly one
physical layer (clutter / localization / range-selection / beamforming / phase).

Pipeline per session
--------------------
    raw cube (F, A, R)                       [mmwave-897, D=1 compatibility adapter]
        -> apply_rda_pipeline(cfg)           [one stage candidate swapped]
        -> squeeze D=1  -> (F, A, R)
        -> phase trace (fixed canonical extraction, or pool phase stage)
        -> 71-d log1p HR-band spectrum       [build_features('single_bin') / trace_to_features]
        -> subject-level GroupKFold(5) Ridge [reuses mmwave_probe's exact CV]
        -> MAE / R² / Pearson r + trace diagnostics

Outputs (machine-readable)
--------------------------
    <out-dir>/phase0_rda_screen.csv    leaderboard
    <out-dir>/phase0_rda_screen.json   full structured results + metadata
    <out-dir>/phase0_rda_screen.md     human-readable summary

Promotion policy (three-tier)
-----------------------------
    DROP    : MAE worsens by >= worsen_margin  OR  diagnostics degraded
    HOLD    : ~baseline, no clear evidence
    PROMOTE : MAE improves by >= improve_margin AND diagnostics not degraded

Only PROMOTE candidates earn a cross-dataset check (PhysDrive / FTU / BGT60) in a
later phase. No candidate is dropped/promoted on a single noisy run; margins are
configurable.

mmwave-897 notes
----------------
    * It ships as (F, 8, 64) with NO Doppler axis (chirps averaged at storage).
      The `D=1` inserted by ``ensure_rda_4d`` is a COMPATIBILITY ADAPTER, not a
      real Doppler bin. Doppler-processing conclusions are therefore LIMITED on
      this dataset (esp. beamforming). PhysDrive (F,8,16,8) is the fuller target.
    * This script is the cheap round-1 filter: 440 sessions, Ridge reference
      12.23 BPM. It must NOT be used to conclude "Doppler beamforming works".

Usage
-----
    python experiments/mmwave_selection/phase0_rda_screen.py \
        --dataset /path/to/mmwave-897-2026 --all \
        --out-dir experiments/mmwave_selection/phase0_output

    # fast smoke on 20 participants:
    python experiments/mmwave_selection/phase0_rda_screen.py \
        --dataset /path/to/mmwave-897-2026 --max-participants 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.signal import get_window

def _repo_root() -> Path:
    """Find the directory whose subtree contains the ``radar`` package.

    Handles both the local layout and Kaggle's script-kernel layout, where the
    committed file becomes ``/kaggle/src/script.py`` and bundled directories are
    preserved as subdirectories (e.g. our ``src/`` -> ``/kaggle/src/src``).
    Kaggle may also flatten, so we add every candidate root to sys.path and the
    ``_import`` helper below tries both ``src.<pkg>`` and a flattened ``<pkg>``.
    """
    here = Path(__file__).resolve()
    roots: List[Path] = []
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "radar" / "__init__.py").exists():
            roots.append(parent)            # import as src.radar
        if (parent / "radar" / "__init__.py").exists() and not (parent / "src").exists():
            roots.append(parent)            # import as radar (flattened)
    # On Kaggle the code may be shipped as a mounted dataset under /kaggle/input.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in sorted(kaggle_input.iterdir()):
            if (child / "src" / "radar" / "__init__.py").exists():
                roots.append(child)
    for r in roots:
        if str(r) not in sys.path:
            sys.path.insert(0, str(r))
    return roots[0] if roots else here.parents[1]

# Kaggle collects kernel output from /kaggle/working, not from the source dir.
_working = Path("/kaggle/working")
DEFAULT_OUT_DIR = (_working / "phase0_output") if _working.is_dir() else (
    Path(__file__).resolve().parent / "phase0_output")


_repo_root()


def _import(dotted: str):
    """Import ``dotted``; if it starts with ``src.`` also try the flattened form.

    Robust to Kaggle flattening the bundled ``src/`` into the script directory.
    """
    candidates = [dotted]
    if dotted.startswith("src."):
        candidates.append(dotted[len("src."):])
    last: BaseException | None = None
    for cand in candidates:
        try:
            return __import__(cand, fromlist=["__name__"])
        except ModuleNotFoundError as exc:
            last = exc          # keep the real cause (often a nested missing import)
            continue
    raise ModuleNotFoundError(
        f"cannot import {dotted} (tried {candidates}): {last}"
    ) from last


_r = _import("src.radar")
RDAConfig = _r.RDAConfig
apply_rda_pipeline = _r.apply_rda_pipeline
list_stage_methods = _r.list_stage_methods

_b = _import("src.data.mmwave_baseline")
REST_HR_BAND_HZ = _b.REST_HR_BAND_HZ
POSTEX_HR_BAND_HZ = _b.POSTEX_HR_BAND_HZ
butter_bandpass = _b.butter_bandpass
hr_from_ecg = _b.hr_from_ecg

_p = _import("src.data.mmwave_probe")
EXCLUDE_FIRST = _p.EXCLUDE_FIRST
F_GRID = _p.F_GRID
ALPHAS = _p.ALPHAS
_build_features = _p.build_features
_group_folds = _p.group_folds
_ridge_fit = _p.ridge_fit
_sanitize = _p._sanitize

_l = _import("src.data.loaders.mmwave_loader")
MMWaveDataLoader = _l.MMWaveDataLoader

N_FREQS = len(F_GRID)  # 71

# Known authoritative linear-probe reference on mmwave-897 full 440 (single_bin).
KNOWN_REFERENCE_MAE = 12.23
KNOWN_REFERENCE_R2 = 0.176

# Which cube-stage candidates to screen (registry name -> human stage tag).
# "none"/baseline methods are skipped (they ARE the baseline).
STAGE_CANDIDATES: Dict[str, str] = {
    "clutter": "C",
    "localization": "L",
    "range_selection": "R",
    "beamforming": "B",
    "phase": "P",
}
# baseline method per cube stage (the "current implicit" behaviour)
BASELINE_METHOD = {
    "clutter": "none",
    "localization": "energy",
    "range_selection": "global_energy",
    "beamforming": "fft",
}


# ---------------------------------------------------------------------------
# Feature / trace helpers (mirror mmwave_probe.bin_spectrum for 1-D traces)
# ---------------------------------------------------------------------------

def single_bin_phase_trace(radar: np.ndarray, fs: float,
                           hr_band: Tuple[float, float]) -> np.ndarray:
    """Canonical single-bin phase trace (unwrapped, band-passed) of `radar` (F,A,R).

    Mirrors the selection used by ``build_features(..., 'single_bin')`` so the
    phase-stage baseline is directly comparable to the cube-stage baseline.
    """
    energy = np.abs(radar[:, :, EXCLUDE_FIRST:]).mean(axis=(0, 1)) ** 2
    b = EXCLUDE_FIRST + int(np.argmax(energy))
    cell = radar[:, :, b].mean(axis=1)
    ph = np.unwrap(np.angle(cell))
    ph = ph - ph.mean()
    return butter_bandpass(ph, fs, hr_band)


def trace_to_features(trace: np.ndarray, fs: float,
                     hr_band: Tuple[float, float]) -> np.ndarray:
    """71-d log1p-normalized HR-band magnitude spectrum of a 1-D phase trace."""
    t = np.asarray(trace, dtype=np.float64)
    if t.size < 8:
        return np.zeros(N_FREQS, dtype=np.float64)
    n = t.size
    windowed = t * get_window("hann", n)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))
    spec = np.abs(np.fft.rfft(windowed, nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.interp(F_GRID, freqs, spec)
    amp = np.log1p(amp)
    nrm = np.linalg.norm(amp)
    if not np.isfinite(nrm) or nrm < 1e-12:
        return np.zeros(N_FREQS, dtype=np.float64)
    return _sanitize(amp / nrm)


def compute_diagnostics(trace: np.ndarray, fs: float,
                        hr_band: Tuple[float, float]) -> Dict[str, float]:
    """Raw-trace diagnostics: tell *why* a candidate helped or hurt.

    Computed on the (unsanitized) trace for NaN/Inf, and on the sanitized trace
    for the rest.
    """
    raw = np.asarray(trace, dtype=np.float64)
    nan_inf = int(np.sum(~np.isfinite(raw)))
    t = _sanitize(raw)
    if t.size == 0:
        return {k: 0.0 for k in (
            "nan_inf", "var", "abs_mean", "std", "discontinuity",
            "hr_band_power", "spectral_flatness", "temporal_cont")}
    var = float(np.var(t))
    abs_mean = float(np.mean(np.abs(t)))
    std = float(np.std(t))
    discontinuity = float(np.mean(np.abs(np.diff(t)))) if t.size > 1 else 0.0
    hr_band_power = float(np.mean(t ** 2))  # trace already band-limited

    spec = np.abs(np.fft.rfft(t))
    freqs = np.fft.rfftfreq(t.size, d=1.0 / fs)
    mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    band = spec[mask]
    if band.size > 0 and band.sum() > 0:
        g = np.exp(np.mean(np.log(band + 1e-12)))
        a = np.mean(band + 1e-12)
        spectral_flatness = float(g / a)
    else:
        spectral_flatness = float("nan")

    if t.size > 1 and np.std(t[:-1]) > 1e-9 and np.std(t[1:]) > 1e-9:
        temporal_cont = float(np.corrcoef(t[:-1], t[1:])[0, 1])
    else:
        temporal_cont = 0.0

    return {
        "nan_inf": float(nan_inf),
        "var": var,
        "abs_mean": abs_mean,
        "std": std,
        "discontinuity": discontinuity,
        "hr_band_power": hr_band_power,
        "spectral_flatness": spectral_flatness,
        "temporal_cont": temporal_cont,
    }


# ---------------------------------------------------------------------------
# Ridge evaluation (identical CV to mmwave_probe -> apples-to-apples)
# ---------------------------------------------------------------------------

def _r2(pred: np.ndarray, yt: np.ndarray) -> float:
    v = np.var(yt)
    if v <= 1e-9:
        return 0.0
    return float(1 - np.mean((pred - yt) ** 2) / v)


def ridge_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               alphas: List[float] = ALPHAS) -> Dict[str, float]:
    """Subject-level GroupKFold(5) Ridge with inner alpha CV. Returns metrics.

    Mirrors ``mmwave_probe.evaluate`` exactly so the baseline row reproduces the
    12.23 / R^2=0.176 reference.
    """
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
        preds[te_m] = np.clip(teXs @ b + ytr_mean, 30.0, 200.0)
    mae = float(np.mean(np.abs(preds - y)))
    r2 = _r2(preds, y)
    pr = float(np.corrcoef(preds, y)[0, 1]) if np.std(preds) > 1e-9 else 0.0
    return {"mae_bpm": mae, "r2": r2, "pearson_r": pr, "best_alpha": best_alpha}


# ---------------------------------------------------------------------------
# Per-session feature extraction
# ---------------------------------------------------------------------------

def features_cube_stage(raw: np.ndarray, config: RDAConfig, fs: float,
                        hr_band: Tuple[float, float], rep: str = "single_bin"
                        ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply cube stages (one candidate swapped), then canonical representation."""
    out = apply_rda_pipeline(raw, config)        # (F,1,A,R)
    cube = out[:, 0, :, :]                        # (F,A,R)
    feat = _build_features(cube, fs, hr_band, rep)
    trace = single_bin_phase_trace(cube, fs, hr_band)
    return feat, compute_diagnostics(trace, fs, hr_band)


def features_phase_stage(raw: np.ndarray, config: RDAConfig, fs: float,
                         hr_band: Tuple[float, float], bandpass: bool = False
                         ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply baseline cube stages, then extract a phase trace via the pool.

    ``bandpass=True`` applies the same ``butter_bandpass`` the canonical
    ``single_bin_phase_trace`` uses. Without it the comparison is confounded:
    the baseline is HR-band limited while raw pool traces are not, so the delta
    measures "band-pass or not" instead of "which phase extraction".
    """
    trace = apply_rda_pipeline(raw, config, return_phase=True)  # 1-D
    diag = compute_diagnostics(trace, fs, hr_band)
    if bandpass:
        trace = butter_bandpass(np.asarray(trace, dtype=np.float64), fs, hr_band)
    feat = trace_to_features(trace, fs, hr_band)
    return feat, diag


# ---------------------------------------------------------------------------
# Screening driver
# ---------------------------------------------------------------------------

def iter_sessions(loader: MMWaveDataLoader, participants: List[int]):
    for pid in participants:
        for posture in loader.POSTURES:
            for condition in loader.CONDITIONS:
                if not (loader._participant_dir(pid) / posture / condition).exists():
                    continue
                raw = loader.load_radar_data(pid, posture, condition)
                ref = loader.load_reference_data(pid, posture, condition)
                hr_band = (POSTEX_HR_BAND_HZ if condition == "Post-exercise"
                           else REST_HR_BAND_HZ)
                hr_ecg, _, _ = hr_from_ecg(ref["ecg_mv"], ref["fs_ecg"])
                if not np.isfinite(hr_ecg):
                    continue
                yield pid, raw, hr_band, hr_ecg


def screen_one(loader: MMWaveDataLoader, participants: List[int],
               extractor: Callable, label: str) -> Dict[str, Any]:
    """Run one candidate/config across all sessions; return aggregated result."""
    feats: List[np.ndarray] = []
    diags_acc: List[Dict[str, float]] = []
    y: List[float] = []
    groups: List[int] = []
    for pid, raw, hr_band, hr_ecg in iter_sessions(loader, participants):
        fs = loader.FRAME_RATE_HZ
        feat, diag = extractor(raw, fs, hr_band)
        feats.append(feat)
        diags_acc.append(diag)
        y.append(hr_ecg)
        groups.append(pid)

    if not feats:
        raise RuntimeError(f"no sessions collected for {label}")

    X = _sanitize(np.stack(feats))
    yv = np.array(y, dtype=float)
    gv = np.array(groups)
    metrics = ridge_eval(X, yv, gv)

    # aggregate diagnostics (mean over sessions; nan_inf summed)
    agg: Dict[str, float] = {}
    keys = diags_acc[0].keys()
    for k in keys:
        vals = [d[k] for d in diags_acc]
        agg[k] = float(np.nansum(vals)) if k == "nan_inf" else float(np.nanmean(vals))

    return {
        "label": label,
        "n_sessions": len(yv),
        "n_participants": int(len(np.unique(gv))),
        "metrics": metrics,
        "diagnostics": agg,
    }


def build_candidate_configs(fs: float,
                            stages: List[str] | None = None) -> List[Dict[str, Any]]:
    """Enumerate one-stage-at-a-time candidate configs (excludes baselines)."""
    methods = list_stage_methods()
    configs: List[Dict[str, Any]] = []
    for stage, tag in STAGE_CANDIDATES.items():
        if stages is not None and stage not in stages:
            continue
        base = RDAConfig(
            clutter=BASELINE_METHOD["clutter"],
            localization=BASELINE_METHOD["localization"],
            range_selection=BASELINE_METHOD["range_selection"],
            beamforming=BASELINE_METHOD["beamforming"],
            fs=fs,
        )
        for method in methods[stage]:
            if method in (BASELINE_METHOD.get(stage), "none", ""):
                continue
            cfg = base.to_dict()
            if stage == "phase":
                # phase stage is cross-layer: baseline cube stages + this phase method
                cfg["phase"] = method
            else:
                cfg[stage] = method
            configs.append({
                "stage": stage,
                "tag": tag,
                "method": method,
                "config": cfg,
            })
    return configs


def classify(status_inputs: List[Dict[str, Any]], improve: float, worsen: float,
             diag_floor: float) -> None:
    """Annotate each result with a DROP/HOLD/PROMOTE status in place."""
    # baselines keyed by stage group
    base_mae = {}
    base_hrpwr = {}
    for r in status_inputs:
        if r["stage"] == "baseline_cube":
            base_mae["cube"] = r["metrics"]["mae_bpm"]
            base_hrpwr["cube"] = r["diagnostics"]["hr_band_power"]
        elif r["stage"] == "baseline_phase":
            base_mae["phase"] = r["metrics"]["mae_bpm"]
            base_hrpwr["phase"] = r["diagnostics"]["hr_band_power"]

    for r in status_inputs:
        if r["stage"].startswith("baseline"):
            r["status"] = "BASELINE"
            r["delta_mae"] = 0.0
            continue
        grp = "phase" if r["tag"] == "P" else "cube"
        ref = base_mae.get(grp)
        ref_hr = base_hrpwr.get(grp)
        delta = (r["metrics"]["mae_bpm"] - ref) if ref is not None else 0.0
        r["delta_mae"] = float(delta)

        diag = r["diagnostics"]
        # degradation guardrails
        degraded = False
        reasons = []
        if diag["nan_inf"] > 0:
            degraded, reasons = True, reasons + ["nan_inf>0"]
        if ref_hr and diag["hr_band_power"] < diag_floor * ref_hr:
            degraded, reasons = True, reasons + ["hr_band_power collapsed"]
        if not np.isfinite(diag["hr_band_power"]) or diag["hr_band_power"] <= 0:
            degraded, reasons = True, reasons + ["hr_band_power<=0"]

        if degraded or delta >= worsen:
            r["status"] = "DROP"
            r["status_reason"] = ";".join(reasons) if reasons else f"delta_mae>={worsen:.2f}"
        elif delta <= -improve:
            r["status"] = "PROMOTE" if not degraded else "HOLD"
            r["status_reason"] = "improve" if not degraded else "improved_but_diag"
        else:
            r["status"] = "HOLD"
            r["status_reason"] = f"|delta|={abs(delta):.2f}<{improve:.2f}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    cols = ["stage", "tag", "method", "downstream", "mae_bpm", "r2",
            "pearson_r", "delta_mae", "status", "status_reason",
            "nan_inf", "var", "abs_mean", "std", "discontinuity",
            "hr_band_power", "spectral_flatness", "temporal_cont"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(stage=r.get("stage"), tag=r.get("tag"),
                       method=r.get("method"), downstream=r.get("downstream"),
                       mae_bpm=r["metrics"]["mae_bpm"], r2=r["metrics"]["r2"],
                       pearson_r=r["metrics"]["pearson_r"],
                       delta_mae=r.get("delta_mae"),
                       status=r.get("status"), status_reason=r.get("status_reason"),
                       **r.get("diagnostics", {}))
            # normalize types
            for k in cols:
                if k in row and isinstance(row[k], float):
                    row[k] = round(row[k], 5)
            w.writerow(row)


def write_md(path: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Phase-0 RDA Candidate Screening — mmwave-897-2026\n")
    lines.append(f"- generated: {meta['generated']}")
    lines.append(f"- dataset: {meta['dataset']}")
    lines.append(f"- sessions: {meta['n_sessions']}  (participants: {meta['n_participants']})")
    lines.append(f"- fs: {meta['fs']} Hz   spect: 71-d log1p HR-band (F_GRID)")
    lines.append(f"- margins: improve={meta['improve']}  worsen={meta['worsen']}  "
                f"hr-power floor={meta['diag_floor']}")
    lines.append(f"- known linear-probe reference: MAE {KNOWN_REFERENCE_MAE} / "
                f"R² {KNOWN_REFERENCE_R2} (single_bin, full 440)")
    lines.append("")

    lines.append("## D=1 compatibility adapter (IMPORTANT)\n")
    lines.append(meta["d1_note"])
    lines.append("")
    lines.append("## Per-stage validity on mmwave-897\n")
    for k, v in meta["stage_validity"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Leaderboard\n")
    lines.append("| Stage | Candidate | Downstream | MAE | ΔMAE | R² | r | "
                 "HR-pwr | Status |")
    lines.append("|-------|-----------|-----------|----:|-----:|---:|---:|------:|--------|")
    order = {"C": 0, "L": 1, "R": 2, "B": 3, "P": 4}
    for r in sorted(rows, key=lambda x: (order.get(x.get("tag"), 9),
                                         x["metrics"]["mae_bpm"])):
        tag = r.get("tag", "-")
        method = r.get("method", r.get("stage", ""))
        ds = r.get("downstream", "-")
        mae = r["metrics"]["mae_bpm"]
        d = r.get("delta_mae", 0.0)
        r2 = r["metrics"]["r2"]
        rr = r["metrics"]["pearson_r"]
        hrp = r.get("diagnostics", {}).get("hr_band_power", float("nan"))
        st = r.get("status", "")
        lines.append(f"| {tag} | {method} | {ds} | {mae:.2f} | {d:+.2f} | "
                     f"{r2:.3f} | {rr:.3f} | {hrp:.4g} | {st} |")
    lines.append("")

    lines.append("## Promotion policy\n")
    lines.append("- **DROP**: MAE worsens by ≥ worsen margin, or diagnostics degraded "
                 "(NaN/Inf present, HR-band power collapsed).")
    lines.append("- **HOLD**: ≈ baseline, no clear evidence (within improve margin).")
    lines.append("- **PROMOTE**: MAE improves by ≥ improve margin AND diagnostics not "
                 "degraded. Only PROMOTE candidates proceed to cross-dataset validation "
                 "(PhysDrive / FTU / BGT60).")
    lines.append("")
    lines.append("## Next step\n")
    promoted = [r for r in rows if r.get("status") == "PROMOTE"]
    if promoted:
        names = ", ".join(f"{r.get('tag')}/{r.get('method')}" for r in promoted)
        lines.append(f"PROMOTED: {names}. Run cross-dataset validation before any NN.")
    else:
        lines.append("No candidate PROMOTED at current margins. Revisit thresholds or "
                     "search new candidates; do NOT proceed to NN yet.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auto_find_dataset():
    """Locate the mmwave-897-2026 root without an explicit --dataset (Kaggle-safe).

    The mounted path varies (dataset slug may differ from the ref, and there may
    be an extra nesting level), so fall back to globbing for a P001 directory.
    """
    cands = [
        Path("/kaggle/input/mmwave-897-2026"),
        Path("/kaggle/input/goldfish9901/mmwave-897-2026"),
        Path.cwd(),
    ]
    for c in cands:
        if (c / "P001").exists():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for p in sorted(base.glob("**/P001")):
            if p.is_dir():
                return p.parent
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-0 RDA candidate screening harness")
    ap.add_argument("--dataset", type=Path, required=False, default=None,
                    help="mmwave-897-2026 root (contains P001/ ...); "
                         "auto-detected on Kaggle if omitted")
    ap.add_argument("--all", action="store_true", help="use all participants")
    ap.add_argument("--participants", type=int, nargs="+", default=None)
    ap.add_argument("--max-participants", type=int, default=None,
                    help="use only the first N participants (smoke mode)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--rep", type=str, default="single_bin",
                    help="cube-stage representation for Ridge (default single_bin)")
    ap.add_argument("--stages", type=str, default=None,
                    help="comma-separated subset of stages to screen "
                         "(clutter,localization,range_selection,beamforming,phase); "
                         "default = all")
    ap.add_argument("--phase-bp", action="store_true",
                    help="band-pass pool phase traces to the HR band before the "
                         "71-d spectrum, matching the canonical baseline trace "
                         "(removes the band-pass confound from the P comparison)")
    ap.add_argument("--improve", type=float, default=0.5,
                    help="MAE improvement (BPM) to PROMOTE")
    ap.add_argument("--worsen", type=float, default=0.5,
                    help="MAE worsening (BPM) to DROP")
    ap.add_argument("--diag-floor", type=float, default=0.5,
                    help="drop if candidate HR-band power < floor * baseline")
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="skip registry methods by name (e.g. rpca pca)")
    args = ap.parse_args()

    if args.dataset is None:
        ds = _auto_find_dataset()
        if ds is None:
            ap.error("could not auto-locate mmwave-897-2026; pass --dataset")
        args.dataset = ds
        print(f"[dataset] auto-detected: {args.dataset}", flush=True)

    loader = MMWaveDataLoader(str(args.dataset))
    # No participant spec -> run the full dataset (the cheap mmwave-897 screen).
    if not (args.all or args.participants or args.max_participants):
        args.all = True
    if args.all:
        participants = sorted({p for p, _, _ in loader.list_sessions()})
    else:
        participants = args.participants or [1]
    if args.max_participants:
        participants = participants[: args.max_participants]

    fs = loader.FRAME_RATE_HZ  # mmwave-897 = 10.0

    results: List[Dict[str, Any]] = []

    # ---- baselines -------------------------------------------------------
    base_cfg = RDAConfig(
        clutter=BASELINE_METHOD["clutter"],
        localization=BASELINE_METHOD["localization"],
        range_selection=BASELINE_METHOD["range_selection"],
        beamforming=BASELINE_METHOD["beamforming"],
        fs=fs,
    )

    def cube_baseline_extractor(raw, fs_, hr_band):
        return features_cube_stage(raw, base_cfg, fs_, hr_band, rep=args.rep)

    print("[baseline] cube (identity transform -> build_features single_bin)", flush=True)
    rb = screen_one(loader, participants, cube_baseline_extractor, "baseline_cube")
    rb["stage"] = "baseline_cube"
    rb["tag"] = "-"
    rb["method"] = "identity"
    rb["downstream"] = args.rep
    results.append(rb)
    print(f"   baseline cube MAE={rb['metrics']['mae_bpm']:.2f} "
          f"R²={rb['metrics']['r2']:.3f}", flush=True)

    def phase_baseline_extractor(raw, fs_, hr_band):
        # phase-stage baseline = canonical single-bin trace -> spectrum
        trace = single_bin_phase_trace(raw, fs_, hr_band)
        return trace_to_features(trace, fs_, hr_band), compute_diagnostics(trace, fs_, hr_band)

    print("[baseline] phase (canonical single-bin trace)", flush=True)
    rpb = screen_one(loader, participants, phase_baseline_extractor, "baseline_phase")
    rpb["stage"] = "baseline_phase"
    rpb["tag"] = "P"
    rpb["method"] = "single_bin_trace"
    rpb["downstream"] = "phase_trace"
    results.append(rpb)
    print(f"   baseline phase MAE={rpb['metrics']['mae_bpm']:.2f} "
          f"R²={rpb['metrics']['r2']:.3f}", flush=True)

    # ---- candidates (one stage swapped at a time) ------------------------
    skip = set(args.skip)
    stages = ([s.strip() for s in args.stages.split(",") if s.strip()]
              if args.stages else None)
    for cand in build_candidate_configs(fs, stages):
        method = cand["method"]
        if method in skip:
            print(f"[skip] {cand['tag']}/{method}", flush=True)
            continue
        cfg = RDAConfig.from_dict(cand["config"])
        try:
            cfg.validate()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cand['tag']}/{method} invalid: {e}", flush=True)
            continue

        if cand["stage"] == "phase":
            def ext(raw, fs_, hr_band, _c=cfg):
                return features_phase_stage(raw, _c, fs_, hr_band,
                                            bandpass=args.phase_bp)
            ds = "phase_trace_bp" if args.phase_bp else "phase_trace"
        else:
            def ext(raw, fs_, hr_band, _c=cfg):
                return features_cube_stage(raw, _c, fs_, hr_band, rep=args.rep)
            ds = args.rep

        print(f"[screen] {cand['tag']}/{method}", flush=True)
        r = screen_one(loader, participants, ext, f"{cand['tag']}/{method}")
        r["stage"] = cand["stage"]
        r["tag"] = cand["tag"]
        r["method"] = method
        r["downstream"] = ds
        results.append(r)
        print(f"   MAE={r['metrics']['mae_bpm']:.2f} R²={r['metrics']['r2']:.3f} "
              f"r={r['metrics']['pearson_r']:.3f}", flush=True)

    # ---- classify --------------------------------------------------------
    classify(results, args.improve, args.worsen, args.diag_floor)

    # ---- outputs ---------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_sessions": results[0]["n_sessions"],
        "n_participants": results[0]["n_participants"],
        "fs": fs,
        "rep": args.rep,
        "stages": stages or "all",
        "phase_bp": bool(args.phase_bp),
        "improve": args.improve,
        "worsen": args.worsen,
        "diag_floor": args.diag_floor,
        "d1_note": (
            "mmwave-897 ships as (F, 8, 64) with NO Doppler axis (chirps averaged at "
            "storage). The D=1 inserted by ensure_rda_4d is a COMPATIBILITY ADAPTER, "
            "not a real Doppler bin. Doppler-processing conclusions are LIMITED on this "
            "dataset. PhysDrive (F,8,16,8) is the fuller RDA target."),
        "stage_validity": {
            "C (clutter)": "valid on mmwave-897 (slow-time processing; no Doppler needed)",
            "L (localization)": "valid on mmwave-897 (range/angle energy & HR-band)",
            "R (range selection)": "valid on mmwave-897 (range axis present)",
            "B (beamforming)": "LIMITED on mmwave-897 — angle axis is antennas, not true "
                               "angular-Doppler; MVDR/Bartlett still test spatial "
                               "reweighting but NOT Doppler discrimination",
            "P (phase)": "valid on mmwave-897 (single/multi-bin phase extraction)",
        },
    }

    # strip heavy config blobs from JSON rows but keep method+stage
    json_rows = []
    for r in results:
        json_rows.append({
            "stage": r.get("stage"), "tag": r.get("tag"),
            "method": r.get("method"), "downstream": r.get("downstream"),
            "metrics": r["metrics"], "diagnostics": r["diagnostics"],
            "status": r.get("status"), "delta_mae": r.get("delta_mae"),
            "status_reason": r.get("status_reason"),
        })

    (args.out_dir / "phase0_rda_screen.json").write_text(
        json.dumps({"meta": meta, "results": json_rows}, indent=2, ensure_ascii=False))
    write_csv(args.out_dir / "phase0_rda_screen.csv", results)
    write_md(args.out_dir / "phase0_rda_screen.md", results, meta)

    print(f"\nDone. Artifacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
