"""Phase-0d: is the HR band dominated by respiration harmonics?

Motivation (from the round-1..3 measurements)
---------------------------------------------
* Round 3 (`phase0_bin_oracle_diag`): the classical FFT-peak estimator on the
  canonical bin (#2, 0.63 m) has MAE 15.74 overall, but **7.14 for Lying/Rest
  versus 25.23 for Sitting/Post-exercise**. It fails exactly where HR is far from
  resting values.
* The "oracle" bin reaches 1.64 BPM, and beats a uniform-random-target control
  (10.19) by 6x. That gap is suspicious: if the per-bin estimates were spread
  over the band, min-over-62-bins would be small for *any* target. A large gap
  means the estimates are **clustered**, and the cluster sits near typical
  resting HR.
* Round 3b (`phase0_rep_probe`): feeding the *oracle* bin's spectrum to the Ridge
  gives 12.28 versus 12.23 for bin 2 -- perfect range-bin selection buys nothing.
  So the Ridge is not reading peak position at all.

Put together: respiration at f_r ~ 0.25 Hz has harmonics at 3f_r = 0.75 Hz
(45 BPM), 4f_r = 1.0 Hz (60 BPM), 5f_r = 1.25 Hz (75 BPM) -- right inside the
0.8-2.0 Hz HR band. A peak-picking estimator locks onto those. It "works" when
true HR happens to sit on a harmonic and collapses otherwise.

This script measures the mechanism directly and tests the obvious fix:

1. **Mechanism**: per session, estimate f_r from the sub-HR band, then report how
   often the HR-band peak actually *is* a respiration harmonic, plus MAE split by
   true-HR decile (the smoking gun is error growing with |HR - 60|).
2. **Fix (readout)**: pick the HR-band peak *excluding* +-harmonic windows.
3. **Fix (signal)**: subtract a least-squares harmonic regression at k*f_r
   (k = 2..8) from the unwrapped phase, then rebuild the 71-d spectrum and re-run
   the identical subject-level GroupKFold(5) Ridge.

Everything downstream (feature build + CV) is identical to `mmwave_probe`, so
`plain` must reproduce the 12.23 / 15.7 references.

Usage
-----
    python experiments/mmwave_selection/phase0_resp_harmonic_probe.py --all \
        --out-dir experiments/mmwave_selection/resp_harmonic_output
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
_build_features = _p.build_features
_sanitize = _p._sanitize

_b = _import("src.data.mmwave_baseline")
REST_HR_BAND_HZ = _b.REST_HR_BAND_HZ
POSTEX_HR_BAND_HZ = _b.POSTEX_HR_BAND_HZ
butter_bandpass = _b.butter_bandpass
extract_phase = _b.extract_phase
hr_from_ecg = _b.hr_from_ecg

_l = _import("src.data.loaders.mmwave_loader")
MMWaveDataLoader = _l.MMWaveDataLoader

import phase0_rda_screen as _screen  # noqa: E402  (ridge_eval + CV, same as probe)

ridge_eval = _screen.ridge_eval
trace_to_features = _screen.trace_to_features

_working = Path("/kaggle/working")
DEFAULT_OUT_DIR = (_working / "resp_harmonic_output") if _working.is_dir() else (
    Path(__file__).resolve().parent / "resp_harmonic_output")

NFFT = 4096
RESP_BAND = (0.10, 0.60)          # breathing
HARM_K = (2, 3, 4, 5, 6, 7, 8)    # harmonics that can land inside the HR band
HARM_GUARD_HZ = 0.08              # +- window when excluding harmonic peaks


def _spectrum(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import get_window
    n = sig.size
    w = sig * get_window("hann", n)
    spec = np.abs(np.fft.rfft(w, NFFT))
    freqs = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    return freqs, spec


def pick_peak(spec: np.ndarray, freqs: np.ndarray, band: Tuple[float, float],
              forbid: np.ndarray | None = None) -> Tuple[float, float]:
    """Strongest peak in ``band``; ``forbid`` is a boolean mask of frequencies to
    skip (respiration harmonics). Returns (freq_hz, amplitude)."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    if forbid is not None:
        m &= ~forbid
    if not m.any():
        m = (freqs >= band[0]) & (freqs <= band[1])
    idx = int(np.argmax(spec[m]))
    return float(freqs[m][idx]), float(spec[m][idx])


def resp_rate(phase: np.ndarray, fs: float) -> float:
    freqs, spec = _spectrum(phase - phase.mean(), fs)
    f, _ = pick_peak(spec, freqs, RESP_BAND)
    return f


def harmonic_mask(freqs: np.ndarray, f_r: float,
                  guard: float = HARM_GUARD_HZ) -> np.ndarray:
    mask = np.zeros(freqs.shape, dtype=bool)
    for k in HARM_K:
        mask |= np.abs(freqs - k * f_r) <= guard
    return mask


def remove_harmonics(phase: np.ndarray, fs: float, f_r: float) -> np.ndarray:
    """Least-squares removal of k*f_r sinusoids (k = 2..8) + linear trend."""
    t = np.arange(phase.size, dtype=np.float64) / float(fs)
    cols = [np.ones_like(t), t]
    for k in HARM_K:
        cols.append(np.cos(2.0 * np.pi * k * f_r * t))
        cols.append(np.sin(2.0 * np.pi * k * f_r * t))
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(phase, dtype=np.float64), rcond=None)
    fit = A[:, 2:] @ coef[2:]          # remove harmonics, keep DC/trend
    return np.asarray(phase, dtype=np.float64) - fit


def canonical_bin(radar: np.ndarray) -> int:
    """mmwave_probe's `single_bin` selection: argmax of (mean|z|)^2, bins >= 2."""
    energy = np.abs(radar).mean(axis=(0, 1)) ** 2
    return int(np.argmax(energy[EXCLUDE_FIRST:]) + EXCLUDE_FIRST)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--participants", type=int, nargs="+", default=None)
    ap.add_argument("--max-participants", type=int, default=None)
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

    rows: List[Dict[str, Any]] = []
    n = 0
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

                b = canonical_bin(radar)
                phase = extract_phase(radar, b)
                f_r = resp_rate(phase, fs)

                freqs, spec = _spectrum(butter_bandpass(phase, fs, hr_band), fs)
                f_peak, _ = pick_peak(spec, freqs, hr_band)
                forbid = harmonic_mask(freqs, f_r)
                f_peak_nh, _ = pick_peak(spec, freqs, hr_band, forbid=forbid)

                # is the winning peak itself sitting on a respiration harmonic?
                on_harm = bool(np.min(np.abs(
                    np.array([k * f_r for k in HARM_K]) - f_peak)) <= HARM_GUARD_HZ)

                clean = remove_harmonics(phase, fs, f_r)
                clean_bp = butter_bandpass(clean, fs, hr_band)
                freqs2, spec2 = _spectrum(clean_bp, fs)
                f_peak_clean, _ = pick_peak(spec2, freqs2, hr_band,
                                            forbid=harmonic_mask(freqs2, f_r))

                rows.append({
                    "pid": pid,
                    "stratum": f"{posture}/{condition}",
                    "bin": b,
                    "hr_ecg": float(hr_ecg),
                    "f_r": float(f_r),
                    "f_peak": f_peak,
                    "f_peak_nh": f_peak_nh,
                    "f_peak_clean": f_peak_clean,
                    "on_harm": on_harm,
                    # representations for the Ridge probe
                    "feat_plain": _sanitize(
                        _build_features(radar, fs, hr_band, "single_bin")),
                    "feat_clean": trace_to_features(clean_bp, fs, hr_band),
                })
                n += 1
                if n % 100 == 0:
                    print(f"[extract] {n} sessions", flush=True)
    print(f"[extract] {n} sessions total", flush=True)

    def mae(key: str) -> float:
        e = [abs(r[key] * 60.0 - r["hr_ecg"]) for r in rows]
        return float(np.mean(e))

    peaks = {
        "plain": mae("f_peak"),
        "peak_excl_harmonics": mae("f_peak_nh"),
        "harmonic_removed_then_peak": mae("f_peak_clean"),
    }
    harm_frac = float(100.0 * np.mean([r["on_harm"] for r in rows]))

    y = np.array([r["hr_ecg"] for r in rows], dtype=float)
    g = np.array([r["pid"] for r in rows])
    ridge: Dict[str, Any] = {}
    for rep, key in (("single_bin_plain", "feat_plain"),
                     ("harmonic_removed", "feat_clean")):
        X = np.stack([r[key] for r in rows])
        m = ridge_eval(X, y, g, ALPHAS)
        ridge[rep] = m
        print(f"[ridge] {rep:20s} MAE={m['mae_bpm']:.2f} R²={m['r2']:.3f} "
              f"r={m['pearson_r']:.3f}", flush=True)

    # ---- MAE by true-HR decile: the smoking gun -------------------------
    order = np.argsort(y)
    dec = np.empty(len(y), dtype=int)
    dec[order] = np.minimum((np.arange(len(y)) * 10) // len(y), 9)
    by_hr = []
    for d in range(10):
        sel = np.flatnonzero(dec == d)
        by_hr.append({
            "decile": d + 1,
            "hr_lo": float(y[sel].min()), "hr_hi": float(y[sel].max()),
            "hr_mean": float(y[sel].mean()),
            "mae_plain": float(np.mean([abs(rows[i]["f_peak"] * 60 - rows[i]["hr_ecg"])
                                        for i in sel])),
            "mae_excl_harm": float(np.mean([abs(rows[i]["f_peak_nh"] * 60 - rows[i]["hr_ecg"])
                                            for i in sel])),
            "mae_harm_removed": float(np.mean([abs(rows[i]["f_peak_clean"] * 60 - rows[i]["hr_ecg"])
                                               for i in sel])),
        })

    strata = sorted({r["stratum"] for r in rows})
    by_stratum = {}
    for st in strata:
        sub = [r for r in rows if r["stratum"] == st]
        by_stratum[st] = {
            "mae_plain": float(np.mean([abs(r["f_peak"] * 60 - r["hr_ecg"]) for r in sub])),
            "mae_excl_harm": float(np.mean([abs(r["f_peak_nh"] * 60 - r["hr_ecg"]) for r in sub])),
            "mae_harm_removed": float(np.mean([abs(r["f_peak_clean"] * 60 - r["hr_ecg"]) for r in sub])),
            "pct_peak_on_harmonic": float(100.0 * np.mean([r["on_harm"] for r in sub])),
            "f_r_mean": float(np.mean([r["f_r"] for r in sub])),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_sessions": n,
        "n_participants": int(len(np.unique(g))),
        "fs": loader.FRAME_RATE_HZ,
        "harmonics_used": list(HARM_K),
        "guard_hz": HARM_GUARD_HZ,
        "peak_picker_mae": peaks,
        "pct_peak_on_resp_harmonic": harm_frac,
        "ridge": ridge,
        "by_true_hr_decile": by_hr,
        "by_stratum": by_stratum,
    }
    (args.out_dir / "resp_harmonic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    lines = ["# Respiration-harmonic hypothesis (mmwave-897, canonical bin)\n",
             f"- sessions: {n}   participants: {payload['n_participants']}"
             f"   fs: {loader.FRAME_RATE_HZ} Hz",
             f"- breathing band {RESP_BAND} Hz; harmonics k={list(HARM_K)}; "
             f"exclusion guard ±{HARM_GUARD_HZ} Hz\n",
             "## Mechanism\n",
             f"- HR-band peak sits on a respiration harmonic in "
             f"**{harm_frac:.1f}%** of sessions\n",
             "## FFT-peak estimator (MAE, BPM)\n",
             "| variant | MAE |", "|---|---:|"]
    for k, v in peaks.items():
        lines.append(f"| {k} | {v:.2f} |")
    lines += ["", "## Ridge on the 71-d spectrum (identical CV)\n",
              "| representation | MAE | R² | r |", "|---|---:|---:|---:|"]
    for k, v in ridge.items():
        lines.append(f"| {k} | {v['mae_bpm']:.2f} | {v['r2']:.3f} | "
                     f"{v['pearson_r']:.3f} |")
    lines += ["", "## MAE by true-HR decile (the smoking gun)\n",
              "| decile | HR range | mean HR | peak | peak excl. harm | harm removed |",
              "|---|---|---:|---:|---:|---:|"]
    for d in by_hr:
        lines.append(f"| {d['decile']} | {d['hr_lo']:.0f}-{d['hr_hi']:.0f} | "
                     f"{d['hr_mean']:.1f} | {d['mae_plain']:.2f} | "
                     f"{d['mae_excl_harm']:.2f} | {d['mae_harm_removed']:.2f} |")
    lines += ["", "## By stratum\n",
              "| stratum | peak | excl. harm | harm removed | %peak on harmonic | f_r |",
              "|---|---:|---:|---:|---:|---:|"]
    for st in strata:
        s = by_stratum[st]
        lines.append(f"| {st} | {s['mae_plain']:.2f} | {s['mae_excl_harm']:.2f} | "
                     f"{s['mae_harm_removed']:.2f} | {s['pct_peak_on_harmonic']:.1f} | "
                     f"{s['f_r_mean']:.3f} |")
    (args.out_dir / "resp_harmonic.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[result] peak-on-harmonic fraction: {harm_frac:.1f}%", flush=True)
    for k, v in peaks.items():
        print(f"[result] {k:28s} MAE={v:.2f}", flush=True)
    print(f"\nDone. Artifacts in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
