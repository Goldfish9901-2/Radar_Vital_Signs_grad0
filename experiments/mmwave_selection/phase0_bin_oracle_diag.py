"""Phase-0b: is there ANY unsupervised range-bin score that approaches the oracle?

Why this exists
---------------
Round 1 of the RDA candidate screen (``phase0_rda_screen.py``) came back flat:
all 15 classical single-stage swaps sit at MAE ~12.4 BPM, nowhere near the
12.23 linear-probe reference. But two earlier measurements show the HR is fully
present in the data and that the bottleneck is *range-bin selection*:

* ``experiments/mmwave_baseline/summary.json`` (P001-P010, paper Fig.5 FFT-peak
  baseline): oracle bin MAE 0.18 / 0.43 / 0.25 / 1.32 BPM vs max-energy bin
  2.97 / 12.46 / 11.55 / 28.09 BPM (Lying/Rest, Lying/Post-ex, Sitting/Rest,
  Sitting/Post-ex).
* ``mmwave_split_half_results.json`` (440 sessions): oracle over all bins 1.64
  BPM, but pick the bin on the first half and evaluate on the second -> 18.32.

So screening more filters is not the bottleneck. The question this script
answers is narrower and decisive:

    For every range bin, compute (a) the HR estimate error against ECG and
    (b) a battery of *unsupervised* bin-quality scores. Then report the honest
    MAE of "pick the top bin by score X" for each score, next to the oracle and
    the current max-energy baseline.

If some unsupervised score gets materially closer to the oracle than
max-energy, that score is the next RDA candidate (and it is cheap to
implement). If none does, hand-designed bin selection is a dead end on this
dataset and the search has to move to learned/temporal selection.

The ECG HR is used ONLY to measure error -- never to select a bin inside a
candidate rule.

Outputs
-------
    <out-dir>/bin_oracle_diag.json   full per-score statistics + decile tables
    <out-dir>/bin_oracle_diag.md     human-readable leaderboard

Usage
-----
    python experiments/mmwave_selection/phase0_bin_oracle_diag.py --all \
        --out-dir experiments/mmwave_selection/bin_oracle_output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "radar" / "__init__.py").exists() or \
           (parent / "src" / "data" / "mmwave_baseline.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return here.parents[1]


_repo_root()


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


_b = _import("src.data.mmwave_baseline")
REST_HR_BAND_HZ = _b.REST_HR_BAND_HZ
POSTEX_HR_BAND_HZ = _b.POSTEX_HR_BAND_HZ
butter_bandpass = _b.butter_bandpass
extract_phase = _b.extract_phase
hr_from_phase = _b.hr_from_phase
hr_from_ecg = _b.hr_from_ecg

_l = _import("src.data.loaders.mmwave_loader")
MMWaveDataLoader = _l.MMWaveDataLoader

_working = Path("/kaggle/working")
DEFAULT_OUT_DIR = (_working / "bin_oracle_output") if _working.is_dir() else (
    Path(__file__).resolve().parent / "bin_oracle_output")

EXCLUDE_FIRST = 2          # first range bins are antenna-bleed / DC, as in mmwave_probe


# ---------------------------------------------------------------------------
# per-bin HR estimate + unsupervised quality scores
# ---------------------------------------------------------------------------

def _band_spectrum(filtered: np.ndarray, fs: float,
                   hr_band: Tuple[float, float], nfft: int = 2048
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann-windowed zero-padded spectrum of the band-passed trace."""
    from scipy.signal import get_window
    n = filtered.size
    windowed = filtered * get_window("hann", n)
    spec = np.abs(np.fft.rfft(windowed, nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    in_band = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    return spec, freqs, in_band


def bin_table(radar: np.ndarray, fs: float,
              hr_band: Tuple[float, float]) -> Dict[str, np.ndarray]:
    """Per-bin HR estimate, error-free scores. All arrays are (n_bins,).

    ``radar`` is (F, A, R) complex. Bins before EXCLUDE_FIRST are dropped.
    """
    from scipy.signal import get_window

    radar = np.asarray(radar)
    n_f, n_a, n_r = radar.shape
    idx = np.arange(EXCLUDE_FIRST, n_r)
    n = n_f
    nfft = max(2048, 2 ** int(np.ceil(np.log2(n))))

    hr = np.full(idx.size, np.nan)
    peak_amp = np.zeros(idx.size)
    peak_prom = np.zeros(idx.size)
    peak_ratio = np.zeros(idx.size)
    band_frac = np.zeros(idx.size)
    hr_power = np.zeros(idx.size)
    phase_var = np.zeros(idx.size)
    temp_cont = np.zeros(idx.size)
    ant_coh = np.zeros(idx.size)
    spec_entropy = np.zeros(idx.size)
    energy = np.zeros(idx.size)                     # E[|z|^2]  (variance-inclusive)
    probe_energy = np.zeros(idx.size)               # (E[|z|])^2 -- mmwave_probe's criterion
    sub_peak_ratio = np.zeros(idx.size)   # respiration leakage proxy
    bp_matrix: List[np.ndarray] = []      # HR-band-passed trace per bin

    for k, b in enumerate(idx):
        cell = radar[:, :, b]                       # (F, A)
        energy[k] = float(np.mean(np.abs(cell) ** 2))
        probe_energy[k] = float(np.mean(np.abs(cell)) ** 2)
        # antenna circular coherence: |mean_a exp(j*angle(z))| averaged over frames
        ang = np.angle(cell)
        ant_coh[k] = float(np.mean(np.abs(np.mean(np.exp(1j * ang), axis=1))))

        phase = extract_phase(radar, int(b))
        phase_var[k] = float(np.var(phase))
        filtered = butter_bandpass(phase, fs, hr_band)
        bp_matrix.append(filtered)
        hr_power[k] = float(np.var(filtered))
        if filtered.size > 2 and np.std(filtered[:-1]) > 1e-12 and np.std(filtered[1:]) > 1e-12:
            temp_cont[k] = float(np.corrcoef(filtered[:-1], filtered[1:])[0, 1])

        spec, freqs, in_band = _band_spectrum(filtered, fs, hr_band, nfft)
        band = spec[in_band]
        if band.size == 0 or not np.isfinite(band).all():
            continue
        pk = int(np.argmax(band))
        peak_amp[k] = float(band[pk])
        med = float(np.median(band))
        mean = float(np.mean(band))
        peak_prom[k] = peak_amp[k] / (med + 1e-12)
        peak_ratio[k] = peak_amp[k] / (mean + 1e-12)
        hr[k] = float(freqs[in_band][pk]) * 60.0

        # spectral concentration inside the band (low entropy = a clean peak)
        p = band / (band.sum() + 1e-12)
        spec_entropy[k] = float(-np.sum(p * np.log(p + 1e-12)))

        # fraction of total phase energy that lands in the HR band
        full = np.abs(np.fft.rfft(phase)) ** 2
        ff = np.fft.rfftfreq(phase.size, d=1.0 / fs)
        band_frac[k] = float(full[(ff >= hr_band[0]) & (ff <= hr_band[1])].sum()
                             / (full.sum() + 1e-12))

        # sub-band (respiration) leakage: strongest peak below the HR band
        below = (ff >= 0.05) & (ff < hr_band[0])
        if below.any():
            sub_peak_ratio[k] = float(np.max(full[below]) / (full.sum() + 1e-12))

    e_peak = int(np.argmax(energy))
    dist_peak = -np.abs(np.arange(idx.size) - e_peak).astype(float)

    # --- data-driven spatial combination (unsupervised) -------------------
    # Every bin sees the same chest wall, so the common component across bins
    # is a far better spatial filter than "pick one bin". PC1 of the
    # HR-band-passed bin ensemble is the cheapest such filter.
    bp = np.asarray(bp_matrix)                      # (n_bins, frames)
    pc1 = np.zeros(n_f)
    pc1_corr = np.zeros(idx.size)
    mean_bp = np.zeros(n_f)
    if bp.size:
        X = bp - bp.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        Xs = X / np.where(norm > 1e-12, norm, 1.0)
        # weight bins by nothing -- plain PCA over bins
        u, s, vt = np.linalg.svd(Xs, full_matrices=False)
        if s[0] > 1e-9:
            pc1 = vt[0] * s[0]
            pc1_corr = np.abs(Xs @ vt[0])
        mean_bp = bp.mean(axis=0)

    scores = {
        # exactly mmwave_probe.build_features(rep='single_bin')'s selection rule
        "probe_energy": np.log1p(probe_energy),
        "energy": np.log1p(energy),
        "hr_power": np.log1p(hr_power),
        "peak_amp": np.log1p(peak_amp),
        "peak_prom": peak_prom,
        "peak_ratio": peak_ratio,
        "band_frac": band_frac,
        "phase_var": np.log1p(phase_var),
        "temp_cont": temp_cont,
        "ant_coh": ant_coh,
        "spec_entropy": -spec_entropy,        # higher = more peaky = better
        "near_energy_peak": dist_peak,
        "low_leak": -sub_peak_ratio,          # higher = less respiration leakage
        "prom_x_coh": peak_prom * ant_coh,
        "prom_x_temp": peak_prom * np.clip(temp_cont, 0.0, None),
        "pc1_corr": pc1_corr,
    }
    return {"hr": hr, "scores": scores, "bins": idx,
            "pc1": pc1, "mean_bp": mean_bp, "energy": energy,
            "probe_energy": probe_energy}


# ---------------------------------------------------------------------------
# driver
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
                yield pid, f"{posture}/{condition}", raw, hr_band, hr_ecg


def collect(loader: MMWaveDataLoader, participants: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pid, stratum, raw, hr_band, hr_ecg in iter_sessions(loader, participants):
        t = bin_table(raw, loader.FRAME_RATE_HZ, hr_band)
        hr = t["hr"]
        err = np.abs(hr - hr_ecg)
        ok = np.isfinite(err)
        if not ok.any():
            continue
        extra: Dict[str, float] = {}
        for key, trace in (("pc1", t["pc1"]), ("mean_bp", t["mean_bp"])):
            if trace.size and np.isfinite(trace).all() and np.var(trace) > 1e-12:
                extra[key] = float(hr_from_phase(trace, loader.FRAME_RATE_HZ,
                                                 hr_band)[0])
            else:
                extra[key] = float("nan")
        rows.append({
            "pid": pid,
            "stratum": stratum,
            "hr_ecg": float(hr_ecg),
            "hr": hr,
            "err": err,
            "valid": ok,
            "scores": t["scores"],
            "extra_hr": extra,
            "band": hr_band,
            "e_argmax_bin": int(t["bins"][int(np.argmax(t["energy"]))]),
            "probe_argmax_bin": int(t["bins"][int(np.argmax(t["probe_energy"]))]),
            "oracle_bin": int(t["bins"][int(np.nanargmin(np.where(ok, err, np.inf)))]),
        })
    return rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def _stats(errs: List[float], hrs: List[float] | None = None,
           ecg: List[float] | None = None) -> Dict[str, float]:
    if not errs:
        return {"mae": float("nan"), "median": float("nan"),
                "pct_lt3": float("nan"), "pct_lt5": float("nan"),
                "pct_harm": float("nan"), "n": 0}
    e = np.asarray(errs, dtype=float)
    out = {"mae": float(e.mean()), "median": float(np.median(e)),
           "pct_lt3": float(100.0 * np.mean(e < 3.0)),
           "pct_lt5": float(100.0 * np.mean(e < 5.0)), "n": int(e.size)}
    # harmonic-tolerant hit rate: count it as a hit if k*estimate matches ECG
    # for k in {0.5, 1, 2}. Separates "wrong bin" from "right bin, wrong harmonic".
    if hrs is not None and ecg is not None:
        h = np.asarray(hrs, dtype=float)
        y = np.asarray(ecg, dtype=float)
        best = np.minimum(np.abs(h - y),
                          np.minimum(np.abs(2.0 * h - y), np.abs(0.5 * h - y)))
        out["pct_harm"] = float(100.0 * np.mean(best < 3.0))
    return out


def rule_mae(rows: List[Dict[str, Any]], pick: callable) -> Dict[str, float]:
    """MAE / median / hit-rate of an unsupervised 'pick one bin' rule."""
    errs: List[float] = []
    hrs: List[float] = []
    ecg: List[float] = []
    for r in rows:
        k = pick(r)
        if k is None or not r["valid"][k]:
            continue
        errs.append(float(r["err"][k]))
        hrs.append(float(r["hr"][k]))
        ecg.append(r["hr_ecg"])
    return _stats(errs, hrs, ecg)


def extra_rule_mae(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Stats for a rule that produces its own HR estimate (not a bin pick)."""
    errs, hrs, ecg = [], [], []
    for r in rows:
        h = r["extra_hr"].get(key, float("nan"))
        if not np.isfinite(h):
            continue
        errs.append(abs(float(h) - r["hr_ecg"]))
        hrs.append(float(h))
        ecg.append(r["hr_ecg"])
    return _stats(errs, hrs, ecg)


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

    print(f"[collect] {len(participants)} participants ...", flush=True)
    rows = collect(loader, participants)
    print(f"[collect] {len(rows)} sessions", flush=True)
    if not rows:
        raise RuntimeError("no sessions")

    score_names = sorted(rows[0]["scores"].keys())

    # ---- oracle / baseline ---------------------------------------------
    def pick_oracle(r):
        e = np.where(r["valid"], r["err"], np.inf)
        return int(np.argmin(e)) if np.isfinite(e).any() else None

    def pick_max_energy(r):
        s = np.asarray(r["scores"]["energy"], dtype=float)
        s = np.where(r["valid"], s, -np.inf)
        return int(np.argmax(s))

    def pick_first(r):
        v = np.flatnonzero(r["valid"])
        return int(v[0]) if v.size else None

    results: Dict[str, Dict[str, float]] = {}
    results["oracle_min_err"] = rule_mae(rows, pick_oracle)
    results["max_energy"] = rule_mae(rows, pick_max_energy)
    results["first_valid_bin"] = rule_mae(rows, pick_first)
    # spatial-combination rules (no bin selection at all)
    results["pc1_trace"] = extra_rule_mae(rows, "pc1")
    results["mean_bp_trace"] = extra_rule_mae(rows, "mean_bp")

    # ---- chance-level oracle control ------------------------------------
    # The oracle picks min-error over ~62 bins. Under the null (bin estimates are
    # noise inside the HR band) that selection alone produces a small MAE. This
    # Monte-Carlo replaces the ECG target with a uniform draw from the same band
    # and re-runs the *same* min-over-bins selection, so we learn how much of
    # "oracle 1.64" is just selection overfitting.
    rng_mc = np.random.default_rng(0)
    n_draw = 64
    chance_err, chance_hit = [], []
    for r in rows:
        hr = np.asarray(r["hr"])
        ok = r["valid"]
        cand = hr[ok]
        if cand.size == 0:
            continue
        lo, hi = r["band"][0] * 60.0, r["band"][1] * 60.0
        tgt = rng_mc.uniform(lo, hi, size=n_draw)
        e = np.min(np.abs(cand[None, :] - tgt[:, None]), axis=1)
        chance_err.append(float(e.mean()))
        chance_hit.append(float(100.0 * np.mean(e < 3.0)))
    results["oracle_chance_uniform"] = {
        "mae": float(np.mean(chance_err)), "median": float("nan"),
        "pct_lt3": float(np.mean(chance_hit)), "pct_lt5": float("nan"),
        "pct_harm": float("nan"), "n": len(chance_err)}

    # ---- per-score rules + rank correlation -----------------------------
    corr: Dict[str, float] = {}
    deciles: Dict[str, List[Dict[str, float]]] = {}
    for name in score_names:
        def pick(r, _n=name):
            s = np.asarray(r["scores"][_n], dtype=float)
            s = np.where(r["valid"], s, -np.inf)
            return int(np.argmax(s)) if np.isfinite(s).any() else None
        results[f"top1::{name}"] = rule_mae(rows, pick)

        # pooled Spearman(score, -err) over valid (session, bin) pairs
        a, b = [], []
        for r in rows:
            m = r["valid"] & np.isfinite(r["scores"][name])
            a.append(np.asarray(r["scores"][name])[m])
            b.append(-np.asarray(r["err"])[m])
        corr[name] = _spearman(np.concatenate(a), np.concatenate(b))

        # within-session decile table: is the top decile of the score better?
        bands = np.zeros(10)
        cnt = np.zeros(10)
        err_sum = np.zeros(10)
        for r in rows:
            m = r["valid"] & np.isfinite(r["scores"][name])
            if m.sum() < 10:
                continue
            s = np.asarray(r["scores"][name])[m]
            e = np.asarray(r["err"])[m]
            order = np.argsort(-s)
            d = np.minimum((np.arange(order.size) * 10) // order.size, 9)
            np.add.at(err_sum, d, e[order])
            np.add.at(cnt, d, 1.0)
            np.add.at(bands, d, (e[order] < 3.0).astype(float))
        deciles[name] = [
            {"decile": i + 1, "mae": float(err_sum[i] / max(cnt[i], 1)),
             "pct_lt3": float(100.0 * bands[i] / max(cnt[i], 1)),
             "n": float(cnt[i])} for i in range(10)]

    # ---- per-stratum breakdown for the headline rules --------------------
    strata = sorted({r["stratum"] for r in rows})
    by_stratum: Dict[str, Dict[str, float]] = {}
    for st in strata:
        sub = [r for r in rows if r["stratum"] == st]
        by_stratum[st] = {
            "oracle": rule_mae(sub, pick_oracle)["mae"],
            "max_energy": rule_mae(sub, pick_max_energy)["mae"],
            "pc1_trace": extra_rule_mae(sub, "pc1")["mae"],
            "mean_bp_trace": extra_rule_mae(sub, "mean_bp")["mae"],
            **{f"top1::{n}": rule_mae(
                sub, (lambda r, _n=n: int(np.argmax(
                    np.where(r["valid"], np.asarray(r["scores"][_n], float), -np.inf))))
            )["mae"] for n in score_names},
        }

    # where does the "max energy" rule actually point? (near-field suspicion)
    e_bins = np.array([r["e_argmax_bin"] for r in rows])
    o_bins = np.array([r["oracle_bin"] for r in rows])
    p_bins = np.array([r["probe_argmax_bin"] for r in rows])
    bin_hist = {
        "probe_argmax": {
            "min": int(p_bins.min()), "max": int(p_bins.max()),
            "median": float(np.median(p_bins)),
            "pct_le4": float(100.0 * np.mean(p_bins <= 4)),
            "pct_le8": float(100.0 * np.mean(p_bins <= 8)),
            "hist": {str(k): int(v) for k, v in
                     zip(*np.unique(p_bins, return_counts=True))},
        },
        "max_energy_argmax": {
            "min": int(e_bins.min()), "max": int(e_bins.max()),
            "median": float(np.median(e_bins)),
            "pct_le4": float(100.0 * np.mean(e_bins <= 4)),
            "pct_le8": float(100.0 * np.mean(e_bins <= 8)),
            "hist": {str(k): int(v) for k, v in
                     zip(*np.unique(e_bins, return_counts=True))},
        },
        "oracle_argmin_err": {
            "min": int(o_bins.min()), "max": int(o_bins.max()),
            "median": float(np.median(o_bins)),
            "pct_le4": float(100.0 * np.mean(o_bins <= 4)),
            "pct_le8": float(100.0 * np.mean(o_bins <= 8)),
            "hist": {str(k): int(v) for k, v in
                     zip(*np.unique(o_bins, return_counts=True))},
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "n_sessions": len(rows),
        "n_participants": int(len({r["pid"] for r in rows})),
        "fs": loader.FRAME_RATE_HZ,
        "rules": results,
        "spearman_score_vs_neg_err": corr,
        "deciles": deciles,
        "by_stratum": by_stratum,
        "bin_index": bin_hist,
    }
    (args.out_dir / "bin_oracle_diag.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    # ---- markdown -------------------------------------------------------
    md = ["# Range-bin selection: oracle vs unsupervised scores\n",
          f"- sessions: {len(rows)}   participants: {payload['n_participants']}"
          f"   fs: {loader.FRAME_RATE_HZ} Hz",
          "- HR is the classical FFT-peak estimate on each bin's unwrapped,"
          " band-passed phase (paper Fig.5 baseline).",
          "- ECG HR is used only to *measure* error, never to pick a bin.\n",
          "## Leaderboard (MAE, BPM)\n",
          "`%<3-harm` counts a hit if k*estimate matches ECG for k in {0.5, 1, 2}"
          " — it separates 'wrong bin' from 'right bin, wrong harmonic'.\n",
          "| rule | MAE | median | %<3 | %<5 | %<3-harm | Spearman(score,-err) |",
          "|------|----:|-------:|----:|----:|---------:|----------------------:|"]
    for key in ["oracle_min_err", "oracle_chance_uniform", "max_energy",
                "first_valid_bin", "pc1_trace", "mean_bp_trace",
                *sorted(k for k in results if k.startswith("top1::"))]:
        m = results[key]
        sc = key.split("::")[1] if "::" in key else ""
        c = corr.get(sc, float("nan"))
        md.append(f"| {key} | {m['mae']:.2f} | {m['median']:.2f} | "
                  f"{m['pct_lt3']:.1f} | {m['pct_lt5']:.1f} | "
                  f"{m.get('pct_harm', float('nan')):.1f} | {c:+.3f} |")
    md.append("")
    md.append("### Where the max-energy rule points\n")
    md.append(f"- probe `single_bin` argmax bin ((E|z|)^2): median "
              f"{bin_hist['probe_argmax']['median']:.0f}, range "
              f"[{bin_hist['probe_argmax']['min']}, {bin_hist['probe_argmax']['max']}], "
              f"{bin_hist['probe_argmax']['pct_le4']:.1f}% <= bin 4, "
              f"{bin_hist['probe_argmax']['pct_le8']:.1f}% <= bin 8")
    md.append(f"- max-energy argmax bin: median {bin_hist['max_energy_argmax']['median']:.0f}, "
              f"range [{bin_hist['max_energy_argmax']['min']}, "
              f"{bin_hist['max_energy_argmax']['max']}], "
              f"{bin_hist['max_energy_argmax']['pct_le4']:.1f}% <= bin 4, "
              f"{bin_hist['max_energy_argmax']['pct_le8']:.1f}% <= bin 8")
    md.append(f"- oracle argmin-err bin: median {bin_hist['oracle_argmin_err']['median']:.0f}, "
              f"range [{bin_hist['oracle_argmin_err']['min']}, "
              f"{bin_hist['oracle_argmin_err']['max']}], "
              f"{bin_hist['oracle_argmin_err']['pct_le4']:.1f}% <= bin 4, "
              f"{bin_hist['oracle_argmin_err']['pct_le8']:.1f}% <= bin 8")
    md.append("")
    md.append("## By stratum (MAE, BPM)\n")
    md.append("| stratum | oracle | max_energy | pc1_trace | mean_bp_trace | " +
              " | ".join(score_names) + " |")
    md.append("|---|---:|---:|---:|---:|" + "---:|" * len(score_names))
    for st in strata:
        row = by_stratum[st]
        md.append(f"| {st} | {row['oracle']:.2f} | {row['max_energy']:.2f} | "
                  f"{row['pc1_trace']:.2f} | {row['mean_bp_trace']:.2f} | " +
                  " | ".join(f"{row[f'top1::{n}']:.2f}" for n in score_names) + " |")
    md.append("")
    md.append("## Best decile separation (top-score decile vs all bins)\n")
    md.append("| score | decile-1 MAE | decile-1 %<3 | decile-10 MAE |")
    md.append("|---|---:|---:|---:|")
    for n in score_names:
        d = deciles[n]
        md.append(f"| {n} | {d[0]['mae']:.2f} | {d[0]['pct_lt3']:.1f} | "
                  f"{d[9]['mae']:.2f} |")
    md.append("")
    (args.out_dir / "bin_oracle_diag.md").write_text("\n".join(md), encoding="utf-8")

    print("\n[result] oracle=%.2f  max_energy=%.2f" %
          (results["oracle_min_err"]["mae"], results["max_energy"]["mae"]), flush=True)
    for key in sorted(k for k in results if k.startswith("top1::")):
        print(f"[result] {key:28s} MAE={results[key]['mae']:.2f} "
              f"(rho={corr.get(key.split('::')[1], float('nan')):+.3f})", flush=True)
    print(f"\nDone. Artifacts in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
