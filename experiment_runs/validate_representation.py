"""Sanity-check a radar representation BEFORE spending GPU time on it.

Philosophy (post head-line analysis): *the model is fine, the representation is
the bottleneck.* So before training 80 epochs on a new signal-processing method,
run this and read the report. It checks, per representation:

  1. shape consistency   -- every window emits the same (C, T) / (C, F)
  2. NaN / Inf           -- fraction of windows with bad values, per channel
  3. channel collapse     -- per-channel variance (a dead channel = no info)
  4. spectrum health      -- energy inside the HR band + spectral flatness
  5. HR correlation       -- does any channel even weakly track the HR label?

Usage
-----
    # validate the proposed representation directly from raw RDA exports
    python experiment_runs/validate_representation.py --representation proposed

    # validate an already-exported representation from on-disk windows
    python experiment_runs/validate_representation.py --representation new_method --source export

Outputs a terminal summary and a self-contained HTML report
(``experiment_runs/benchmark_v3/representation_report_<rep>.html``).
"""

from __future__ import annotations

import argparse
import html
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.features.representations import radar_to_feature_bundle  # noqa: E402

DATASETS = ["FTU", "BGT60TR13C", "PhysDrive"]
HR_BAND_HZ = (0.5, 3.667)  # ~30..220 bpm


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _iter_raw(dataset: str, raw_dir: Path):
    d = raw_dir / dataset / "samples"
    files = sorted(d.glob("*.npz")) if d.is_dir() else []
    if not files:  # fall back to a recursive search under raw_dir
        files = sorted(raw_dir.glob(f"**/{dataset}/samples/*.npz"))
    for f in files:
        data = np.load(f, allow_pickle=False)
        if "radar" not in data:
            continue
        radar = data["radar"]
        hr = float(np.mean(np.asarray(data["heart_rate"], dtype=np.float32))) if "heart_rate" in data else np.nan
        yield radar, hr


def _iter_export(dataset: str, export_dir: Path):
    for split in ("train", "val", "test"):
        d = export_dir / "windows" / dataset / split
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npz")):
            data = np.load(f, allow_pickle=False)
            if "x_time" not in data:
                continue
            x_time = data["x_time"].astype(np.float32)
            x_freq = data["x_freq"].astype(np.float32) if "x_freq" in data else None
            hr = float(np.asarray(data["label_heart_rate"])) if "label_heart_rate" in data else np.nan
            yield x_time, x_freq, hr


def collect(representation: str, dataset: str, source: str, raw_dir: Path, export_dir: Path, max_windows: int):
    """Yield (x_time, x_freq, hr) triples for one dataset (capped at max_windows)."""
    if source == "raw":
        gen = _iter_raw(dataset, raw_dir)
    else:
        gen = _iter_export(dataset, export_dir)
    for item in itertools.islice(gen, max_windows):
        if source == "raw":
            radar, hr = item
            rep = radar_to_feature_bundle(radar, representation)
            xt, xf = rep["x_time"], rep["x_freq"]
            if xt is None:
                yield None, None, hr  # representation does not emit a dual bundle
                continue
            yield np.asarray(xt, dtype=np.float32), (np.asarray(xf, dtype=np.float32) if xf is not None else None), hr
        else:
            xt, xf, hr = item
            yield xt, xf, hr


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def _spectrum(x_time: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum (over windows, channels) + rfftfreq."""
    n = x_time.shape[-1]
    spec = np.fft.rfft(x_time, axis=-1)
    power = np.abs(spec) ** 2
    power = power.reshape(-1, power.shape[-1]).mean(axis=0)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return power, freqs


def check_dataset(representation: str, dataset: str, source: str, raw_dir: Path, export_dir: Path,
                  max_windows: int, sr: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"dataset": dataset, "n_windows": 0, "dual": True,
                           "checks": [], "shapes": set(), "freq_shapes": set()}
    xt_list, xf_list, hrs = [], [], []
    for xt, xf, hr in collect(representation, dataset, source, raw_dir, export_dir, max_windows):
        if xt is None:
            out["dual"] = False
            continue
        out["shapes"].add(xt.shape)
        if xf is not None:
            out["freq_shapes"].add(xf.shape)
        xt_list.append(xt)
        xf_list.append(xf)
        hrs.append(hr)

    out["n_windows"] = len(xt_list)
    if not xt_list:
        out["checks"].append(("data", "FAIL", "no dual-bundle windows collected (representation may not emit x_time/x_freq, or no samples found)"))
        return out

    X = np.stack(xt_list)              # (W, C, T)
    F = np.stack([x for x in xf_list if x is not None]) if xf_list and xf_list[0] is not None else None
    hrs = np.asarray(hrs, dtype=np.float32)
    W, C, T = X.shape

    # 1. shape consistency
    if len(out["shapes"]) == 1:
        out["checks"].append(("shape", "PASS", f"all {W} windows (C={C}, T={T})"))
    else:
        out["checks"].append(("shape", "FAIL", f"inconsistent x_time shapes: {sorted(out['shapes'])}"))

    # 2. NaN / Inf
    bad = int(np.isnan(X).any(axis=(1, 2)).sum() + np.isinf(X).any(axis=(1, 2)).sum())
    if F is not None:
        bad += int(np.isnan(F).any(axis=(1, 2)).sum() + np.isinf(F).any(axis=(1, 2)).sum())
    frac = bad / W
    if frac == 0:
        out["checks"].append(("nan_inf", "PASS", "no NaN/Inf in any window"))
    elif frac < 0.05:
        out["checks"].append(("nan_inf", "WARN", f"{bad}/{W} windows ({frac:.1%}) contain NaN/Inf"))
    else:
        out["checks"].append(("nan_inf", "FAIL", f"{bad}/{W} windows ({frac:.1%}) contain NaN/Inf"))

    # 3. per-channel variance (collapse detection)
    var = X.var(axis=2).mean(axis=0)  # (C,)
    min_var, mean_var = float(var.min()), float(var.mean())
    if min_var < 1e-6:
        out["checks"].append(("channel_variance", "FAIL", f"collapsed channel(s): min var={min_var:.2e}, mean={mean_var:.2e}"))
    elif min_var < 1e-3:
        out["checks"].append(("channel_variance", "WARN", f"low channel variance: min={min_var:.2e}, mean={mean_var:.2e}"))
    else:
        out["checks"].append(("channel_variance", "PASS", f"min={min_var:.2e}, mean={mean_var:.2e} (per-channel)"))

    # 4. spectrum health
    power, freqs = _spectrum(X, sr)
    band = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    band_energy = float(power[band].sum() / (power.sum() + 1e-12))
    flat = float(np.exp(np.mean(np.log(power + 1e-12))) / (power.mean() + 1e-12))
    spec_note = f"HR-band energy={band_energy:.1%}, spectral flatness={flat:.3f}"
    if band_energy < 0.03 or flat > 0.6:
        out["checks"].append(("spectrum", "WARN", spec_note + " (little energy in HR band / very flat)"))
    else:
        out["checks"].append(("spectrum", "PASS", spec_note))

    # 5. HR correlation (per-channel mean vs label; plus a cheap linear probe R^2)
    chan_mean = X.mean(axis=2)  # (W, C)
    if np.isfinite(hrs).any():
        finite = np.isfinite(hrs)
        y = hrs[finite]
        Xc = chan_mean[finite]
        rs = [float(np.corrcoef(Xc[:, c], y)[0, 1]) if Xc[:, c].std() > 0 else 0.0 for c in range(C)]
        best = int(np.argmax(np.abs(rs)))
        max_r = float(rs[best])
        # linear probe R^2 over per-channel-mean features (+bias)
        A = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ beta
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        corr_note = f"max|r|={max_r:.3f} (ch {best}), linear-probe R^2={r2:.3f}"
        if max_r < 0.05:
            out["checks"].append(("hr_correlation", "WARN", corr_note + " (no channel tracks HR — likely little usable signal)"))
        else:
            out["checks"].append(("hr_correlation", "PASS", corr_note))
        out["hr"] = {"max_abs_r": max_r, "best_channel": best, "probe_r2": r2}
    else:
        out["checks"].append(("hr_correlation", "WARN", "no finite HR labels available"))

    out["C"], out["T"] = C, T
    return out


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
COLOR = {"PASS": "#1a7f37", "WARN": "#9a6700", "FAIL": "#cf222e"}


def _row(name: str, status: str, detail: str) -> str:
    return (f"<tr><td>{html.escape(name)}</td>"
            f"<td style='color:{COLOR[status]};font-weight:700'>{status}</td>"
            f"<td>{html.escape(detail)}</td></tr>")


def render_html(representation: str, results: List[Dict[str, Any]], src: str) -> str:
    body = [f"<h2>Representation: <code>{html.escape(representation)}</code> (source: {html.escape(src)})</h2>"]
    for r in results:
        body.append(f"<h3>Dataset {html.escape(r['dataset'])} — {r['n_windows']} windows"
                    + ("" if r.get("dual", True) else " <span style='color:#cf222e'>(no dual bundle)</span>") + "</h3>")
        body.append("<table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse'>"
                    "<tr><th>check</th><th>status</th><th>detail</th></tr>")
        for name, status, detail in r["checks"]:
            body.append(_row(name, status, detail))
        body.append("</table>")
        if "hr" in r:
            hr = r["hr"]
            body.append(f"<p>HR correlation: max|r|={hr['max_abs_r']:.3f} (channel {hr['best_channel']}), "
                        f"linear-probe R²={hr['probe_r2']:.3f}</p>")
    # legend
    body.append("<p style='margin-top:24px;color:#57606a'>PASS = healthy · WARN = suspicious, "
                "investigate before training · FAIL = do not train on this. Spectral flatness near 1.0 "
                "means a flat/white spectrum (no structure); HR-band energy is the fraction of power in "
                "~0.5–3.7 Hz.</p>")
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>representation validation</title></head><body style='font-family:sans-serif;max-width:900px;margin:2em'>"
            + "".join(body) + "</body></html>")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--representation", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--source", choices=["raw", "export"], default="raw",
                    help="raw = run radar_to_feature_bundle on raw RDA exports (exercises the method); "
                         "export = read already-exported windows from disk")
    ap.add_argument("--raw-dir", type=Path, default=Path("/home/agent-dev-radar/radar/work/exports"),
                    help="raw RDA exports (contain <dataset>/samples/*.npz with a 'radar' array)")
    ap.add_argument("--export-dir", type=Path, default=Path("/home/agent-dev-radar/radar/work/run/upstream/training_exports"),
                    help="processed windows (contain windows/<dataset>/<split>/*.npz with x_time/x_freq)")
    ap.add_argument("--max-windows", type=int, default=120, help="cap windows per dataset (CPU time)")
    ap.add_argument("--sample-rate", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=None, help="HTML report path")
    args = ap.parse_args()

    out_path = args.out or (ROOT / "experiment_runs" / "benchmark_v3" / f"representation_report_{args.representation}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for ds in args.datasets:
        print(f"[validate] {args.representation} / {ds} (source={args.source}) ...", flush=True)
        r = check_dataset(args.representation, ds, args.source, args.raw_dir, args.export_dir,
                          args.max_windows, args.sample_rate)
        results.append(r)
        status = {c[0]: c[1] for c in r["checks"]}
        print(f"  n={r['n_windows']} dual={r.get('dual')} -> "
              + ", ".join(f"{k}={v}" for k, v in status.items()), flush=True)

    out_path.write_text(render_html(args.representation, results, args.source), encoding="utf-8")
    print(f"\n[written] {out_path}")

    # terminal verdict
    fails = [c for r in results for c in r["checks"] if c[1] == "FAIL"]
    warns = [c for r in results for c in r["checks"] if c[1] == "WARN"]
    print(f"\nVerdict for '{args.representation}': "
          f"{len(fails)} FAIL, {len(warns)} WARN across {len(results)} datasets.")
    if fails:
        print("DO NOT train on this representation until FAILs are resolved.")
    elif warns:
        print("Usable, but review WARNs (especially hr_correlation) before committing GPU time.")
    else:
        print("Clean.")


if __name__ == "__main__":
    main()
