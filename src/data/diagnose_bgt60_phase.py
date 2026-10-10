"""Label-independent native range calibration and direct phase audit (no VMD)."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import detrend, find_peaks
from src.data.diagnose_bgt60_adc import range_profiles, save_json, spectral_metrics
from src.features.edacm import edacm_phase


def peak_candidates(power):
    peaks, _ = find_peaks(np.log(np.maximum(power, 1e-30)), prominence=.5)
    peaks = peaks[peaks >= 2]
    return sorted(peaks.tolist(), key=lambda p: power[p], reverse=True)[:8]


def alignment_audit(ref, frames, fs):
    t = ref[:, 0]
    return {"reference_first_time": float(t[0]), "reference_last_time": float(t[-1]),
            "reference_rows": len(t), "reference_dt_median": float(np.median(np.diff(t))),
            "reference_strictly_increasing": bool(np.all(np.diff(t) > 0)),
            "radar_last_time_assumed_s": (frames-1)/fs,
            "radar_frame_rate_assumed_hz": fs,
            "start_offset_verified": False,
            "warning": "No radar timestamps or acquisition start metadata: shared start is unverified. No label-based offset optimization."}


def direct_traces(z):
    return {"complex_real": detrend(z.real), "complex_imag": detrend(z.imag),
            "unwrap": detrend(np.unwrap(np.angle(z))), "edacm": edacm_phase(z)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-records", type=int)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in args.input_root.rglob("radar_raw_data.npy")
                   if re.search(r"participant_[1-8](?:[/\\])", str(f), re.I))
    if not files:
        raise ValueError("No BGT60 records found")
    if args.max_records:
        files = files[:args.max_records]
    summaries, rows = [], []
    for raw in files:
        pid = int(re.search(r"participant_(\d+)", str(raw), re.I).group(1))
        distance = float(raw.parent.name.rstrip("m"))
        tag = f"p{pid:02d}_{distance:.1f}m"
        parent = next(a for a in raw.parents if a.name.lower() == f"participant_{pid}")
        reference = parent / "HR_Ref_Data" / f"Participant{pid}" / raw.parent.name / "HR_ref.csv"
        with reference.open(encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream))
        ref = np.genfromtxt(reference, delimiter=",", skip_header=1)
        if ref.ndim != 2 or ref.shape[1] != 3 or not np.isfinite(ref).all() or not np.all(np.diff(ref[:, 0]) > 0):
            raise ValueError(f"Invalid reference: {reference}")
        adc = np.load(raw, mmap_mode="r")
        power = range_profiles(adc)["mean"]
        candidates = peak_candidates(power)
        # Preregistered strongest local peak above bin 1, fixed for entire record.
        center = candidates[0] if candidates else int(2+np.argmax(power[2:]))
        # RX selected by mean power only; identical ROI/RX across phase methods.
        traces = np.empty((len(adc), adc.shape[1], 3), dtype=np.complex128)
        for start in range(0, len(adc), 64):
            x = np.asarray(adc[start:start+64], dtype=np.float64)
            x -= x.mean(axis=-1, keepdims=True)
            r = np.fft.fft(x, axis=-1)[..., center]
            d = np.fft.fft(r, axis=-1)
            traces[start:start+len(x), :, 0] = r[:, :, 0]
            traces[start:start+len(x), :, 1] = d[:, :, 0]/r.shape[-1]
            traces[start:start+len(x), :, 2] = d[:, :, 1]/r.shape[-1]
        rx = int(np.argmax(np.mean(np.abs(traces[:, :, 1])**2, axis=0)))
        audit = alignment_audit(ref, len(adc), 30.)
        audit["reference_header"] = header
        summaries.append({"sample": tag, "participant_id": pid, "known_distance_m": distance,
                          "native_peak_candidates": candidates, "selected_native_bin": center,
                          "rx": rx, "time_audit": audit})
        for coefficient, index in [("first_chirp", 0), ("zero_doppler", 1), ("adjacent_doppler", 2)]:
            z = traces[:, rx, index]
            for start in range(0, len(z)-383, 192):
                times = np.arange(start, start+384)/30.
                for method, y in direct_traces(z[start:start+384]).items():
                    metrics = spectral_metrics(y, fs=30.)
                    for offset in [0, -60, 60, -120, 120]:
                        labels = np.interp(times+offset, ref[:, 0], ref[:, 1], left=np.nan, right=np.nan)
                        # Equal complete-window coverage; no clipping or circular wrapping.
                        label = float(labels.mean()) if np.isfinite(labels).all() else None
                        rows.append({"sample": tag, "subject": pid, "coefficient": coefficient,
                                     "method": method, "window_start": start, "label_offset_s": offset,
                                     "label_bpm": label, **metrics})
        save_json(out / "records.json", summaries)
        print(tag, "complete", flush=True)
    # Distance labels alone are allowed; never ECG labels. Report association rather than inventing a slope.
    centers = np.array([s["selected_native_bin"] for s in summaries])
    distances = np.array([s["known_distance_m"] for s in summaries])
    fit = np.polyfit(distances, centers, 1) if len(set(distances)) > 1 else None
    save_json(out / "distance_calibration.json", {
        "original_device_parameters": {"chirp_slope_hz_per_s": None, "verified": False},
        "experimental_peak_regression": None if fit is None else {
            "bin_per_m": float(fit[0]), "intercept_bin": float(fit[1]),
            "residual_rmse_bins": float(np.sqrt(np.mean((centers-np.polyval(fit, distances))**2))),
            "status": "exploratory association only; not used to define physical ROI or infer slope"},
        "selection": "Strongest local native DC-removed range peak above bin 1; fixed per recording; no ECG"})
    with (out / "windows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    results = {}
    bounds = {s['sample']: (s['time_audit']['reference_first_time']+120,
                           min(s['time_audit']['reference_last_time']-120,
                               s['time_audit']['radar_last_time_assumed_s'])) for s in summaries}
    for coefficient in ["first_chirp", "zero_doppler", "adjacent_doppler"]:
        for method in ["complex_real", "complex_imag", "unwrap", "edacm"]:
            for offset in [0, -60, 60, -120, 120]:
                # Matched windows across all offsets for valid comparisons.
                selected = [r for r in rows if r["coefficient"] == coefficient and r["method"] == method
                            and r["label_offset_s"] == offset and bounds[r['sample']][0]*30 <= r["window_start"]
                            and r["window_start"]+384 <= bounds[r['sample']][1]*30
                            and r["label_bpm"] is not None and r["fft_hr_bpm"] is not None]
                folds = []
                for pid in sorted({r["subject"] for r in selected}):
                    sub = [r for r in selected if r["subject"] == pid]
                    pred = np.array([r["fft_hr_bpm"] for r in sub])
                    label = np.array([r["label_bpm"] for r in sub])
                    corr = float(np.corrcoef(pred, label)[0, 1]) if min(pred.std(), label.std()) > .01 else None
                    folds.append({"subject": pid, "windows": len(sub), "mae_bpm": float(np.abs(pred-label).mean()), "pearson_r": corr})
                results[f"{coefficient}/{method}/{offset}"] = {"folds": folds,
                    "macro_mae_bpm": float(np.mean([f["mae_bpm"] for f in folds])) if folds else None}
    save_json(out / "summary.json", results)
    save_json(out / "protocol.json", {"frame_rate_hz": 30, "frame_rate_verified": False,
        "window_frames": 384, "stride_frames": 192, "negative_offsets_s": [-120, -60, 60, 120],
        "offset_selection": "fixed controls, never optimized", "vmd": False, "training": False,
        "interpretation": "Exploratory; fixed energy ROI is not validated chest localization. Shared start unverified."})
    report = ["# BGT60 direct phase and timing audit", "", "No VMD or model training. No label-based ROI or offset selection.",
              "Physical distance calibration and absolute synchronization remain unverified; see records.json and distance_calibration.json.",
              "Negative controls use matched interior windows, no circular wrapping. Slow reference HR may make offsets weak controls.", "",
              "| Coefficient / phase / reference offset seconds | Subject macro MAE |", "| --- | ---: |"]
    for key, result in results.items():
        report.append(f"| {key} | {result['macro_mae_bpm']} |")
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
