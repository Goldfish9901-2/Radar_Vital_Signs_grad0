"""CPU ADC range/ROI/phase audit; labels never enter target selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import resample_poly

from src.data.loaders.export_all_datasets import (
    _build_rda_cube_from_frame, _range_window_indices, _select_even_indices,
    convert_adc_cube_to_rda,
)
from src.features.edacm import edacm_phase, select_target_bin_indices, zscore_1d
from src.features.hr_adavmd import hr_adavmd_decompose
from src.features.representations import frequency_features

VARIANTS = {
    "legacy_mean": ("legacy", "chirp_mean", "none", "legacy"),
    "full_mean": ("full_fft_crop", "chirp_mean", "none", "legacy"),
    "legacy_none": ("legacy", "none", "none", "legacy"),
    "full_none": ("full_fft_crop", "none", "none", "legacy"),
    "full_none_dc": ("full_fft_crop", "none", "mean", "legacy"),
    "full_mean_dc": ("full_fft_crop", "chirp_mean", "mean", "legacy"),
    "native_none_dc": ("full_fft_crop", "none", "mean", "native"),
    "native_mean_dc": ("full_fft_crop", "chirp_mean", "mean", "native"),
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def distance_axis(n, calibration):
    if calibration is None:
        return None
    fs = float(calibration["sample_rate_hz"])
    slope = float(calibration["chirp_slope_hz_per_s"])
    if not np.isfinite(fs) or not np.isfinite(slope) or min(fs, slope) <= 0:
        raise ValueError("Calibration frequency and slope must be positive and finite")
    if not calibration.get("source"):
        raise ValueError("Calibration must identify its acquisition source")
    return np.arange(n//2) * (299792458.0 * fs / (2*slope*n))


def range_profiles(adc):
    """Native, unwindowed Range FFT power, including every frame/RX/chirp."""
    n = adc.shape[-1]
    total = {"none": np.zeros(n//2), "mean": np.zeros(n//2)}
    denominator = adc.shape[0]*adc.shape[1]*adc.shape[2]
    for start in range(0, len(adc), 64):
        chunk = np.asarray(adc[start:start+64], dtype=np.float64)
        for dc in total:
            x = chunk if dc == "none" else chunk-chunk.mean(axis=-1, keepdims=True)
            spectrum = np.fft.fft(x, axis=-1)[..., :n//2]
            total[dc] += (np.abs(spectrum)**2).sum(axis=(0, 1, 2))
    return {k: v/denominator for k,v in total.items()}


def candidate_profile(adc, options):
    sampling = options["sampling_mode"]
    chirps = _select_even_indices(adc.shape[2], min(adc.shape[2], 64)) if sampling == "legacy" else np.arange(adc.shape[2])
    samples = _select_even_indices(adc.shape[3], min(adc.shape[3], 256)) if sampling == "legacy" else np.arange(adc.shape[3])
    amplitude = np.zeros(max(1, len(samples)//2))
    for frame in adc:
        cube = _build_rda_cube_from_frame(frame, chirps, samples, 8, 16,
                                         options["doppler_mode"], options["clutter_mode"],
                                         options["fast_time_dc"])
        amplitude += np.abs(cube).mean(axis=(0, 1))
    return amplitude/len(adc)


def spectral_metrics(phase, fs=20.):
    y = np.asarray(phase, dtype=np.float64)
    y = y-y.mean()
    power = np.abs(np.fft.rfft(y*np.hanning(len(y))))**2
    frequencies = np.fft.rfftfreq(len(y), 1/fs)
    mask = (frequencies >= .75) & (frequencies <= 2.5)
    energy = float(power[mask].sum())
    total = float(power[1:].sum())
    peak = int(np.flatnonzero(mask)[np.argmax(power[mask])])
    valid = np.std(y) > 1e-6 and energy > 1e-12
    return {"phase_variance": float(np.var(y)), "hr_band_energy": energy,
            "hr_band_ratio": energy/(total+1e-30),
            "phase_increment_std": float(np.std(np.diff(y))),
            "fft_hr_bpm": float(frequencies[peak]*60) if valid else None}


def phase_diagnostics(window):
    selected, weights, _ = select_target_bin_indices(window, 3)
    traces = [edacm_phase(window[:, d, a, r]) for d,a,r in selected]
    phase = np.sum(np.stack(traces)*weights[:, None], axis=0)
    phase = np.asarray(phase, dtype=np.float32)
    unwrapped = np.unwrap(np.angle(window[:, *selected[0]])).astype(np.float64)
    t = np.linspace(-1, 1, len(unwrapped))
    unwrapped -= np.polyval(np.polyfit(t, unwrapped, 1), t)
    metrics = spectral_metrics(phase)
    unwrap_metrics = spectral_metrics(unwrapped)
    modes, _ = hr_adavmd_decompose(zscore_1d(phase))
    freq, _, _ = frequency_features(modes)
    metrics.update({"unwrap_phase_variance": unwrap_metrics["phase_variance"],
                    "unwrap_fft_hr_bpm": unwrap_metrics["fft_hr_bpm"],
                    "x_time_std": float(modes.std()), "x_freq_std": float(freq.std()),
                    "feature_nonzero": bool(np.any(np.abs(modes) > 1e-6) and np.any(np.abs(freq) > 1e-6)),
                    "selected_range_bins_in_roi": selected[:, 2].tolist()})
    normalized = zscore_1d(phase)
    psd = np.log1p(np.abs(np.fft.rfft(normalized*np.hanning(len(phase))))**2)
    return metrics, psd


def summarize(rows):
    valid = [r for r in rows if r["label_bpm"] is not None and r["fft_hr_bpm"] is not None]
    return {"windows": len(rows), "nonzero_feature_fraction": float(np.mean([r["feature_nonzero"] for r in rows])),
            "phase_variance_median": float(np.median([r["phase_variance"] for r in rows])),
            "hr_band_energy_median": float(np.median([r["hr_band_energy"] for r in rows])),
            "fft_valid_windows": len(valid),
            "fft_mae_bpm": float(np.mean([abs(r["fft_hr_bpm"]-r["label_bpm"]) for r in valid])) if valid else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    metric_axis = distance_axis(512, calibration)
    raw_files = sorted(p for p in args.input_root.rglob("radar_raw_data.npy")
                       if re.search(r"participant_[1-8]", str(p), re.I))
    if len(raw_files) != 32:
        raise ValueError(f"Expected 32 short recordings, found {len(raw_files)}")
    save_json(out / "protocol.json", {"variants": VARIANTS, "training": "none; optional grouped Ridge probes only on Kaggle",
        "window_size": 256, "stride": 128, "source_frame_rate_hz": 30, "analysis_frame_rate_hz": 20,
        "frame_rate_source": "candidate original HeartTimeMixer publication; acquisition JSON not recovered",
        "calibration": calibration, "physical_roi_status": "available" if calibration else "skipped: missing acquisition slope",
        "nonDC_first_bin": 2, "physical_roi_half_width_m": .15,
        "reference_alignment": "shared start assumed; labels used only after ROI selection",
        "sampling_note": "legacy grid is nonuniform: no metric distance axis; do not transfer integer ROI to native"})
    all_rows, probes, sample_summaries = [], {}, []
    for raw in raw_files:
        pid = int(re.search(r"participant_(\d+)", str(raw), re.I).group(1))
        distance = raw.parent.name
        tag = f"p{pid:02d}_{distance}_short"
        parent = next(p for p in raw.parents if p.name.lower() == f"participant_{pid}")
        ref_file = parent / "HR_Ref_Data" / f"Participant{pid}" / distance / "HR_ref.csv"
        ref = np.genfromtxt(ref_file, delimiter=",", skip_header=1)
        if ref.ndim != 2 or ref.shape[1] != 3 or not np.all(np.diff(ref[:, 0]) > 0):
            raise ValueError(f"Invalid reference format/timing: {ref_file}")
        adc = np.load(raw, mmap_mode="r")
        if adc.shape != (18000, 3, 16, 512):
            raise ValueError(f"Unexpected ADC shape: {adc.shape}")
        profiles = range_profiles(adc)
        np.savez_compressed(out / f"{tag}_native_range_profiles.npz", **profiles)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        axis = np.arange(256) if metric_axis is None else metric_axis
        for method, power in profiles.items():
            ax.plot(axis, 10*np.log10(np.maximum(power, 1e-30)), label=f"fast-time DC: {method}")
        ax.set(xlabel="Native range FFT bin" if metric_axis is None else "Calibrated range (m)", ylabel="Mean power (dB, ADC units)", title=tag)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{tag}_range.png", dpi=150)
        plt.close(fig)
        baseline_center = None
        record_summary = {"sample": tag, "participant_id": pid, "distance_m": float(distance.rstrip("m")),
            "native_range_peaks": {k: int(np.argmax(v)) for k,v in profiles.items()},
            "native_dc_power_fraction": {k: float(v[0]/(v.sum()+1e-30)) for k,v in profiles.items()}, "candidates": {}}
        for name, (doppler, clutter, dc, sampling) in VARIANTS.items():
            options = dict(doppler_mode=doppler, clutter_mode=clutter, fast_time_dc=dc, sampling_mode=sampling)
            profile = candidate_profile(adc, options)
            if baseline_center is None:
                baseline_center = int(np.argmax(profile))
            centers = {"auto_energy": int(np.argmax(profile)), "auto_nonDC": int(2+np.argmax(profile[2:]))}
            if sampling == "legacy":
                centers["fixed_legacy"] = baseline_center
            if sampling == "native" and metric_axis is not None:
                allowed = np.flatnonzero(np.abs(metric_axis-record_summary["distance_m"]) <= .15)
                if len(allowed):
                    centers["physical_prior"] = int(allowed[np.argmax(profile[allowed])])
            record_summary["candidates"][name] = {"mean_amplitude_profile": profile.tolist(), "roi_centers": centers}
            # Reuse exactly identical cubes when two ROI policies choose the same center.
            for center in sorted(set(centers.values())):
                cube, bins, _ = convert_adc_cube_to_rda(adc, range_center_bin=center, **options)
                cube = resample_poly(cube, 2, 3, axis=0).astype(np.complex64)
                roi_policies = [p for p,c in centers.items() if c == center]
                roi_bins = _range_window_indices(len(profile), 8, center)
                roi_mask = roi_bins >= 2
                for start in range(0, len(cube)-255, 128):
                    # NonDC/physical policies gate the cells themselves, not just the crop center.
                    # Legacy/fixed policies preserve the historical full eight-bin crop.
                    for policy in roi_policies:
                        gated = policy in {"auto_nonDC", "physical_prior"}
                        active = roi_mask.copy() if gated else np.ones(len(bins), dtype=bool)
                        if policy == "physical_prior" and metric_axis is not None:
                            active &= np.abs(metric_axis[bins]-record_summary["distance_m"]) <= .15
                        window = cube[start:start+256, :, :, active]
                        metrics, vector = phase_diagnostics(window)
                        times = np.arange(start, start+256)/20
                        labels = np.interp(times, ref[:, 0], ref[:, 1], left=np.nan, right=np.nan)
                        label = float(np.nanmean(labels)) if np.isfinite(labels).mean() >= .8 else None
                        row = {"sample": tag, "participant_id": pid, "distance_m": record_summary["distance_m"],
                               "variant": name, "roi_policy": policy, "window_start": start,
                               "range_center_bin": center, "label_bpm": label, **metrics,
                               "selected_original_range_bins": bins[active][np.array(metrics["selected_range_bins_in_roi"])].tolist()}
                        all_rows.append(row)
                        probes.setdefault(f"{name}/{policy}", []).append((pid, label, vector))
                del cube
            print(f"{tag} {name} complete", flush=True)
        sample_summaries.append(record_summary)
        save_json(out / "sample_summaries.json", sample_summaries)
    with (out / "window_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    grouped = {}
    for row in all_rows:
        grouped.setdefault(f'{row["variant"]}/{row["roi_policy"]}', []).append(row)
    save_json(out / "summary.json", {k: summarize(v) for k,v in grouped.items()})
    by_subject = {}
    by_sample = {}
    for key, rows in grouped.items():
        by_subject[key] = {str(pid): summarize([r for r in rows if r["participant_id"] == pid])
                           for pid in sorted({r["participant_id"] for r in rows})}
        by_sample[key] = {tag: summarize([r for r in rows if r["sample"] == tag])
                          for tag in sorted({r["sample"] for r in rows})}
    save_json(out / "by_subject.json", by_subject)
    save_json(out / "by_sample.json", by_sample)
    # Save probe inputs; fitting is restricted to the remote Kaggle runner.
    for key, records in probes.items():
        np.savez_compressed(out / (key.replace("/", "__")+"_probe.npz"),
            subjects=np.array([r[0] for r in records]),
            y=np.array([np.nan if r[1] is None else r[1] for r in records]),
            x=np.stack([r[2] for r in records]))
    report = ["# BGT60 ADC distance / target / phase diagnostic", "",
        f"All {len(sample_summaries)} short recordings and all overlapping windows audited. No neural training.", "",
        "Physical ROI: " + ("calibrated from supplied acquisition parameters" if calibration else "not evaluated: acquisition slope unavailable"), "",
        "NonDC gating excludes bins 0 and 1; this is a diagnostic hypothesis, not validated chest localization.", "",
        "| Candidate / ROI | Windows | Nonzero features | Median phase variance | FFT valid | FFT MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, rows in grouped.items():
        s = summarize(rows)
        mae = "NA" if s["fft_mae_bpm"] is None else f'{s["fft_mae_bpm"]:.3f}'
        report.append(f'| {key} | {s["windows"]} | {s["nonzero_feature_fraction"]:.3f} | {s["phase_variance_median"]:.6g} | {s["fft_valid_windows"]} | {mae} |')
    report += ["", "FFT MAE uses valid phase windows only: compare coverage alongside error. HR-band energy alone does not establish heartbeat origin. Labels do not select bins. Reference start synchronization remains an assumption.", ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
