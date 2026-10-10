"""Kaggle-only, fixed-ROI ADC frontend ablation with CycleFormer v3-small."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

WORK = Path("/kaggle/working")
VARIANTS = {
    "legacy_mean": ("legacy", "chirp_mean"),
    "full_mean": ("full_fft_crop", "chirp_mean"),
    "legacy_none": ("legacy", "none"),
    "full_none": ("full_fft_crop", "none"),
}


def run(args, cwd=None):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def signature(folder):
    with (folder / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    keys = ("dataset", "sample_tag", "group_key", "split", "window_start", "label_heart_rate")
    result = sorted(tuple(row.get(k, "") for k in keys) for row in rows)
    for split in ("train", "val", "test"):
        assert any(row[3] == split for row in result), f"Empty {split} split"
    return result


def main():
    if not WORK.is_dir() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("This runner is restricted to Kaggle; no local training")
    archives = list(Path("/kaggle/input").rglob("radar_vital_signs_code.zip"))
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(WORK / "code")
        code = WORK / "code" / "Radar_Vital_Signs_grad0"
    else:
        code = next(p.parent for p in Path("/kaggle/input").rglob("validate_adc_rda_synthetic.py")
                    if (p.parent / "src").is_dir())
    os.chdir(code)
    sys.path.insert(0, str(code))
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run([sys.executable, "-m", "uv", "pip", "install", "--system",
         "neurokit2", "h5py", "pyyaml", "pandas", "scipy", "scikit-learn", "matplotlib"])
    run([sys.executable, "validate_adc_rda_synthetic.py"])

    import numpy as np
    import pandas as pd
    from scipy.signal import resample_poly
    import torch
    from src.data.loaders.export_all_datasets import convert_adc_cube_to_rda
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing an unexpected CPU training run")
    # Detect unsupported GPU architectures before lengthy ADC processing.
    torch.ones(4, device="cuda").square().sum().item()

    raw_files = sorted(p for p in Path("/kaggle/input").rglob("radar_raw_data.npy")
                       if re.search(r"participant_[1-8]", str(p), re.I))
    if len(raw_files) != 32:
        raise RuntimeError(f"Expected 8 subjects x 4 distances, found {len(raw_files)} files")
    inputs = []
    for raw in raw_files:
        match = re.search(r"participant_(\d+)", str(raw), re.I)
        pid = int(match.group(1))
        distance = raw.parent.name
        participant_root = next(p for p in raw.parents if p.name.lower() == f"participant_{pid}")
        reference = participant_root / "HR_Ref_Data" / f"Participant{pid}" / distance / "HR_ref.csv"
        if not reference.is_file():
            raise FileNotFoundError(reference)
        inputs.append((pid, distance, raw, reference))
    protocol = {
        "variants": VARIANTS, "sampling_mode": "legacy", "roi": "legacy_mean fixed per sample",
        "raw_frame_rate_hz": 30, "representation_frame_rate_hz": 20,
        "frame_resampling": "complex polyphase 2/3, after RDA, all variants",
        "reference_alignment": "frame index / 30 s, assumes shared acquisition start; no ECG oracle",
        "split_mode": "stable_grouped", "seed": 42, "window_size": 256, "stride": 128,
        "model": "cycleformer_v3_small", "epochs": 80, "patience": 12,
        "subjects": list(range(1, 9)), "long_duration": "excluded",
        "inputs": [dict(participant=pid, distance=d, radar=str(p), reference=str(r))
                   for pid, d, p, r in inputs],
        "production_source_sha256": hashlib.sha256(
            (code / "src/data/loaders/export_all_datasets.py").read_bytes()).hexdigest(),
    }
    write_json(WORK / "protocol.json", protocol)
    centers = {}
    baseline_signature = None
    feature_dirs = {}
    for variant, (doppler, clutter) in VARIANTS.items():
        export = WORK / "exports" / variant
        dataset = export / "BGT60TR13C"
        (dataset / "samples").mkdir(parents=True, exist_ok=True)
        rows = []
        for pid, distance, raw, reference in inputs:
            tag = f"p{pid:02d}_{distance.replace('.', '_')}_short"
            print(f"ADC {variant} {tag}", flush=True)
            adc = np.load(raw, mmap_mode="r")
            if adc.shape != (18000, 3, 16, 512):
                raise ValueError(f"Unexpected ADC shape {adc.shape} in {raw}")
            ref = pd.read_csv(reference)
            time = np.asarray(ref["Time"], dtype=float)
            if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
                raise ValueError(f"Invalid reference timestamps in {reference}")
            if abs(time[0]) > 1 or time[-1] < 590 or time[-1] > 610:
                raise ValueError(f"Reference timing differs from assumed 10-minute 30Hz capture: {reference}")
            cube, bins, center = convert_adc_cube_to_rda(
                adc, doppler_mode=doppler, clutter_mode=clutter,
                sampling_mode="legacy", range_center_bin=centers.get(tag))
            centers[tag] = center
            cube = resample_poly(cube, 2, 3, axis=0).astype(np.complex64)
            frame_time = np.arange(len(cube), dtype=np.float32) / 20
            hr = np.interp(frame_time, time, ref["HR (bpm)"], left=np.nan, right=np.nan).astype(np.float32)
            rr = np.interp(frame_time, time, ref["RR (bpm)"], left=np.nan, right=np.nan).astype(np.float32)
            out = dataset / "samples" / f"{tag}.npz"
            np.savez(out, radar=cube, time=frame_time, heart_rate=hr, respiration_rate=rr)
            write_json(dataset / "meta" / f"{tag}.json", {
                "dataset": "BGT60TR13C", "participant_id": pid, "sample_tag": tag,
                "adc_doppler_mode": doppler, "adc_clutter_mode": clutter,
                "range_center_bin": center, "range_selected_bins": bins.tolist(),
                "frame_rate_hz": 20, "source_frame_rate_hz": 30,
            })
            rows.append(dict(sample_tag=tag, participant_id=pid, distance=distance,
                             measurement_type="short", npz_path=str(out), status="ok"))
            del adc, cube
        write_json(WORK / "fixed_range_centers.json", centers)
        with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        features = WORK / "training_exports" / variant
        run([sys.executable, "-m", "src.data.build_training_dataset",
             "--exports-dir", export, "--output-dir", features,
             "--datasets", "BGT60TR13C", "--representation", "proposed",
             "--window-size", "256", "--stride", "128", "--drop-rda",
             "--split-mode", "stable_grouped", "--seed", "42"])
        current = signature(features)
        if baseline_signature is None:
            baseline_signature = current
            write_json(WORK / "frozen_windows.json", current)
        elif current != baseline_signature:
            raise RuntimeError("Window/label/split mismatch across frontends")
        feature_dirs[variant] = features
        # Remove only this runner's own large temporary exports after feature extraction.
        target = export.resolve()
        if target.parent != (WORK / "exports").resolve():
            raise RuntimeError("Export cleanup path escaped task directory")
        shutil.rmtree(target)

    reports = {}
    for variant, features in feature_dirs.items():
        model = WORK / "models" / variant
        run([sys.executable, "-m", "src.training.train_model", "--model", "cycleformer",
             "--datasets", "BGT60TR13C", "--export-dir", features, "--output-dir", model,
             "--epochs", "80", "--batch-size", "32", "--num-workers", "2",
             "--seed", "42", "--d-model", "32", "--d-ff", "64", "--num-layers", "1",
             "--nhead", "4", "--hidden-channels", "48", "--num-blocks", "4",
             "--kernel-size", "7", "--dropout", "0.15"])
        for mode, checkpoint in (("retrained", model), ("frozen_legacy", WORK / "models/legacy_mean")):
            dest = WORK / "evaluations" / f"{variant}_{mode}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "src.training.evaluate_model", "--model-dir", checkpoint,
                 "--export-dir", features, "--target-datasets", "BGT60TR13C", "--split", "test",
                 "--output-json", dest.with_suffix(".json"), "--dump-predictions", dest.with_suffix(".csv")])
            pred = pd.read_csv(dest.with_suffix(".csv"))
            config = json.loads((checkpoint / "run_config.json").read_text())
            constant = float(config["label_stats"]["mean"])
            reports[f"{variant}_{mode}"] = {
                "prediction_std_bpm": float(pred.pred_bpm.std(ddof=0)),
                "constant_train_mean_bpm": constant,
                "constant_mae_bpm": float(np.abs(pred.label_bpm-constant).mean()),
                "checkpoint_sha256": hashlib.sha256((checkpoint / "best.pt").read_bytes()).hexdigest(),
                "evaluation": json.loads(dest.with_suffix(".json").read_text()),
            }
            write_json(WORK / "comparison.json", reports)
    print("BGT60 frontend ablation finished", flush=True)


if __name__ == "__main__":
    main()
