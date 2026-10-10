"""Analyze downloaded Kaggle metrics without training or loading torch."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/bgt60_frontend_results"
OUTPUT = ROOT / "experiment_runs/bgt60_frontend_2026-10-10"
OUTPUT.mkdir(parents=True, exist_ok=True)
comparison = json.loads((SOURCE / "comparison.json").read_text())
centers = json.loads((SOURCE / "fixed_range_centers.json").read_text())
windows = json.loads((SOURCE / "frozen_windows.json").read_text())
summary = {}
baseline_keys = None
for name, item in comparison.items():
    with (SOURCE / "evaluations" / f"{name}.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: (r["sample_tag"], int(r["window_start"])))
    keys = [(r["sample_tag"], int(r["window_start"]), float(r["label_bpm"])) for r in rows]
    if baseline_keys is None:
        baseline_keys = keys
    assert keys == baseline_keys, "Unpaired evaluation windows or labels"
    y = np.array([float(r["label_bpm"]) for r in rows])
    p = np.array([float(r["pred_bpm"]) for r in rows])
    # 0.01 bpm is a declared reporting threshold, not a physiological validity test.
    near_constant = float(p.std()) < 0.01
    summary[name] = {
        "mae_bpm": float(np.abs(p-y).mean()), "rmse_bpm": float(np.sqrt(np.mean((p-y)**2))),
        "prediction_mean_bpm": float(p.mean()), "prediction_std_bpm": float(p.std()),
        "prediction_span_bpm": float(np.ptp(p)), "label_std_bpm": float(y.std()),
        "raw_pearson_r": item["evaluation"]["overall"]["pearson_r"],
        "reported_pearson_r": None if near_constant else item["evaluation"]["overall"]["pearson_r"],
        "near_constant": near_constant,
        "mean_prediction_as_constant_mae_bpm": float(np.abs(p.mean()-y).mean()),
        "train_mean_constant_mae_bpm": item["constant_mae_bpm"],
        "subjects": sorted(set(r["participant_id"] for r in rows)), "windows": len(rows),
        "checkpoint_sha256": item["checkpoint_sha256"],
    }
frozen_hashes = {v["checkpoint_sha256"] for k,v in summary.items() if k.endswith("frozen_legacy")}
assert len(frozen_hashes) == 1, "Frozen-weight checkpoints differ"
split_subjects = {}
for row in windows:
    split_subjects.setdefault(row[3], set()).add(row[2])
assert not (split_subjects["train"] & split_subjects["test"])
assert not (split_subjects["train"] & split_subjects["val"])
assert not (split_subjects["val"] & split_subjects["test"])
output = {"metrics": summary, "paired_test_windows": True,
          "frozen_checkpoint_identical": True,
          "range_centers": {"samples": len(centers), "unique_centers": sorted(set(centers.values()))},
          "split_subjects": {k: sorted(v) for k,v in split_subjects.items()}}
samples = []
for path in sorted((SOURCE / "training_exports").glob("*/windows/BGT60TR13C/test/*w00000.npz")):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        samples.append({"variant": path.relative_to(SOURCE / "training_exports").parts[0],
                        "sample": path.name,
                        "target_bins_dar": meta["feature_meta"]["target_bins_dar"],
                        "arrays": {k: {"std": float(data[k].std()),
                                       "nonzero": int(np.count_nonzero(data[k]))}
                                   for k in ("x_time", "x_freq")}})
output["sampled_feature_audit"] = samples
(OUTPUT / "analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
for filename in ("comparison.json", "protocol.json", "fixed_range_centers.json", "frozen_windows.json"):
    (OUTPUT / filename).write_bytes((SOURCE / filename).read_bytes())
lines = ["# BGT60 frontend ablation", "", "Kaggle kernel completed. Seed 42; CycleFormer v3-small; fixed legacy ROI; 30→20 Hz complex resampling.", "",
         "| Frontend | Retrained MAE | Prediction std | Pearson r | Frozen legacy MAE |",
         "| --- | ---: | ---: | ---: | ---: |"]
for name in ("legacy_mean", "full_mean", "legacy_none", "full_none"):
    r, f = summary[name+"_retrained"], summary[name+"_frozen_legacy"]
    corr = "uninterpretable (near constant)" if r["near_constant"] else f'{r["reported_pearson_r"]:.3f}'
    lines.append(f'| {name} | {r["mae_bpm"]:.3f} | {r["prediction_std_bpm"]:.6g} | {corr} | {f["mae_bpm"]:.3f} |')
lines += ["", "The apparent full_none improvement is a change in almost constant prediction level, not demonstrated HR tracking. Its raw Pearson r of 0.476 is computed from only micro-bpm numerical variation; it should not be presented as evidence of useful tracking.", "",
          "The training-mean constant baseline MAE is 7.882 bpm. Beating it alone does not establish signal use: the full_none mean prediction used as a constant reproduces its MAE to numerical precision.", "",
          "All 32 baseline range centers were bin 0, so every variant used bins 0–7. This is a localization/DC warning requiring raw range-profile checks; this experiment cannot rule out benefits at a valid chest ROI or with native sampling.", "",
          "The test split contains only participant 2 (368 overlapping windows); validation is participant 4 and training is participants 1, 3, 5, 6, 7, 8. A single seed and single test subject cannot establish generalization or statistical significance. The assumed common radar/reference acquisition start has not been independently verified.", "",
          "In the four downloaded full_none first-window samples (one per distance, participant 2), both x_time and x_freq are entirely zero. Their EDACM target bins all have range index 0. This directly confirms feature collapse in these inspected windows; it is not an audit of every window in the dataset.", "",
          "Next: inspect raw range profiles with and without fast-time ADC DC removal and native sampling, then validate phase/feature variance before more training. Do not promote full_none as the best physiological frontend on these results.", ""]
(OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Saved {OUTPUT / 'REPORT.md'}; {len(samples)} sampled windows audited")
