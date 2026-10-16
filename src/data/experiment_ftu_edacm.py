"""FTU physical-time export and prespecified, non-training signal diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, stft

from src.data.ftu_timebase import frame_times, labels_at_frames
from src.features.edacm import target_edacm_signal
from src.features.representations import frequency_features

OFFSETS = [-2, -1, 0, 1, 2]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def peaks(x):
    f = np.fft.rfftfreq(256, 1 / 20) * 60
    p = abs(np.fft.rfft(detrend(x) * np.hanning(256))) ** 2
    valid = np.flatnonzero((f >= 45) & (f <= 150))
    candidates = find_peaks(p[valid])[0]
    # Include a band boundary only when it is a one-sided local maximum.
    candidates = np.unique(np.r_[candidates,
        [0] if p[valid[0]] > p[valid[1]] else [],
        [len(valid)-1] if p[valid[-1]] > p[valid[-2]] else []]).astype(int)
    ranked = valid[candidates[np.argsort(p[valid[candidates]])[::-1]]][:3]
    return float(f[valid[np.argmax(p[valid])]]), f[ranked]


def model_check():
    import torch
    from types import SimpleNamespace
    from src.models.factory import create_model_and_config
    torch.set_num_threads(1)
    results = {}
    for name in ["tcn", "transformer", "cycleformer"]:
        args = SimpleNamespace(model=name,time_channels=1,freq_channels=1,d_model=64,nhead=4,
            num_layers=2,d_ff=128,dropout=.1,time_only=False,hidden_channels=48,num_blocks=4,
            kernel_size=7,heart_periods=None,respiration_periods=None)
        m, cfg = create_model_and_config(args)
        assert cfg.time_channels == cfg.freq_channels == 1
        m.eval()
        with torch.no_grad():
            y = m(torch.zeros(2, 1, 256), torch.zeros(2, 1, 129))
        assert y.shape == (2,) and torch.isfinite(y).all(), (name, y.shape)
        results[name] = list(y.shape)
    return results


def run(root, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    history_paths = list(root.rglob("training_exports/manifest.csv"))
    assert len(history_paths) == 1, history_paths
    history_path = history_paths[0]
    history = pd.read_csv(history_path)
    records = history.groupby("sample_tag").first()
    assert len(records) == 440
    paths = {p.stem: p for p in root.rglob("*.npz") if "ftu-rda-exports" in str(p)}
    assert set(records.index) <= paths.keys()
    protocol = dict(sampling_rate_hz=20, start_offset_s=0, synchronization="assumed_unverified",
        offsets_s=OFFSETS, window=256, stride=128, min_coverage=0.8,
        sensitivity_comparison="intersection of windows >=80% coverage at every offset",
        reference="original RDA export time and HR arrays; no time stretching or extrapolation",
        representation="edacm_only", normalization="none after representation z-score",
        peak_band_bpm=[45,150], top3_tolerance_bpm=5,
        negative_control="roll window labels by 4 windows (25.6 s); circular diagnostic, not synchronization",
        split="frozen historical E2a groups; exploratory, not untouched test subjects",
        history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(), training_launched=False)
    write_json(out / "protocol.json", protocol)
    staged = out / "input_manifest" / "FTU"
    staged.mkdir(parents=True)
    source_rows = []
    for tag, row in records.iterrows():
        source_rows.append(dict(status="ok", sample_tag=tag, npz_path=str(paths[tag]),
                                participant_id=str(row.group_key).split("/")[-1]))
    pd.DataFrame(source_rows).to_csv(staged / "manifest.csv", index=False)
    export = out / "training_exports_ftu_edacm_v2"
    subprocess.run([sys.executable, "-m", "src.data.build_training_dataset",
        "--exports-dir", str(staged.parent), "--output-dir", str(export),
        "--datasets", "FTU", "--representation", "edacm_only", "--normalize", "none",
        "--split-manifest", str(history_path), "--split-mode", "stable_grouped",
        "--ftu-start-offset-s", "0", "--drop-rda"], check=True)
    manifest = pd.read_csv(export / "manifest.csv")
    assert manifest.groupby("group_key").split.nunique().max() == 1
    assert len(manifest.sample_tag.unique()) == len(records)
    rows, alignment, source_hashes = [], [], {}
    figures = out / "spectrograms"
    figures.mkdir()
    # One predeclared lexicographic record per subject; no error-based selection.
    plot_tags = set(history.sort_values("sample_tag").groupby("group_key").first().sample_tag)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for tag, group in manifest.groupby("sample_tag", sort=True):
        group = group.sort_values("window_start")
        path = paths[tag]
        source_hashes[tag] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            radar, rt, hr = z["radar"], z["time"], z["heart_rate"]
        times = frame_times(len(radar), 20, radar_start_on_reference_clock_s=0)
        mapped = {o: labels_at_frames(rt, hr, times + o) for o in OFFSETS}
        full_phase, _ = target_edacm_signal(radar)
        sf, st, sz = stft(full_phase, fs=20, window="hann", nperseg=256,
            noverlap=128, boundary=None, padded=False, detrend="linear")
        band = (sf * 60 >= 45) & (sf * 60 <= 150)
        stft_hr = sf[band][abs(sz[band]).argmax(axis=0)] * 60
        all_labels = group.label_heart_rate.to_numpy()
        shifted = np.roll(all_labels, 4)
        for j, row in enumerate(group.itertuples()):
            s, e = int(row.window_start), int(row.window_end)
            with np.load(row.window_path) as z:
                xt, xf = z["x_time"], z["x_freq"]
                assert xt.shape == (1,256) and xf.shape == (1,129)
                assert np.isfinite(xt).all() and np.isfinite(xf).all()
                expected_freq = frequency_features(xt)[0]
                np.testing.assert_allclose(xf, expected_freq, atol=1e-6)
                np.testing.assert_allclose(z["label_series_heart_rate"], mapped[0][s:e], atol=1e-5)
                np.testing.assert_allclose(z["label_heart_rate"], np.nanmean(mapped[0][s:e]), atol=1e-5)
                pred, top = peaks(xt[0])
            common = all(np.isfinite(mapped[o][s:e]).mean() >= .8 for o in OFFSETS)
            for offset in OFFSETS:
                v = mapped[offset][s:e]
                alignment.append(dict(sample_tag=tag, window_start=s, offset_s=offset,
                    coverage=float(np.isfinite(v).mean()), label=float(np.nanmean(v)), common=common))
            baseline = stft_hr[np.argmin(abs(st - (s+128)/20))]
            rows.append(dict(sample_tag=tag, group_key=row.group_key, split=row.split,
                window_start=s, label=row.label_heart_rate, fft=pred, stft=baseline,
                top3_hit=bool(np.any(abs(top-row.label_heart_rate)<=5)),
                shifted_top3_hit=bool(np.any(abs(top-shifted[j])<=5)),
                shifted_label=shifted[j], common=common))
        if tag in plot_tags:
            fig, ax = plt.subplots(figsize=(11,4))
            ax.pcolormesh(st, sf[band]*60, 20*np.log10(abs(sz[band])+1e-8), shading="auto")
            ax.plot(rt, hr, "w-", label="Reference (offset=0 assumed)")
            ax.plot(st, stft_hr, "r.", label="STFT peak")
            ax.set(xlabel="Radar time (s)", ylabel="bpm", ylim=(45,150), title=tag)
            ax.legend(); fig.tight_layout(); fig.savefig(figures / (tag+".png")); plt.close(fig)
    df = pd.DataFrame(rows)
    train = df[df.split == "train"].label
    df["train_mean"] = train.mean(); df["train_median"] = train.median()
    df.to_csv(out / "signal_baselines.csv", index=False)
    align = pd.DataFrame(alignment)
    align.to_csv(out / "label_alignment.csv", index=False)
    common = align[align.common].pivot(index=["sample_tag","window_start"], columns="offset_s", values="label")
    sensitivity = {str(o): dict(mean_abs_change=float(abs(common[o]-common[0]).mean()),
        max_abs_change=float(abs(common[o]-common[0]).max())) for o in OFFSETS}
    summaries=[]
    for scope, groups in [("split",df.groupby("split")), ("subject",df.groupby("group_key"))]:
        for key, g in groups:
            for method in ["train_mean","train_median","fft","stft"]:
                summaries.append(dict(scope=scope, group=key, method=method, n=len(g),
                    mae=float(abs(g[method]-g.label).mean()),
                    top3_hit=float(g.top3_hit.mean()), shifted_top3_hit=float(g.shifted_top3_hit.mean())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "baseline_summary.csv", index=False)
    record_metrics = []
    for tag, g in df.groupby("sample_tag"):
        for method in ["fft", "stft"]:
            sy, sp = g.label.std(ddof=0), g[method].std(ddof=0)
            record_metrics.append(dict(sample_tag=tag, group_key=g.group_key.iloc[0], method=method,
                n=len(g), mae=float(abs(g[method]-g.label).mean()),
                pearson_r=float(g[method].corr(g.label)) if sy>1e-8 and sp>1e-8 else None,
                variance_ratio=float(sp/sy) if sy>1e-8 else None,
                shifted_mae=float(abs(g[method]-g.shifted_label).mean())))
    pd.DataFrame(record_metrics).to_csv(out / "record_metrics.csv", index=False)
    sensitivity_metrics = []
    for offset in OFFSETS:
        matched = df.merge(common[offset].rename("offset_label").reset_index(),
                           on=["sample_tag","window_start"], validate="one_to_one")
        mean = matched.loc[matched.split=="train", "offset_label"].mean()
        for split, g in matched.groupby("split"):
            for method in ["fft","stft","train_mean"]:
                pred = mean if method=="train_mean" else g[method]
                sensitivity_metrics.append(dict(offset_s=offset,split=split,method=method,n=len(g),
                    mae=float(abs(pred-g.offset_label).mean())))
    pd.DataFrame(sensitivity_metrics).to_csv(out / "offset_baselines.csv", index=False)
    write_json(out / "source_hashes.json", source_hashes)
    validation = dict(records=len(records), windows=len(df), common_offset_windows=len(common),
        elapsed_sec=time.perf_counter()-started, finite_features=True, label_recomputed=True,
        frequency_recomputed=True, split_group_disjoint=True, single_channel_forward=model_check(),
        sensitivity=sensitivity, training_launched=False)
    write_json(out / "validation.json", validation)
    table = summary[summary.scope=="split"].to_csv(index=False)
    (out / "REPORT.md").write_text("# FTU physical time + EDACM-only v2\n\n"
        "Exploratory signal diagnostic; no neural model training. Offset zero is an assumption, not verified synchronization.\n\n"
        f"Validated {len(df)} windows from {len(records)} records. Offset sensitivity uses {len(common)} common windows.\n\n"
        "FFT uses window-local EDACM; STFT uses full-record EDACM (target selection context differs). "
        "Top-three peaks within 5 bpm are an oracle diagnostic, not a deployable estimator. "
        "Circular label shifting is a weak negative control when HR changes little.\n\n"
        "```csv\n"+table+"```\n\nSee per-subject CSV, alignment CSV and validation.json. "
        "Legacy exports are unchanged.\n", encoding="utf-8")


if __name__ == "__main__":
    run(Path("/kaggle/input"), Path("/kaggle/working/ftu_timebase_edacm_v2"))
