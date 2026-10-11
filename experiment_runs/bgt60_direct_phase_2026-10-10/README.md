# BGT60 direct phase audit and subsequent analysis

Kaggle run: https://www.kaggle.com/code/goldfish9901/bgt60-direct-phase-audit
Version 2 completed successfully; outputs downloaded on 2026-10-11. No VMD or
neural model training. The directory date identifies the experiment launch date.

`REPORT.md`, `protocol.json`, `records.json`, `distance_calibration.json`,
`summary.json` and `windows.jsonl.gz` are actual run outputs (the JSONL is losslessly
compressed). `execution_log.json` is the downloaded run log.
`artifact_manifest.json` records the original JSONL hash, size and row count.

`WINDOW_ANALYSIS.md` / `window_analysis.json` and `P03_CHECK.md` / `P03_CHECK.json`
are subsequent local statistical analyses, not additional Kaggle experiments.
The two small p03 native range-profile arrays come from the previous ADC audit.
No raw ADC or raw ECG is included. The run did not export complex time traces or
complete per-window spectra, so these artifacts cannot establish the physical
source of successful or failed estimates.

Reproduce statistics with uv (no model training):

```powershell
uv run --no-project --with numpy python kaggle/analyze_bgt60_phase.py
uv run --no-project --with numpy python kaggle/check_bgt60_p03.py --range-profile-dir experiment_runs/bgt60_direct_phase_2026-10-10
```

The scripts regenerate JSON statistics; the first also regenerates the basic
analysis table. The checked-in Markdown reports include additional interpretation.
To retain those annotations, pass `--input-dir` pointing to a copy of this folder.

Results remain exploratory: distance-dependent peaks are observed, while full
cross-subject direct-phase error is poor. The p03 0.3 m record shows localized
zero-offset correspondence, with deterioration late in the recording. It was
identified after screening records, not validated independently. Frame rate,
chirp slope and absolute radar/reference synchronization remain unverified.
