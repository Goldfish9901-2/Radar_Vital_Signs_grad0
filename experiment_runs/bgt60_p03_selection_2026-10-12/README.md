# p03 frozen channel selection: Kaggle outputs and local follow-up

Kaggle run: https://www.kaggle.com/code/goldfish9901/bgt60-p03-quality-selection
Version 1 completed. Downloads and local reproduction: 2026-10-12.

## Results

All selectors share the same 384-frame unwrap/detrend/Hann FFT estimator.
Energy and quality selectors only inspect zero-Doppler radar coefficients;
reference HR is used after choices are fixed. Exact rules are in `protocol.json`.
No VMD, model training or rule tuning occurred.

| Record | Selector | Full MAE bpm | Full r | Switches |
|---|---|---:|---:|---:|
| 0.3 m | Fixed | 4.862 | 0.395 | 0 |
| 0.3 m | Energy | 4.516 | 0.382 | 8 |
| 0.3 m | Quality | 5.493 | 0.446 | 11 |
| 0.6 m | Fixed | 20.683 | 0.126 | 0 |
| 0.6 m | Energy | 17.720 | -0.001 | 25 |
| 0.6 m | Quality | 16.765 | -0.023 | 41 |

Each full-record result has 91 valid reference windows out of 92. No quality
threshold removes windows. The 55 matched interior windows support identical
coverage across the four pre-specified reference offsets.

0.6 m quality selection reduces mean error by 3.918 bpm but improves only 35,
worsens 18 and ties 38 of the 91 valid windows. Its matched-window r is -0.163;
negative reference offsets -60/-120 s actually have slightly lower MAE than zero.
This does not demonstrate better synchronous HR tracking. The heuristic switches
41 times, versus 25 for energy selection. Median prediction change at switches
is 14.063 bpm versus 9.375 bpm; these descriptive differences do not establish
that switching itself causes the error.

0.3 m energy selection improves MAE by only 0.347 bpm: six windows improve,
five worsen and 80 tie the fixed FFT estimate. Quality selection worsens MAE by
0.630 bpm. These are two exploratory records selected after the earlier audit;
no cross-subject generalization or statistical significance claim is justified.

Pre-specified controls remain negative or mixed: bin23 at RX2 in the first
150 s of 0.6 m reduces error but produces negative correlation; switching RX
does not reliably improve the last 150 s of 0.3 m. All controls are retained in
`prespecified_controls.json`, without selecting a winner by label error.

## Local continuation and validation

`REPORT.md`, `summary.json`, `protocol.json`, `prespecified_controls.json`,
`input_provenance.json`, both selection JSONs and `execution_log.json` are actual
Kaggle artifacts. `reproduction_validation.json` is the subsequent local uv
verification: six JSON files reproduce within `1e-8`, including input hashes,
selection trajectories and every prediction. Calculation took 1.42 s;
tests plus calculation and verification took 3.73 s.

`FOLLOWUP.md`, `followup.json`, `timing.json` and the two selection plots are
local follow-up outputs. That experiment took 1.72 s and changed no selector
parameters. Independent constant phase rotations of each channel preserve all
energy/quality choices and FFT predictions. Altering samples after frame 9000
preserves all selections whose windows end before that frame. These tests check
implementation invariance and causality, not physiological validity.

The two approximately 24.3 MB complex archives are retained locally and in the
prior Kaggle paired audit output; hashes identify them in `input_provenance.json`.
Frame rate, absolute reference start and metric distance calibration remain
unverified. No continuous phase stitching or test-HR-selected bin is used.

## Reproduction

```powershell
uv sync --locked --python 3.13
uv run --locked python reproduce_bgt60_selection.py --reference-dir experiment_runs/bgt60_p03_selection_2026-10-12
uv run --locked python analyze_bgt60_selection.py --input-dir experiment_runs/bgt60_p03_selection_2026-10-12
```

Commands need the previously downloaded complex archives at the default local
paths; use `--input-dir` / `--trace-dir` for alternative locations. Outputs go to
`tmp/` rather than modifying the archived evidence. Tasks known to finish within
five minutes run locally; training continues on Kaggle.

## Decision

Keep the quality rule as a negative exploratory candidate, not an established
improvement. Do not tune weights or stability penalties to these HR labels, and
do not resume NN/domain-adaptation training on the basis of lower p03 MAE.
Independent-subject data are required before claiming a useful selection method.
