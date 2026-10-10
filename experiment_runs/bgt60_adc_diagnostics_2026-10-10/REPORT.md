# BGT60 ADC distance / target / phase diagnostic

## Run provenance and interpretation

Completed Kaggle run: https://www.kaggle.com/code/goldfish9901/bgt60-adc-roi-phase-diagnostics
Outputs downloaded on 2026-10-10. This directory contains the generated report,
six JSON artifacts and all 32 range-profile figures. Raw ADC, probe arrays and
the large per-window JSONL remain outside Git. The run preceded commit 1b156b7;
that commit published the diagnostic implementation, not these outputs.

All 22 candidate/ROI configurations have 2,944 overlapping windows from eight
subjects. The zero-feature configurations reproduce constant predictions. Their
LOSO Ridge macro MAE is approximately 10.057 bpm, equal to the training-subject
mean predictor. Nonzero features alone do not demonstrate heartbeat recovery:
the remaining Ridge configurations do not improve that aggregate baseline, and
the phase FFT errors remain approximately 25-30 bpm.

These results are exploratory candidate screening. Selecting a configuration on
these eight subjects and reporting its same LOSO score does not provide an
independent validation of the selection. A subsequent confirmatory comparison
requires independent data or a nested selection protocol. Physical ROI remains
unevaluated because the acquisition chirp slope is unverified; radar/reference
start synchronization and the assumed frame rate also require validation.

Range-profile figures are named `<sample_tag>_range.png`; per-record details are
in `sample_summaries.json` and `by_sample.json`, with evaluation in
`linear_probes.json` and configuration in `protocol.json`.

All 32 short recordings and all overlapping windows audited. No neural training.

Physical ROI: not evaluated: acquisition slope unavailable

NonDC gating excludes bins 0 and 1; this is a diagnostic hypothesis, not validated chest localization.

| Candidate / ROI | Windows | Nonzero features | Median phase variance | FFT valid | FFT MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy_mean/auto_energy | 2944 | 1.000 | 0.00611246 | 2944 | 29.599 |
| legacy_mean/fixed_legacy | 2944 | 1.000 | 0.00611246 | 2944 | 29.599 |
| legacy_mean/auto_nonDC | 2944 | 1.000 | 0.160762 | 2944 | 29.334 |
| full_mean/auto_energy | 2944 | 1.000 | 0.0411977 | 2944 | 28.878 |
| full_mean/fixed_legacy | 2944 | 1.000 | 0.0411977 | 2944 | 28.878 |
| full_mean/auto_nonDC | 2944 | 1.000 | 0.242411 | 2944 | 29.011 |
| legacy_none/auto_energy | 2944 | 0.000 | 0 | 0 | NA |
| legacy_none/fixed_legacy | 2944 | 0.000 | 0 | 0 | NA |
| legacy_none/auto_nonDC | 2944 | 1.000 | 4.59072e-06 | 2944 | 25.071 |
| full_none/auto_energy | 2944 | 0.000 | 0 | 0 | NA |
| full_none/fixed_legacy | 2944 | 0.000 | 0 | 0 | NA |
| full_none/auto_nonDC | 2944 | 1.000 | 4.34164e-06 | 2944 | 25.061 |
| full_none_dc/fixed_legacy | 2944 | 1.000 | 7.88944e-07 | 2944 | 25.355 |
| full_none_dc/auto_energy | 2944 | 1.000 | 7.88944e-07 | 2944 | 25.355 |
| full_none_dc/auto_nonDC | 2944 | 1.000 | 4.34164e-06 | 2944 | 25.061 |
| full_mean_dc/fixed_legacy | 2944 | 1.000 | 0.0264251 | 2944 | 28.157 |
| full_mean_dc/auto_energy | 2944 | 1.000 | 0.0264251 | 2944 | 28.157 |
| full_mean_dc/auto_nonDC | 2944 | 1.000 | 0.242411 | 2944 | 29.011 |
| native_none_dc/auto_energy | 2944 | 1.000 | 7.98637e-07 | 2944 | 25.375 |
| native_none_dc/auto_nonDC | 2944 | 1.000 | 4.32353e-06 | 2944 | 25.072 |
| native_mean_dc/auto_energy | 2944 | 1.000 | 0.0259227 | 2944 | 28.133 |
| native_mean_dc/auto_nonDC | 2944 | 1.000 | 0.230986 | 2944 | 28.858 |

FFT MAE uses valid phase windows only: compare coverage alongside error. HR-band energy alone does not establish heartbeat origin. Labels do not select bins. Reference start synchronization remains an assumption.

## Leave-one-subject-out spectral Ridge probes

Fixed alpha=10; each scaler uses training subjects only. Confidence intervals bootstrap the eight subject MAEs, not overlapping windows.

| Candidate | Subject macro MAE | Subject bootstrap 95% CI |
| --- | ---: | --- |
| legacy_mean__auto_energy_probe | 10.284 | 6.059–16.103 |
| full_mean__auto_nonDC_probe | 10.571 | 6.316–16.430 |
| legacy_mean__fixed_legacy_probe | 10.284 | 6.059–16.103 |
| native_mean_dc__auto_nonDC_probe | 10.562 | 6.352–16.359 |
| full_mean_dc__auto_nonDC_probe | 10.571 | 6.316–16.430 |
| full_mean__auto_energy_probe | 10.302 | 5.989–16.166 |
| full_mean_dc__auto_energy_probe | 10.291 | 5.975–16.113 |
| legacy_none__auto_energy_probe | 10.057 | 5.478–16.105 |
| full_none_dc__auto_nonDC_probe | 10.670 | 6.337–16.566 |
| legacy_none__fixed_legacy_probe | 10.057 | 5.478–16.105 |
| full_mean_dc__fixed_legacy_probe | 10.291 | 5.975–16.113 |
| full_none_dc__auto_energy_probe | 11.074 | 6.629–17.114 |
| full_none__auto_energy_probe | 10.057 | 5.478–16.105 |
| legacy_mean__auto_nonDC_probe | 10.361 | 6.062–16.221 |
| native_none_dc__auto_energy_probe | 11.048 | 6.601–17.068 |
| native_none_dc__auto_nonDC_probe | 10.676 | 6.372–16.554 |
| native_mean_dc__auto_energy_probe | 10.240 | 5.959–16.059 |
| full_none__fixed_legacy_probe | 10.057 | 5.478–16.105 |
| full_mean__fixed_legacy_probe | 10.302 | 5.989–16.166 |
| legacy_none__auto_nonDC_probe | 10.682 | 6.463–16.532 |
| full_none_dc__fixed_legacy_probe | 11.074 | 6.629–17.114 |
| full_none__auto_nonDC_probe | 10.670 | 6.337–16.566 |
