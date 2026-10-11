# p03 distance comparison

Source: windows.jsonl, records.json and the earlier native whole-record range profiles.
No raw complex time series or complete per-window spectra were exported by this run;
those cannot be reconstructed from summary statistics. This review establishes
prediction-level differences, not their physical cause.

| Zero-Doppler unwrap | 0.3 m | 0.6 m |
|---|---:|---:|
| Native bin / RX (zero based) | 14 / 2 | 22 / 2 |
| Full valid windows | 91 | 91 |
| Full MAE bpm | 4.862 | 20.683 |
| Full Pearson r | 0.395 | 0.126 |
| Matched interior MAE bpm | 3.008 | 24.993 |
| Matched interior Pearson r | 0.602 | -0.026 |
| Matched reference mean bpm | 85.971 | 85.169 |
| Matched prediction mean bpm | 85.227 | 62.983 |
| Matched HR-band energy ratio median | 0.0349 | 0.0607 |
| Matched lowest-bin fraction | 0% | 20% |
| Nonoverlapping full-window MAE bpm | 5.901 | 21.546 |
| Nonoverlapping full-window Pearson r | 0.350 | 0.033 |

Interior windows are selected solely for complete coverage of all fixed offsets;
their better score at 0.3 m is not the full-record score. Nonoverlapping windows
reduce direct overlap but are still temporally dependent.

| Window-start block (s) | 0.3 m MAE / r | 0.6 m MAE / r |
|---|---|---|
| 0–150 | 5.373 / 0.411 | 24.676 / 0.132 |
| 150–300 | 1.680 / 0.556 | 27.125 / -0.073 |
| 300–450 | 3.809 / 0.700 | 23.702 / -0.172 |
| 450–600 | 8.993 / 0.170 | 5.804 / 0.155 |

Blocks group window starts; windows may cross block boundaries. No time blocks
were used to choose ROI, method or alignment. The late improvement at 0.6 m
alongside late deterioration at 0.3 m is consistent with time-varying signal
quality, but motion, ROI stability and phase corruption cannot be distinguished
without complex traces.

| Reference offset (s) | 0.3 m MAE / r | 0.6 m MAE / r |
|---|---|---|
| 0 | 3.008 / 0.602 | 24.993 / -0.026 |
| -60 | 4.347 / -0.183 | 24.409 / 0.353 |
| +60 | 4.140 / 0.016 | 24.857 / 0.030 |
| -120 | 4.500 / -0.111 | 24.575 / -0.164 |
| +120 | 4.139 / 0.062 | 25.140 / -0.151 |

The 0.3 m record shows stronger zero-offset correspondence than its fixed
negative controls. It was identified after exploring 32 records, so this is
evidence for targeted follow-up, not independent validation. Do not select -60 s
for 0.6 m on the basis of its higher correlation.

Next causal inspection must export native complex coefficients, per-window phase
and spectra, plus range/RX stability through time for these two records. Keep
the current bins, RX and offsets fixed. Without raw traces, the present outputs
cannot establish whether chest motion, noise, respiration harmonics or motion
artifacts cause the difference. No NN training is justified by this comparison.
