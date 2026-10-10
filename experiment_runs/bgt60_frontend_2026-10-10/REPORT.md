# BGT60 frontend ablation

Kaggle kernel completed. Seed 42; CycleFormer v3-small; fixed legacy ROI; 30→20 Hz complex resampling.

| Frontend | Retrained MAE | Prediction std | Pearson r | Frozen legacy MAE |
| --- | ---: | ---: | ---: | ---: |
| legacy_mean | 5.251 | 0.283092 | 0.088 | 5.251 |
| full_mean | 5.326 | 0.297329 | -0.109 | 5.255 |
| legacy_none | 5.227 | 0 | uninterpretable (near constant) | 6.236 |
| full_none | 4.068 | 3.51052e-06 | uninterpretable (near constant) | 6.236 |

The apparent full_none improvement is a change in almost constant prediction level, not demonstrated HR tracking. Its raw Pearson r of 0.476 is computed from only micro-bpm numerical variation; it should not be presented as evidence of useful tracking.

The training-mean constant baseline MAE is 7.882 bpm. Beating it alone does not establish signal use: the full_none mean prediction used as a constant reproduces its MAE to numerical precision.

All 32 baseline range centers were bin 0, so every variant used bins 0–7. This is a localization/DC warning requiring raw range-profile checks; this experiment cannot rule out benefits at a valid chest ROI or with native sampling.

The test split contains only participant 2 (368 overlapping windows); validation is participant 4 and training is participants 1, 3, 5, 6, 7, 8. A single seed and single test subject cannot establish generalization or statistical significance. The assumed common radar/reference acquisition start has not been independently verified.

In the four downloaded full_none first-window samples (one per distance, participant 2), both x_time and x_freq are entirely zero. Their EDACM target bins all have range index 0. This directly confirms feature collapse in these inspected windows; it is not an audit of every window in the dataset.

Next: inspect raw range profiles with and without fast-time ADC DC removal and native sampling, then validate phase/feature variance before more training. Do not promote full_none as the best physiological frontend on these results.
