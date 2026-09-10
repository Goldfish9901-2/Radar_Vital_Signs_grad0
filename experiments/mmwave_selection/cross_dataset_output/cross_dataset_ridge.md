# Canonical cross-dataset Ridge probe

- cv modes: grouped, random  (`grouped` = subject-level, `random` = folds ignore subject identity)
- exports: `/srv/ext-timeshift/radar_vital_signs/run/upstream/training_exports`
- protocol: subject-level GroupKFold(5) + inner GroupKFold(5) alpha CV
- alphas: [0.001, 0.1, 10.0, 1000.0]; standardize on train folds; y centered on train folds; predictions clipped to [30, 200] BPM
- identical to `src/data/mmwave_probe.evaluate`, which produces the mmwave-897 12.23 reference


## grouped folds (window level)

| dataset | features | constant | Ridge | gain | r | R² | pred σ | label σ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FTU | phase71 | 8.71 | 8.77 | -0.05 | -0.644 | -0.380 | 2.24 | 9.04 |
| FTU | x_freq | 8.71 | 8.75 | -0.04 | -0.590 | -0.372 | 2.34 | 9.04 |
| BGT60TR13C | phase71 | 11.34 | 11.39 | -0.06 | -0.772 | -0.530 | 3.39 | 11.74 |
| BGT60TR13C | x_freq | 11.34 | 11.45 | -0.12 | -0.766 | -0.543 | 3.48 | 11.74 |
| PhysDrive | phase71 | 10.29 | 10.35 | -0.06 | -0.270 | -0.093 | 1.65 | 12.05 |
| PhysDrive | x_freq | 10.29 | 10.39 | -0.10 | -0.257 | -0.108 | 1.93 | 12.05 |

## random folds (window level)

| dataset | features | constant | Ridge | gain | r | R² | pred σ | label σ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FTU | phase71 | 8.71 | 7.31 | +1.41 | 0.051 | -0.007 | 1.36 | 9.04 |
| FTU | x_freq | 8.71 | 7.28 | +1.43 | 0.115 | -0.022 | 2.73 | 9.04 |
| BGT60TR13C | phase71 | 11.34 | 8.96 | +2.38 | 0.079 | 0.005 | 1.37 | 11.74 |
| BGT60TR13C | x_freq | 11.34 | 9.01 | +2.32 | 0.036 | -0.006 | 1.42 | 11.74 |
| PhysDrive | phase71 | 10.29 | 9.81 | +0.47 | 0.170 | 0.028 | 2.37 | 12.05 |
| PhysDrive | x_freq | 10.29 | 9.85 | +0.44 | 0.123 | 0.015 | 1.72 | 12.05 |

## Reading it

- `gain` = constant MAE − Ridge MAE. That is the quantity to compare across datasets, not the absolute MAE.
- `pred σ` vs `label σ` is the collapse check: if pred σ is tiny relative to label σ, the model is predicting the mean regardless of how low its MAE looks.
- mmwave-897's row (not computed here) comes from a *different* representation (71-d log1p HR-band phase spectrum) because the exports store `x_rda` as log-magnitude, so the complex phase is not recoverable. Reference: constant 14.28, FFT peak 15.7, Ridge 12.23 (r 0.420, R² 0.176, session-level).