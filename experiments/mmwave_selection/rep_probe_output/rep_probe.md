# Spatial-combination Ridge probe (mmwave-897)

- sessions: 440   participants: 110   fs: 10.0 Hz
- identical subject-level GroupKFold(5) Ridge + inner alpha CV as `mmwave_probe`; the 71-d log1p HR-band spectrum is the feature.

| representation | MAE | Δ vs single_bin | R² | r |
|---|---:|---:|---:|---:|
| single_bin | 12.23 | +0.00 | 0.176 | 0.420 |
| spec_mean | 12.46 | +0.23 | 0.140 | 0.378 |
| pc1 | 12.59 | +0.36 | 0.142 | 0.377 |
| pc2 | 12.45 | +0.22 | 0.143 | 0.378 |
| mean_bp | 12.76 | +0.53 | 0.116 | 0.342 |
| oracle_bin | 12.28 | +0.06 | 0.173 | 0.416 |

`oracle_bin` is a **supervised upper bound** (the bin whose FFT peak is closest to the ECG HR). It is not a method -- it bounds how much perfect range-bin selection could possibly buy.