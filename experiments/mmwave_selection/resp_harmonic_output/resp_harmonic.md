# Respiration-harmonic hypothesis (mmwave-897, canonical bin)

- sessions: 440   participants: 110   fs: 10.0 Hz
- breathing band (0.1, 0.6) Hz; harmonics k=[2, 3, 4, 5, 6, 7, 8]; exclusion guard ±0.08 Hz

## Mechanism

- HR-band peak sits on a respiration harmonic in **61.4%** of sessions

## FFT-peak estimator (MAE, BPM)

| variant | MAE |
|---|---:|
| plain | 15.72 |
| peak_excl_harmonics | 17.47 |
| harmonic_removed_then_peak | 17.47 |

## Ridge on the 71-d spectrum (identical CV)

| representation | MAE | R² | r |
|---|---:|---:|---:|
| single_bin_plain | 12.23 | 0.176 | 0.420 |
| harmonic_removed | 12.26 | 0.173 | 0.417 |

## MAE by true-HR decile (the smoking gun)

| decile | HR range | mean HR | peak | peak excl. harm | harm removed |
|---|---|---:|---:|---:|---:|
| 1 | 41-57 | 52.1 | 8.51 | 21.22 | 21.22 |
| 2 | 57-61 | 58.9 | 4.58 | 9.52 | 9.52 |
| 3 | 61-64 | 62.6 | 6.25 | 8.25 | 8.32 |
| 4 | 64-68 | 65.9 | 6.57 | 7.85 | 7.85 |
| 5 | 68-72 | 70.0 | 7.57 | 11.88 | 11.86 |
| 6 | 73-77 | 74.8 | 11.16 | 13.43 | 13.43 |
| 7 | 77-82 | 79.2 | 17.19 | 15.80 | 15.80 |
| 8 | 82-87 | 84.3 | 19.62 | 15.33 | 15.33 |
| 9 | 87-99 | 92.8 | 24.45 | 23.28 | 23.27 |
| 10 | 100-179 | 118.7 | 51.31 | 48.11 | 48.11 |

## By stratum

| stratum | peak | excl. harm | harm removed | %peak on harmonic | f_r |
|---|---:|---:|---:|---:|---:|
| Lying/Post-exercise | 16.19 | 18.31 | 18.31 | 73.6 | 0.291 |
| Lying/Rest | 7.17 | 11.94 | 11.94 | 61.8 | 0.191 |
| Sitting/Post-exercise | 25.22 | 24.43 | 24.45 | 58.2 | 0.234 |
| Sitting/Rest | 14.30 | 15.18 | 15.18 | 51.8 | 0.160 |