# Eight-subject frozen direct-phase baseline reaudit

This reuses existing actual ADC-derived outputs, not a new experiment or independent cohort. No algorithm/ROI changes, no training.

| Subject | Full valid/total | FFT MAE | Other-seven-subject mean MAE | Pearson r |
|---|---:|---:|---:|---:|
| 1 | 364/368 | 8.369 | 28.778 | 0.054 |
| 2 | 364/368 | 33.517 | 5.920 | -0.027 |
| 3 | 364/368 | 13.915 | 2.811 | 0.242 |
| 4 | 364/368 | 37.994 | 14.832 | 0.004 |
| 5 | 364/368 | 21.509 | 3.237 | 0.066 |
| 6 | 364/368 | 15.896 | 7.089 | 0.026 |
| 7 | 364/368 | 29.973 | 9.248 | -0.095 |
| 8 | 364/368 | 31.444 | 8.585 | -0.020 |

Subject macro FFT MAE: 24.077 bpm; constant: 10.062 bpm.
Mean within-record r (descriptive arithmetic mean): 0.024.
Overlapping windows are not independent subjects. p03 is exploratory; all others were also examined previously. Absolute synchronization and frame rate remain unverified.