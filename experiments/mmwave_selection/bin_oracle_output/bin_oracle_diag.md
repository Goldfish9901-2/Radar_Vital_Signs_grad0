# Range-bin selection: oracle vs unsupervised scores

- sessions: 440   participants: 110   fs: 10.0 Hz
- HR is the classical FFT-peak estimate on each bin's unwrapped, band-passed phase (paper Fig.5 baseline).
- ECG HR is used only to *measure* error, never to pick a bin.

## Leaderboard (MAE, BPM)

`%<3-harm` counts a hit if k*estimate matches ECG for k in {0.5, 1, 2} — it separates 'wrong bin' from 'right bin, wrong harmonic'.

| rule | MAE | median | %<3 | %<5 | %<3-harm | Spearman(score,-err) |
|------|----:|-------:|----:|----:|---------:|----------------------:|
| oracle_min_err | 1.64 | 0.29 | 90.7 | 95.0 | 90.7 | +nan |
| oracle_chance_uniform | 10.19 | nan | 58.0 | nan | nan | +nan |
| max_energy | 15.74 | 9.53 | 29.5 | 35.9 | 30.7 | +nan |
| first_valid_bin | 15.73 | 9.53 | 29.5 | 36.1 | 30.7 | +nan |
| pc1_trace | 15.31 | 9.94 | 22.7 | 35.0 | 23.2 | +nan |
| mean_bp_trace | 15.99 | 11.52 | 19.3 | 29.5 | 19.8 | +nan |
| top1::ant_coh | 18.74 | 12.43 | 10.9 | 20.9 | 12.3 | +0.002 |
| top1::band_frac | 21.12 | 15.10 | 14.3 | 21.6 | 15.9 | -0.053 |
| top1::energy | 15.74 | 9.53 | 29.5 | 35.9 | 30.7 | +0.049 |
| top1::hr_power | 17.42 | 13.07 | 13.6 | 22.7 | 14.8 | -0.174 |
| top1::low_leak | 19.42 | 12.77 | 14.1 | 22.3 | 15.5 | -0.022 |
| top1::near_energy_peak | 15.74 | 9.53 | 29.5 | 35.9 | 30.7 | +0.052 |
| top1::pc1_corr | 17.68 | 13.10 | 18.0 | 26.6 | 19.3 | +0.052 |
| top1::peak_amp | 17.57 | 13.84 | 13.4 | 20.9 | 14.3 | -0.108 |
| top1::peak_prom | 16.58 | 11.51 | 24.5 | 33.9 | 25.2 | -0.052 |
| top1::peak_ratio | 16.40 | 11.41 | 27.0 | 35.5 | 28.0 | -0.063 |
| top1::phase_var | 18.15 | 13.42 | 13.0 | 21.6 | 14.1 | -0.102 |
| top1::probe_energy | 15.74 | 9.53 | 29.5 | 35.9 | 30.7 | +0.046 |
| top1::prom_x_coh | 16.58 | 11.72 | 20.2 | 29.3 | 21.1 | -0.026 |
| top1::prom_x_temp | 16.89 | 11.99 | 23.6 | 32.0 | 24.1 | +0.185 |
| top1::spec_entropy | 16.12 | 11.28 | 24.8 | 33.9 | 25.0 | +0.294 |
| top1::temp_cont | 18.06 | 13.86 | 18.0 | 25.2 | 18.9 | +0.296 |

### Where the max-energy rule points

- probe `single_bin` argmax bin ((E|z|)^2): median 2, range [2, 3], 100.0% <= bin 4, 100.0% <= bin 8
- max-energy argmax bin: median 2, range [2, 3], 100.0% <= bin 4, 100.0% <= bin 8
- oracle argmin-err bin: median 27, range [2, 63], 8.6% <= bin 4, 18.9% <= bin 8

## By stratum (MAE, BPM)

| stratum | oracle | max_energy | pc1_trace | mean_bp_trace | ant_coh | band_frac | energy | hr_power | low_leak | near_energy_peak | pc1_corr | peak_amp | peak_prom | peak_ratio | phase_var | probe_energy | prom_x_coh | prom_x_temp | spec_entropy | temp_cont |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lying/Post-exercise | 0.83 | 16.19 | 17.83 | 17.72 | 20.81 | 26.29 | 16.19 | 19.55 | 27.12 | 16.19 | 21.30 | 20.04 | 19.39 | 19.08 | 20.38 | 16.19 | 18.53 | 20.29 | 18.80 | 20.72 |
| Lying/Rest | 1.18 | 7.14 | 9.16 | 11.20 | 14.41 | 13.90 | 7.14 | 13.22 | 13.71 | 7.14 | 11.08 | 12.11 | 8.51 | 7.97 | 14.54 | 7.14 | 10.11 | 8.64 | 8.20 | 11.45 |
| Sitting/Post-exercise | 2.21 | 25.23 | 22.03 | 22.19 | 23.70 | 28.93 | 25.23 | 22.39 | 22.87 | 25.23 | 23.61 | 23.96 | 25.04 | 25.30 | 24.47 | 25.23 | 23.29 | 24.79 | 23.46 | 25.04 |
| Sitting/Rest | 2.36 | 14.39 | 12.22 | 12.86 | 16.03 | 15.36 | 14.39 | 14.52 | 13.99 | 14.39 | 14.72 | 14.16 | 13.36 | 13.24 | 13.20 | 14.39 | 14.39 | 13.84 | 14.01 | 15.05 |

## Best decile separation (top-score decile vs all bins)

| score | decile-1 MAE | decile-1 %<3 | decile-10 MAE |
|---|---:|---:|---:|
| ant_coh | 18.64 | 13.1 | 17.62 |
| band_frac | 19.36 | 15.8 | 17.42 |
| energy | 17.36 | 20.8 | 18.39 |
| hr_power | 18.28 | 12.5 | 18.73 |
| low_leak | 18.82 | 15.4 | 17.82 |
| near_energy_peak | 16.97 | 21.4 | 18.87 |
| pc1_corr | 17.61 | 18.3 | 18.69 |
| peak_amp | 17.70 | 13.6 | 19.34 |
| peak_prom | 16.75 | 21.6 | 21.39 |
| peak_ratio | 16.90 | 21.5 | 21.13 |
| phase_var | 17.96 | 13.6 | 19.05 |
| probe_energy | 17.44 | 20.8 | 18.41 |
| prom_x_coh | 17.20 | 16.7 | 18.61 |
| prom_x_temp | 16.89 | 21.5 | 22.95 |
| spec_entropy | 16.70 | 20.7 | 20.16 |
| temp_cont | 17.34 | 18.3 | 23.61 |
