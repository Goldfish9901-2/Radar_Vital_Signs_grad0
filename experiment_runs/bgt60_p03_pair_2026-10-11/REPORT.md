# BGT60 p03 paired complex signal diagnostic

Kaggle: https://www.kaggle.com/code/goldfish9901/bgt60-p03-complex-pair-audit
Version 2 completed successfully; downloaded and checked on 2026-10-11.
Version 1 stopped while serializing a NumPy boolean. Version 2 fixes JSON export
without changing signal processing or experiment parameters.

The experiment retains all three RX channels, center ±2 native range bins,
first-chirp and zero-Doppler complex coefficients. Baseline centers remain
14 (0.3 m) and 22 (0.6 m), with RX 2 (zero based). Fast-time mean is removed;
chirp clutter subtraction, VMD and neural training are absent. The assumed
frame rate is 30 Hz; absolute synchronization and chirp slope remain unverified.

## Fixed baseline results

Zero Doppler + unwrap, no label-based channel or offset optimization:

| Metric | p03 0.3 m | p03 0.6 m |
|---|---:|---:|
| Total / valid full-record windows | 92 / 91 | 92 / 91 |
| Full-record MAE (bpm) | 4.862 | 20.683 |
| Full-record Pearson r | 0.395 | 0.126 |
| Common valid windows across all offsets | 55 | 55 |
| Common-window MAE (bpm) | 3.008 | 24.993 |
| Common-window Pearson r | 0.602 | -0.026 |

The full-record metrics reproduce the previous direct-phase audit exactly.
The first window lacks complete reference coverage and is excluded, not
extrapolated. Overlapping windows are not independent observations.

## Output inventory and integrity

Each record has 18,000 complex frames, 3 RX, 5 bins, 2 coefficients, 92 analysis
windows, 2 phase methods and 193 one-sided spectral frequencies. Each exported
window JSON contains 5,520 channel/method records. Both NPZ archives pass ZIP CRC,
shape and finite-value checks; hashes and original sizes are recorded in
`download_validation.json`.

Git contains the full window JSON files compressed losslessly as `.json.gz`,
label-free range/RX stability statistics, protocol, metrics, execution log,
diagnostic figures and download validation. The two approximately 24.3 MB NPZ
archives remain available in the private Kaggle version 2 output and locally at
`tmp/bgt60_p03_pair_v2_results/pair_audit/`; they are not included in this summary
commit. They retain complex traces, full unwrap, window phases and full spectra.

The figures show fixed-baseline IQ, amplitude and windowed phase spectra, with
reference HR overlaid under the unverified shared-start assumption. The color
scale is chosen independently for each record; brightness cannot be compared
directly across the two PNGs.

The stability arrays describe all local bins/RX without changing the baseline.
No thresholds are fitted to exclude poor windows. These artifacts enable causal
inspection, but this summary does not yet establish whether motion, amplitude
nulls, ROI drift, respiration harmonics or spectral peak selection cause failure.
The favorable 0.3 m case was discovered after screening 32 records, so it is an
exploratory finding rather than independent validation. No model training resumed.
