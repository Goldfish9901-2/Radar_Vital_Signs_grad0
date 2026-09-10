# Phase-0 RDA Candidate Screening — mmwave-897-2026

- generated: 2026-09-10T03:36:32+00:00
- dataset: /kaggle/input/datasets/goldfish9901/mmwave-897-2026
- sessions: 440  (participants: 110)
- fs: 10.0 Hz   spect: 71-d log1p HR-band (F_GRID)
- margins: improve=0.5  worsen=0.5  hr-power floor=0.5
- known linear-probe reference: MAE 12.23 / R² 0.176 (single_bin, full 440)

## D=1 compatibility adapter (IMPORTANT)

mmwave-897 ships as (F, 8, 64) with NO Doppler axis (chirps averaged at storage). The D=1 inserted by ensure_rda_4d is a COMPATIBILITY ADAPTER, not a real Doppler bin. Doppler-processing conclusions are LIMITED on this dataset. PhysDrive (F,8,16,8) is the fuller RDA target.

## Per-stage validity on mmwave-897

- **C (clutter)**: valid on mmwave-897 (slow-time processing; no Doppler needed)
- **L (localization)**: valid on mmwave-897 (range/angle energy & HR-band)
- **R (range selection)**: valid on mmwave-897 (range axis present)
- **B (beamforming)**: LIMITED on mmwave-897 — angle axis is antennas, not true angular-Doppler; MVDR/Bartlett still test spatial reweighting but NOT Doppler discrimination
- **P (phase)**: valid on mmwave-897 (single/multi-bin phase extraction)

## Leaderboard

| Stage | Candidate | Downstream | MAE | ΔMAE | R² | r | HR-pwr | Status |
|-------|-----------|-----------|----:|-----:|---:|---:|------:|--------|
| R | hr_band | spec_mean | 12.35 | -0.16 | 0.165 | 0.406 | 0.3321 | HOLD |
| R | weighted_center | spec_mean | 12.50 | +0.00 | 0.150 | 0.388 | 0.3321 | HOLD |
| R | peak | spec_mean | 12.55 | +0.05 | 0.141 | 0.375 | 0.3321 | HOLD |
| P | single_bin_trace | phase_trace | 12.23 | +0.00 | 0.176 | 0.420 | 0.2983 | BASELINE |
| - | identity | spec_mean | 12.50 | +0.00 | 0.150 | 0.388 | 0.3321 | BASELINE |

## Promotion policy

- **DROP**: MAE worsens by ≥ worsen margin, or diagnostics degraded (NaN/Inf present, HR-band power collapsed).
- **HOLD**: ≈ baseline, no clear evidence (within improve margin).
- **PROMOTE**: MAE improves by ≥ improve margin AND diagnostics not degraded. Only PROMOTE candidates proceed to cross-dataset validation (PhysDrive / FTU / BGT60).

## Next step

No candidate PROMOTED at current margins. Revisit thresholds or search new candidates; do NOT proceed to NN yet.
