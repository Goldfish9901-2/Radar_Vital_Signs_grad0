# Unified Benchmark Report — FTU → FTU (this study 2026-07-26)

Includes the 5 newly integrated backbones trained under the **same 80-epoch FTU→FTU protocol** as the June baselines (lr=3e-4, AdamW, SmoothL1Loss β=0.5, seed=42, dual time+freq). Existing June / TSLANet / HeartTimeMixer rows are carried over for context.

## Ranked results (by MAE)

| Rank | Model | Prov | Ep | Params | MAE | RMSE | Pearson r | ≤5% | ≤10% | Latency ms/b | GPU MB | Best ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CycleFormer v3-small | reproduced_june_2026-06-23 | 80 | — | 3.43 | 4.31 | 0.17 | 76.85 | 97.73 | — | — | — |
| 2 | TSLANet (official) | this_study_2026-07-25 | 20 | — | 3.69 | 4.76 | 0.11 | 69.32 | 96.16 | — | — | — |
| 3 | HeartTimeMixer | heart_timemixer_june_direct_transfer_ftu_source | n/a | — | 3.69 | 4.80 | -0.05 | 70.31 | — | — | — | — |
| 4 | ContiFormer | this_study_2026-07-26 | 80 | 176145 | 3.76 | 4.70 | 0.16 | 71.02 | 97.02 | 101.66 | 1191.40 | 16 |
| 5 | TCN | reproduced_june_2026-06-23 | 80 | — | 3.87 | 4.95 | 0.15 | 69.03 | 96.45 | — | — | — |
| 6 | DLinear | this_study_2026-07-26 | 80 | 4019 | 3.92 | 5.04 | -0.07 | 69.03 | 94.74 | 0.56 | 10.90 | 1 |
| 7 | Transformer | reproduced_june_2026-06-23 | 80 | — | 4.03 | 5.13 | 0.33 | 66.34 | 95.17 | — | — | — |
| 8 | CycleFormer v3 | reproduced_june_2026-06-23 | 80 | — | 4.19 | 5.41 | -0.20 | 66.76 | 91.62 | — | — | — |
| 9 | Mamba | this_study_2026-07-26 | 80 | 186113 | 4.70 | 5.95 | 0.02 | 59.66 | 90.62 | 77.69 | 449.20 | 8 |
| 10 | CycleFormer (v1) | reproduced_june_2026-06-23 | 80 | — | 4.99 | 6.06 | 0.33 | 55.40 | 91.90 | — | — | — |
| 11 | TimesNet | reproduced_june_2026-06-23 | 80 | — | 5.14 | 6.34 | 0.09 | 53.55 | 89.20 | — | — | — |
| 12 | NLinear | this_study_2026-07-26 | 80 | 1342 | 5.24 | 6.38 | 0.01 | 53.69 | 86.22 | 0.43 | 10.40 | 15 |
| 13 | TSMixer | this_study_2026-07-26 | 80 | 225029 | 5.25 | 6.58 | -0.01 | 56.25 | 84.94 | 7.42 | 30.50 | 3 |
| 14 | PatchTST | reproduced_june_2026-06-23 | 80 | — | 5.75 | 6.84 | 0.07 | 48.01 | 86.79 | — | — | — |

## Per-sample plots (new backbones)

Scatter (Pearson), error histogram, and prediction-vs-label tracking are now generated from `predictions_test.csv` for the 5 new models (`figures/scatter_pearson.png`, `hist_error.png`, `pred_vs_label.png`). The June/TSLANet/HeartTimeMixer rows still lack per-sample files, so their per-sample plots remain unavailable.

## Training stability (new backbones)

`figures/learning_curves.png` shows val (and train, dashed) MAE per epoch. Best epoch and post-best degradation are in the table above.

## Notes

- Batch size was adjusted only where the 4 GB GPU would OOM: Mamba=16, ContiFormer=32; DLinear/NLinear/TSMixer=64. This affects throughput, not the model or final quality.
- HeartTimeMixer is a collapsed near-constant predictor (Pearson ≈ 0) and is kept only as a June-era reference; it should not be read as a competitive baseline.
- All new models are epoch-matched (80) to the June baselines; TSLANet (20 ep) remains the only epoch-mismatched entry.

