# Unified Benchmark Report — FTU → FTU (same-dataset, test split)

Assembled from **existing results only**; no model was retrained.

## Data sources & provenance

- **New (this study, 2026-07-25)**: TSLANet (official) (20 epochs, `experiment_runs/tslanet_2026-07-25`).
- **Reproduced from June (2026-06-23)**: CycleFormer (v1), CycleFormer v3, CycleFormer v3-small, PatchTST, TCN, TimesNet, Transformer (80 epochs, `experiment_runs/cycleformer_2026-06-23/tables/overall_metrics.csv`).
- **HeartTimeMixer (June-era)**: NOT present in the June `cycleformer_2026-06-23` tables. Used `work/run/upstream/model_outputs/direct_transfer_ftu_source/eval_test_FTU.json` (same `evaluate_model.py` schema; count=704, label_mean=77.5000502 — identical test set). Epoch count not recorded in its run_config.

## Ranked results (by MAE)

| Rank | Model | Provenance | Epochs | MAE | RMSE | Pearson r | ≤5 bpm % | ≤10 bpm % |
|---|---|---|---|---|---|---|---|---|
| 1 | CycleFormer v3-small | reproduced_june_2026-06-23 | 80 | 3.43 | 4.31 | 0.166 | 76.8 | 97.7 |
| 2 | TSLANet (official) | this_study_2026-07-25 | 20 | 3.69 | 4.76 | 0.114 | 69.3 | 96.2 |
| 3 | HeartTimeMixer | heart_timemixer_june_direct_transfer_ftu_source | n/a | 3.69 | 4.80 | -0.054 | 70.3 | — |
| 4 | TCN | reproduced_june_2026-06-23 | 80 | 3.87 | 4.95 | 0.152 | 69.0 | 96.4 |
| 5 | Transformer | reproduced_june_2026-06-23 | 80 | 4.03 | 5.13 | 0.328 | 66.3 | 95.2 |
| 6 | CycleFormer v3 | reproduced_june_2026-06-23 | 80 | 4.19 | 5.41 | -0.197 | 66.8 | 91.6 |
| 7 | CycleFormer (v1) | reproduced_june_2026-06-23 | 80 | 4.99 | 6.06 | 0.335 | 55.4 | 91.9 |
| 8 | TimesNet | reproduced_june_2026-06-23 | 80 | 5.14 | 6.34 | 0.088 | 53.6 | 89.2 |
| 9 | PatchTST | reproduced_june_2026-06-23 | 80 | 5.75 | 6.84 | 0.072 | 48.0 | 86.8 |

## Per-sample plots: intentionally omitted

Pearson scatter, error histogram, and prediction-vs-label plots are **not produced**. No eval artifact in this checkout stores per-sample `(pred, label)` arrays for any model, so these charts cannot be reproduced without retraining. They are explicitly omitted rather than fabricated.

## Note on epoch budgets

TSLANet official (20 epochs) is compared against June backbones (80 epochs) on the same FTU→FTU test set. TSLANet official ranks #2 by MAE, just behind CycleFormer v3-small (80 ep) and essentially tied with HeartTimeMixer (MAE 3.69 vs 3.69). A strictly epoch-matched comparison would require a unified re-run (deferred).
