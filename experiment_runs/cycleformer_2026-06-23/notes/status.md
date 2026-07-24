# Status

Last updated: 2026-06-23

## Completed

1. Docker container `radar_dev` started and verified.
2. PyTorch in container verified: `2.6.0+cu124`, CUDA available.
3. Smoke tests passed:
   - CycleFormer
   - PatchTST
   - TimesNet
4. Signal-processing baselines completed on test split:
   - FFT: FTU, PhysDrive, BGT60TR13C
   - STFT: FTU, PhysDrive, BGT60TR13C
5. CycleFormer FTU source model trained.
6. CycleFormer Source Only evaluated:
   - FTU -> FTU
   - FTU -> PhysDrive
   - FTU -> BGT60TR13C
7. CycleFormer Pseudo-label Adaptation completed:
   - FTU -> PhysDrive
   - FTU -> BGT60TR13C
8. FTU-source deep baselines completed:
   - TCN
   - Transformer
   - PatchTST
   - TimesNet
9. Deep baseline Source Only evaluation completed on FTU, PhysDrive, and BGT60TR13C.

## Key Results

| Setting | Source | Target | MAE | RMSE | Pearson r | within 5 bpm |
|---|---|---|---:|---:|---:|---:|
| CycleFormer Source Only | FTU | FTU | 4.989 | 6.058 | 0.335 | 55.40 |
| CycleFormer Source Only | FTU | PhysDrive | 10.280 | 12.692 | 0.029 | 35.62 |
| CycleFormer Source Only | FTU | BGT60TR13C | 11.120 | 11.715 | -0.067 | 3.78 |
| CycleFormer Pseudo-label | FTU | PhysDrive | 10.256 | 12.693 | 0.033 | 36.16 |
| CycleFormer Pseudo-label | FTU | BGT60TR13C | 11.089 | 11.685 | -0.039 | 3.78 |

## FTU-source Deep Model Comparison

| Model | Target | MAE | RMSE | Pearson r | within 5 bpm |
|---|---|---:|---:|---:|---:|
| TCN | FTU | 3.866 | 4.948 | 0.152 | 69.03 |
| Transformer | FTU | 4.034 | 5.132 | 0.328 | 66.34 |
| CycleFormer | FTU | 4.989 | 6.058 | 0.335 | 55.40 |
| TimesNet | FTU | 5.144 | 6.341 | 0.088 | 53.55 |
| PatchTST | FTU | 5.753 | 6.838 | 0.072 | 48.01 |
| PatchTST | PhysDrive | 10.238 | 12.669 | 0.089 | 34.54 |
| CycleFormer | PhysDrive | 10.280 | 12.692 | 0.029 | 35.62 |
| Transformer | PhysDrive | 10.298 | 12.742 | 0.021 | 34.01 |
| TimesNet | PhysDrive | 10.502 | 12.909 | -0.028 | 30.65 |
| TCN | PhysDrive | 10.552 | 12.928 | 0.056 | 30.11 |
| TCN | BGT60TR13C | 8.851 | 9.734 | -0.032 | 17.99 |
| Transformer | BGT60TR13C | 9.929 | 10.594 | -0.030 | 7.91 |
| TimesNet | BGT60TR13C | 11.010 | 11.770 | -0.057 | 7.19 |
| CycleFormer | BGT60TR13C | 11.120 | 11.715 | -0.067 | 3.78 |
| PatchTST | BGT60TR13C | 11.387 | 12.090 | -0.040 | 5.76 |

Current observation: the first CycleFormer implementation is runnable but not yet stronger than TCN/Transformer. The next model iteration should improve cycle-token selectivity or fusion before using CycleFormer as the main claimed method.

## Result Files

- Main CSV table: `experiment_runs/cycleformer_2026-06-23/tables/overall_metrics.csv`
- Paper deep comparison CSV: `experiment_runs/cycleformer_2026-06-23/tables/ftu_source_deep_comparison.csv`
- Paper signal baseline CSV: `experiment_runs/cycleformer_2026-06-23/tables/signal_baselines.csv`
- Signal baseline JSON: `experiment_runs/cycleformer_2026-06-23/results/signal_baselines/`
- CycleFormer source model: `experiment_runs/cycleformer_2026-06-23/results/deep_models/cycleformer_ftu_source/`
- Source Only JSON: `experiment_runs/cycleformer_2026-06-23/results/source_only/`
- Pseudo-label adaptation JSON: `experiment_runs/cycleformer_2026-06-23/results/pseudo_adaptation/`

## Next

## CycleFormer Improvement

The improved model uses a local TCN-style predictor plus cycle-token residual correction.

| Model | Macro MAE over FTU/PhysDrive/BGT60TR13C | FTU | PhysDrive | BGT60TR13C |
|---|---:|---:|---:|---:|
| CycleFormer v3-small | 7.069 | 3.425 | 11.164 | 6.616 |
| CycleFormer v3 | 7.740 | 4.194 | 10.365 | 8.662 |
| TCN | 7.756 | 3.866 | 10.552 | 8.851 |
| Transformer | 8.087 | 4.034 | 10.298 | 9.929 |
| CycleFormer original | 8.796 | 4.989 | 10.280 | 11.120 |
| TimesNet | 8.885 | 5.144 | 10.502 | 11.010 |
| PatchTST | 9.126 | 5.753 | 10.238 | 11.387 |

Current best: `cycleformer_v3_small`.

Recommended next step: run pseudo-label adaptation for `cycleformer_v3_small`, then decide whether to include adaptation as a separate result table.
