# p03 phase-correlation fusion: local controlled ablations

All experiments ran locally in the copy-based uv environment. No training or Kaggle submission.
Literature inspiration: [Xu et al., 2026, sections 3.2/4.1](https://doi.org/10.3390/s26175587). This is not the complete published pipeline.

The frozen threshold is 0.75. Fusion uses only same-RX local bins and positive correlation weights; no HR labels select bins/weights. All methods use the same 384-frame unwrap/detrend/Hann peak estimator at assumed 30Hz and 0.75–2.5Hz. No CEEMD, harmonic subtraction or time smoothing.

## Component definitions

- single: original fixed or maximum-energy channel.
- fusion: full-phase correlations weight original neighboring phases.
- hrcorr: HR-band-projected correlations weight original phases.
- hrprojected: same hrcorr weights, HR-projected reconstruction.
- single_projected: original single channel with exactly the same HR projection, isolating preprocessing from fusion.

Projection is a finite-window FFT band restriction; it may introduce boundary artifacts. It is a separate preprocessing experiment, not selection-only benefit.

| Distance | Method | Full MAE | Full r | Matched r |
|---|---|---:|---:|---:|
| 0.3m | fixed_single | 4.862 | 0.395 | 0.602 |
| 0.3m | fixed_fusion | 6.625 | 0.130 | -0.145 |
| 0.3m | fixed_hrcorr | 5.571 | 0.251 | 0.134 |
| 0.3m | fixed_hrprojected | 5.674 | 0.244 | 0.154 |
| 0.3m | fixed_single_projected | 4.965 | 0.380 | 0.602 |
| 0.3m | energy_single | 4.516 | 0.382 | 0.452 |
| 0.3m | energy_fusion | 7.531 | -0.050 | -0.337 |
| 0.3m | energy_hrcorr | 4.867 | 0.322 | 0.438 |
| 0.3m | energy_hrprojected | 5.186 | 0.364 | 0.546 |
| 0.3m | energy_single_projected | 4.835 | 0.419 | 0.556 |
| 0.6m | fixed_single | 20.683 | 0.126 | -0.026 |
| 0.6m | fixed_fusion | 20.793 | 0.004 | -0.136 |
| 0.6m | fixed_hrcorr | 20.952 | 0.103 | -0.096 |
| 0.6m | fixed_hrprojected | 19.901 | 0.104 | -0.118 |
| 0.6m | fixed_single_projected | 19.632 | 0.129 | -0.046 |
| 0.6m | energy_single | 17.720 | -0.001 | 0.027 |
| 0.6m | energy_fusion | 18.305 | -0.087 | 0.036 |
| 0.6m | energy_hrcorr | 17.772 | 0.003 | 0.027 |
| 0.6m | energy_hrprojected | 17.149 | 0.001 | 0.036 |
| 0.6m | energy_single_projected | 16.947 | -0.003 | 0.034 |

## Findings

Full-phase fusion worsens MAE under both anchors at both distances. Restricting correlation to the HR band reduces some degradation but does not outperform its original single-channel baseline.
At 0.6 m, projected energy single-channel MAE is 16.947 bpm versus 17.149 for projected fusion; fixed projected single is 19.632 versus 19.901 for projected fusion. The modest changes are attributable to preprocessing rather than an additional fusion benefit. HR tracking correlations remain poor.
At 0.3 m, no fusion variant improves on its original single-channel MAE. High whole-phase correlation is not sufficient evidence of shared heartbeat content; these outputs do not prove respiration is the cause.
Every full-record metric uses 91 reference-valid windows out of 92, and matched offset comparisons use 55 windows. Per-window predictions, all correlation weights, segment metrics and fixed offsets are retained; no window quality exclusion or post hoc threshold tuning.

## Provenance, checks and limitations

The three stages were sequential exploratory experiments, not a pre-registered ten-method competition. Stage compute times were approximately 0.36 s, 0.45 s and 0.57 s (timing.json contains measured values). Six synthetic fusion/projection checks pass. Each stage reproduces the preceding single-channel baseline exactly; array CRC, shapes and finite values are validated.
The archived small NPZ files contain reconstructed window phases and spectra, not the original 24.3 MB complex archives. Original complex input hashes are retained in each stage input_provenance.json. No raw ADC or raw ECG is included.
p03 was chosen after earlier screening. These two distances are not independent subjects. Absolute synchronization, frame rate and metric range calibration remain unverified. No cross-subject effectiveness or clinical performance claim.

## Reproduction

```cmd
setup_local_uv.cmd
uv run --locked --no-sync python validate_bgt60_fusion.py
uv run --locked --no-sync python -m src.data.evaluate_bgt60_fusion --output-dir tmp/bgt60_p03_fusion_results
uv run --locked --no-sync python -m src.data.evaluate_bgt60_fusion --hr-correlation-ablation --output-dir tmp/bgt60_p03_hrcorr_results
uv run --locked --no-sync python -m src.data.evaluate_bgt60_fusion --projected-reconstruction-ablation --output-dir tmp/bgt60_p03_projected_results
uv run --locked --no-sync python summarize_bgt60_fusion.py --output-dir tmp/fusion_archive_reproduction
```

Commands require the previously downloaded paired complex arrays at the default paths. Use --input-dir for alternative locations. Current generator adds explicit protocol flags and method labels absent from early snapshots, so regenerated JSON metadata may differ while original method metrics reproduce.

## Decision

Keep these variants as negative component results. Do not tune the fusion threshold on p03 or resume CycleFormer/domain adaptation based on these MAEs. Independent-subject validation is needed before any positive method claim.