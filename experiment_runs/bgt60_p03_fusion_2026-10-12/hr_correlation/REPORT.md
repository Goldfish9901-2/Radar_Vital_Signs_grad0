# Phase-correlation fusion ablation (local uv)

Literature-inspired component test, not full Xu 2026 reproduction. No label-based fusion, VMD, harmonic filtering or smoothing.
[Source, sections 3.2 and 4.1](https://doi.org/10.3390/s26175587): positive phase correlation and weighted neighboring-bin aggregation.
All six methods use the same FFT estimator. Base fusion uses full detrended phase correlation; hrcorr uses HR-band correlation only for weights, reconstructing original phase.

| Distance | Method | Valid/total | Full MAE | Full r | Matched r |
|---|---|---:|---:|---:|---:|
| 0.3m | fixed_single | 91/92 | 4.862 | 0.395 | 0.602 |
| 0.3m | fixed_fusion | 91/92 | 6.625 | 0.130 | -0.145 |
| 0.3m | fixed_hrcorr | 91/92 | 5.571 | 0.251 | 0.134 |
| 0.3m | energy_single | 91/92 | 4.516 | 0.382 | 0.452 |
| 0.3m | energy_fusion | 91/92 | 7.531 | -0.050 | -0.337 |
| 0.3m | energy_hrcorr | 91/92 | 4.867 | 0.322 | 0.438 |
| 0.6m | fixed_single | 91/92 | 20.683 | 0.126 | -0.026 |
| 0.6m | fixed_fusion | 91/92 | 20.793 | 0.004 | -0.136 |
| 0.6m | fixed_hrcorr | 91/92 | 20.952 | 0.103 | -0.096 |
| 0.6m | energy_single | 91/92 | 17.720 | -0.001 | 0.027 |
| 0.6m | energy_fusion | 91/92 | 18.305 | -0.087 | 0.036 |
| 0.6m | energy_hrcorr | 91/92 | 17.772 | 0.003 | 0.027 |

All segment and offset controls are in summary.json; all contributing-bin correlations and weights are preserved.
p03 was selected after earlier screening. No cross-subject efficacy claim; frame rate and absolute synchronization remain unverified.

Additional hrcorr methods compute correlation on a fixed HR-band projection, but reconstruct original phase. This is a new exploratory component, not a published-method reproduction. All six methods share the same FFT estimator; no thresholds tuned.
