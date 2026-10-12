# Phase-correlation fusion ablation (local uv)

Literature-inspired component test, not full Xu 2026 reproduction. No label-based fusion, VMD, harmonic filtering or smoothing.
[Source, sections 3.2 and 4.1](https://doi.org/10.3390/s26175587): positive phase correlation and weighted neighboring-bin aggregation.
Only fusion changes; all four methods use the same FFT estimator. Correlation is computed on full detrended phase, which may emphasize respiration.

| Distance | Method | Valid/total | Full MAE | Full r | Matched r |
|---|---|---:|---:|---:|---:|
| 0.3m | fixed_single | 91/92 | 4.862 | 0.395 | 0.602 |
| 0.3m | fixed_fusion | 91/92 | 6.625 | 0.130 | -0.145 |
| 0.3m | energy_single | 91/92 | 4.516 | 0.382 | 0.452 |
| 0.3m | energy_fusion | 91/92 | 7.531 | -0.050 | -0.337 |
| 0.6m | fixed_single | 91/92 | 20.683 | 0.126 | -0.026 |
| 0.6m | fixed_fusion | 91/92 | 20.793 | 0.004 | -0.136 |
| 0.6m | energy_single | 91/92 | 17.720 | -0.001 | 0.027 |
| 0.6m | energy_fusion | 91/92 | 18.305 | -0.087 | 0.036 |

All segment and offset controls are in summary.json; all contributing-bin correlations and weights are preserved.
p03 was selected after earlier screening. No cross-subject efficacy claim; frame rate and absolute synchronization remain unverified.

## Outcome

The single-channel full-record metrics exactly reproduce the preceding fixed and
energy-selection experiment. All four methods retain 91 valid reference windows
out of 92, so fusion changes are not explained by removing difficult windows.
Four fusion tests and three prior selection tests pass.

Fusion increases full-record MAE under both anchor policies at both distances.
At 0.3 m it substantially degrades tracking: fixed-anchor matched r changes from
0.602 to -0.145; energy-anchor matched r changes from 0.452 to -0.337. The
full-record fixed fusion MAE increases by 1.762 bpm and energy fusion by 3.015 bpm.
At 0.6 m increases are 0.110 and 0.585 bpm, with poor full-record correlations.

Fixed fusion averages 4.02 contributors at 0.3 m, versus 1.42 at 0.6 m. At 0.6 m,
66.3% of fixed-fusion windows retain only the anchor. Thus fusion does execute
substantially at 0.3 m; its failure is not merely universal single-bin fallback.

The experiment does not identify whether respiration-dominated correlation,
phase noise or different scattering components explain the degradation. It tests
one frozen simplified component on p03; it does not refute the complete published
method, which includes different preprocessing and later interference removal.
Do not adjust the threshold to these HR labels or resume NN training from this
result. Full-window phase correlation alone is not a validated HR quality score.
