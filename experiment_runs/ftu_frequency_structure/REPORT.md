# Six-record archived Range/Beam frequency structure diagnostic

The archived DC-only region does not establish the prespecified 5/6-record evidence needed to expand to54 records. Neither candidate passes any record. Three records are **unevaluable under the frozen common-control support**, rather than demonstrated correspondence failures. Training, VMD changes and adaptation remain paused. This is a six-record diagnostic, not a population estimate or proof that heartbeat information is absent.

## Scope and protocol

Protocol committed before new spectra: `bb786e7`; support addendum committed before the complete rerun: `c4a4091`. The first attempt stopped on an empty-support reduction; it did not complete a results report. The disclosed addition is correct-label-only descriptive scoring, not a replacement gate.

Only existing archived complex DC sequences are used: six frozen training-subject records,16 beams ×8 saved ranges,1200 frames at assumed20Hz. No ADC reload/export, additional subjects, model fitting, HR-guided cell/peak selection or offset search. This does not evaluate nonzero Doppler or ranges outside the saved ROI. Calibration is first256 frames; each saved range uses the highest calibration `full_none` mean-amplitude beam, same eight identities across arms. Original three targets and weights are retained for separate fusion analysis. No oracle is computed.

## Time support limitation

p03/p04/p05 references end at59s. Circular wrong-label queries modulo60 enter59--60s, so intersecting all controls and all +0/+1/+2s offsets leaves zero common windows at both lengths. No extrapolation or imputation is used. These records retain spectra, quality checks and correct-label-only descriptive scores. They cannot satisfy the gate. Other records have common counts shown below. The controls' artificial circular wrap is itself a diagnostic limitation; this is not a permutation test or causal proof.

## Frozen expansion gate

Primary512 frames: coverage≥.9; ≥.6 fixed-panel peaks within5bpm; reference-near power ≥.01 above **each** wrong-label control; MAE below **each** control. All criteria at all three offsets, and the same candidate in≥5/6 records. No best offset/cell is chosen.

| arm | evaluable | passes |
| --- | --- | --- |
| full_none | 3 | 0 |
| legacy_none | 3 | 0 |

## Six-record descriptive results, correct labels only

512frames, default +0s; MAE in bpm, eight fixed cells equally weighted. These windows intersect the three correct-label offsets only, so are **not** the control-comparison set. See `descriptive_true_only.csv` for all offsets, both lengths, panel and original fixed fusion. The narrowed band and longer windows differ from the prior256-frame audit; these are not a one-factor comparison with its MAE.

| subject | legacy_mean | legacy_none | full_none |
| --- | --- | --- | --- |
| p02 | 17.892 | 16.766 | 17.016 |
| p03 | NA | 15.793 | 17.021 |
| p04 | NA | 7.620 | 8.123 |
| p05 | 26.728 | 18.814 | 12.746 |
| p07 | 26.214 | 11.639 | 11.522 |
| p09 | 28.215 | 30.998 | 31.916 |

Record-equal means over available scores. Candidates have six scored records; legacy_mean has only four because p03/p04 are unreliable. Coverage averages include all six. These unequal available-record means are **not a paired estimate of improvement**; correlations are descriptive, not population inference.

| arm | kind | records_with_mae | mae | centered_rmse | r | std_ratio | within5 | coverage | near_power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_none | fusion | 6 | 14.361 | 5.797 | -0.458 | 21.434 | 0.324 | 1.000 | 0.128 |
| full_none | panel | 6 | 16.391 | 10.548 | -0.072 | 30.865 | 0.241 | 1.000 | 0.094 |
| legacy_mean | fusion | 4 | 22.030 | 18.351 | 0.013 | 59.156 | 0.123 | 0.667 | 0.036 |
| legacy_mean | panel | 4 | 24.762 | 20.324 | -0.081 | 56.723 | 0.088 | 0.584 | 0.051 |
| legacy_none | fusion | 6 | 13.382 | 5.563 | -0.102 | 20.373 | 0.342 | 1.000 | 0.128 |
| legacy_none | panel | 6 | 16.938 | 11.382 | -0.097 | 32.685 | 0.232 | 1.000 | 0.093 |

Both none candidates cover the entire fixed panel in all six records. Their smaller observed phase increments and improved peak MAE in some records (notably p05/p07) do not establish recovery of HR tracking: within5 fractions remain below the frozen .6 threshold in the evaluable records, and power gains over all wrong-label controls remain below .01. legacy_mean has no reliable panel score for p03/p04; NA is not a large error value or a zero.

## Correct vs shifted labels on identical support

Primary512frames, +0s. `min_power_advantage` is true power minus the largest control power; `min_mae_advantage` is the smallest control MAE minus true MAE. Positive values favor the correct labels; the power gate additionally requires≥.01. All +1/+2s comparisons are saved in `record_metrics.csv`; gate decisions use all three, without choosing the best.

| subject | arm | windows | coverage | mae | within5 | near_power | min_power_advantage | min_mae_advantage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p02 | full_none | 20 | 1.000 | 17.016 | 0.019 | 0.052 | 3.39e-05 | -0.468 |
| p02 | legacy_mean | 20 | 0.631 | 17.892 | 0.108 | 0.075 | 0.003 | -0.690 |
| p02 | legacy_none | 20 | 1.000 | 16.766 | 0.031 | 0.053 | -0.000234 | -0.462 |
| p07 | full_none | 21 | 1.000 | 11.522 | 0.226 | 0.087 | 0.002 | 0.318 |
| p07 | legacy_mean | 21 | 1.000 | 26.214 | 0.060 | 0.036 | -0.008 | -0.318 |
| p07 | legacy_none | 21 | 1.000 | 11.639 | 0.232 | 0.084 | 0.000149 | 0.248 |
| p09 | full_none | 21 | 1.000 | 31.916 | 0.167 | 0.071 | 0.000734 | 0.264 |
| p09 | legacy_mean | 21 | 1.000 | 28.215 | 0.065 | 0.051 | -0.001 | 0.050 |
| p09 | legacy_none | 21 | 1.000 | 30.998 | 0.143 | 0.070 | 8.92e-05 | 0.286 |

Mean absolute true/control label separation, averaged over available record/control combinations; compare with native Fourier bin spacings2.34375/1.5625bpm:

| length | offset | mean_absolute_separation | max_absolute_separation | fraction_within_native |
| --- | --- | --- | --- | --- |
| 512.000 | 0.000 | 1.803 | 2.962 | 0.663 |
| 512.000 | 1.000 | 1.833 | 3.060 | 0.651 |
| 512.000 | 2.000 | 1.864 | 3.166 | 0.643 |
| 768.000 | 0.000 | 1.574 | 1.983 | 0.609 |
| 768.000 | 1.000 | 1.671 | 2.072 | 0.564 |
| 768.000 | 2.000 | 1.759 | 2.146 | 0.530 |

Long-window averages and low reference dynamics can make shifted labels weak controls. A visually nearby ridge alone is insufficient. A failed or unevaluable gate cannot establish absence of weaker HR information.

## Amplitude and phase reliability

Across all6×128 cells, median quality diagnostics below. `amplitude_ratio` is relative to the same full_none cell, correcting the8/64 versus64/64 coherent DC scaling. Dropout is below1% of the scaled full_none calibration median. `near_pi_fraction` concerns observed P2 increments, not measured chest displacement. Raw quality numbers are retained even when phase scoring is rejected.

| arm | amplitude_ratio | dropout | near_pi_fraction | abs_increment_p99 |
| --- | --- | --- | --- | --- |
| full_mean | 9.67868e-11 | 1.000000 | 0.102585 | 3.109608 |
| full_none | 1.000000 | 0.000000 | 0.000000 | 0.085102 |
| legacy_mean | 0.027042 | 0.025000 | 0.000417 | 1.493859 |
| legacy_none | 1.010451 | 0.000000 | 0.000000 | 0.092278 |

full_mean is retained as a numerical-floor mechanism control, not given physiological phase scores. Segment reliability requires finite/nonzero observations, RMS above1e-6 of the scaled reference and≤1% dropout. Invalid segments remainNaN; expected empty-spectrum mean warnings denote unscorable cells, not filled data. Original8×3 target masks were independently reproduced for all24 record/arm combinations. New masks are separately archived.

## Spectral and metric definitions

P2 uses complex128 conjugate principal increments; cumulative phase is detrended independently per segment. Phase origin is removed by detrending, never treated as a discontinuity. Original fusion sums the three detrended phases with unchanged weights; final scalar normalization does not change normalized power. Log-amplitude spectra are separate diagnostics. No HR-guided filtering or peak selection.

Hann windows512/768frames (25.6/38.4s), hop1s, FFT4096 for display. Native bin spacings2.34375/1.5625bpm; display grid0.29296875bpm **does not increase resolving power**, and Hann broadens spectral response. Primary scoring excludes all segments overlapping initial12.8s calibration. Reference labels are mean interpolated HR over the exact segment support, no extrapolation. Positive offset queries reference at radar time+offset; these are assumptions, not measured synchronization.

Power is normalized within45--150bpm. Reference-near fraction integrates ±one native bin. Global band maximum is a descriptive readout, not an assumed heartbeat peak. Primary MAE/near power/within5 use equal-cell means over valid windows; coverage and missingness are explicit. Dynamic RMSE/r/std ratio pool time deviations after removing each cell's temporal mean; they do not imply absolute HR calibration. `ridge_step` reports median adjacent valid peak change; smoothness alone is not correspondence evidence. Overlapping windows are not independent samples; no window-count significance tests are performed.

## Figures and archived evidence

Each record has four PNGs in `figures/`: phase spectra for three arms across the fixed eight cells; full_none phase vs log-amplitude spectra; descriptive all-beam range-frequency maps and128-cell reliability; raw full_none amplitude/P2 increments/unwrapped phase. White curves are referenceHR at+0s, red raw-amplitude lines mark dropout thresholds. Spectral colors are within-band normalized dB (−45 to−10); blank cells are unreliable. Mean maps include the continuous recording, whereas numerical scores use post-calibration windows.

`spectra/*.npz` stores both lengths, frequencies, segment starts, all reference/control labels, common-support masks, fixed cells, original fusion,128-cell validity and mean spectra. `cell_metrics.csv`, `cell_quality.csv`, `panel_selection.csv`, `label_separation.csv` and `gate.csv` provide underlying values. Input archives are referenced by10 verified hashes, not duplicated.

## Execution and independent checks

Local uv diagnostic elapsed **101.45s**; independent validation and raw-trace plots **40.53s**. No training/Kaggle run. Independent unwrap+SciPy linear detrend agrees with conjugate-increment spectra: maximum absolute normalized-power difference **3.62e-09**. Checked 8208 segment reliability masks, all saved labels/common supports, 810 primary metric values, and all gate decisions. Source scripts, hashes and execution versions are archived. The failed initial run is documented above; successful-run timing is in `execution.json`.

## Decision

Do not expand to54 records or restart model training on this evidence. The archived DC region has not shown the required stable reference correspondence, and the frozen shifted-control comparison is unavailable in three records. A future protocol must explicitly solve the finite-reference-support control design before claiming a six-record negative-control conclusion. Continue upstream coherence/complex-signal interpretation and radar/reference time-axis auditing; no conclusion here treats the ADC candidates as a repaired pipeline.
