# ADC → RDA physical validation

CPU validation command:

```powershell
python validate_adc_rda_synthetic.py
```

Nine checks passed on 2026-10-10. The validator extracts the production functions
with Python AST to avoid importing optional dataset-loader dependencies. This
checks the conversion arithmetic, not loader integration or ECG alignment.

## Findings

| Check | Result |
| --- | --- |
| Historical frame formula | Default helper output exactly matches the previous FFT formula |
| Chirps after index 7 | A signal supported only after index 7 vanishes in legacy FFT and survives full FFT |
| Known range/Doppler | Native-grid range bin 12 and Doppler bin +2 are recovered |
| Known array phase | Uniform half-wavelength array with spatial frequency 0.25 peaks at angle FFT index 12 |
| Cross-frame micro-motion | 0.2 rad sinusoidal phase at 1.2 Hz is recovered within 1e-6 rad; spectral peak is 72 bpm |
| Frame-internal mean subtraction | The same chirp-stationary micro-motion target is suppressed below 1e-5 of the unsuppressed amplitude |
| Odd / padded Doppler dimensions | DC stays at target_doppler // 2 |
| Fixed range ROI | A stronger reflector at bin 25 does not move the requested bin-12 ROI |
| Invalid configuration / empty frames | Rejected with ValueError |

The synthetic target is idealized: micro-motion varies between frames and is
constant within a frame. Suppression in this case establishes a failure mode,
not a measured loss on FTU or BGT60. The angle check assumes a uniform array;
it does not establish that a dataset's RX order or antenna geometry satisfies
this assumption.

## Independent controls

`convert_adc_cube_to_rda` retains historical defaults for reproducibility:

- `doppler_mode="legacy"`: an 8-point FFT truncates longer selected chirp input.
- `doppler_mode="full_fft_crop"`: FFT all selected chirps, zero-pad only if fewer
  than the requested bins, then retain the bins around shifted DC.
- `clutter_mode="chirp_mean"` or `"none"`: independent frame-internal suppression.
- `sampling_mode="legacy"`: historical evenly selected ≤64 chirps and ≤256 ADC
  samples; `"native"` preserves all chirps and contiguous ADC samples.
- `range_center_bin=<int>`: reuse a baseline center for a controlled comparison.
  The selected range grid must be identical when reusing this integer.

Full FFT cropping retains only low Doppler frequencies. It is not a resampling
of the whole velocity spectrum. FFT length also changes amplitude scaling;
frozen-weight comparisons measure the combined frontend distribution shift.
Legacy nonuniform index selection does not provide a single exact chirp/sample
interval, so its FFT bins should not be assigned native physical coordinates.

FTU and BGT60 exports accept these controls and save them in each sample's JSON:

```powershell
python -m src.data.loaders.export_all_datasets --ftu --bgt60 --dataset-root V:/ --output-dir exports/full_fft_no_chirp_mean --adc-doppler-mode full_fft_crop --adc-clutter-mode none --adc-sampling-mode legacy
```

Use distinct output directories for each configuration. PhysDrive's existing
RDA alignment is unchanged; these ADC options do not operate on its supplied
RDA cubes.

## Real-data comparison protocol (pending)

1. Freeze sample tags, subject/session splits, window offsets, labels, frame
   rates, representation settings and checkpoint hashes before evaluation.
2. With identical sampling, run legacy/chirp_mean, full_fft_crop/chirp_mean,
   legacy/none and full_fft_crop/none. Reuse the baseline's per-sample range
   center through the conversion API for the fixed-ROI experiment.
3. Separately test automatic ROI selection and native sampling. Do not combine
   these into a claim about Doppler alone.
4. Verify reference timestamps and frame rates from loader metadata, then run
   the representation validator and frozen-checkpoint inference. Report paired
   errors, prediction standard deviation, Pearson r, per-subject/session errors
   and a training-label constant baseline.
5. Run any subsequent training only on Kaggle with identical budgets and seeds.

No raw ADC arrays or model checkpoints were found in the workspace file scan.
Consequently no real-data MAE improvement, ECG alignment validation or
frozen-weight result is claimed here. Synthetic validation is the completed P0
conversion check; the real-data comparison remains pending.
