# Spectral attribution and eight-subject baseline audit

Local uv only; no estimator changes, training or Kaggle submission. All thresholds and candidate peak rules were fixed before computing attribution.

## Spectral attribution

Fixed p03 bin/RX and zero-Doppler unwrap are reused. Respiration is estimated by peak search in 0.1–0.6 Hz; HR in 0.75–2.5 Hz. Top three peaks include explicitly flagged band endpoints. Harmonic orders 2–6 and one-bin tolerance (4.6875 bpm) are fixed. Reference HR/RR only audit the radar-derived peaks; no label chooses a channel or peak.

At 0.6 m, 71.9% of wrong estimates match radar-derived respiratory multiples, but 68.8% also match a half-order-shifted comb. Only 31.2% match multiples derived from reference RR. These coincidences do not establish a respiratory origin; mask coverage and coarse respiration estimates are documented.
At 0.3 m, 83.6% of accurate windows match estimated harmonics, and 82.2% have reference HR near those harmonics. Blind harmonic removal could suppress useful cardiac components. In 0.6 m wrong windows, reference HR is near a top-three candidate in 53.1%, but that is label-assisted diagnosis, not oracle estimator accuracy.

## Candidate-peak negative controls

| Record | Matched windows | Top3 HR proximity at 0 s | -60 s | +60 s | -120 s | +120 s |
|---|---:|---:|---:|---:|---:|---:|
| 0.3m | 55 | 0.891 | 0.745 | 0.709 | 0.727 | 0.782 |
| 0.6m | 55 | 0.600 | 0.545 | 0.600 | 0.564 | 0.545 |

0.6 m top3 hit rate is unchanged at +60 s, so candidate availability alone is weak evidence for synchronous HR. The nonoverlapping matched subset also lacks a zero-offset advantage at 0.6 m. Slow HR and a coarse, tolerant three-peak comparison are plausible confounds, not proven mechanisms.
The nonoverlapping and reference-RR-agreement strata are descriptive checks. They do not authorize quality-based window removal; RR agreement uses reference labels. No significance tests treat overlapping windows as independent.

## Eight-subject frozen baseline

The existing 32-record native Range FFT/label-free local peak/whole-record energy RX/zero-Doppler unwrap output already covers all eight subjects. This stage reaudits that actual output, rather than rerunning ADC or presenting it as new independent data. No rule was extended from p03 to an untouched cohort.

| Subject | Valid windows | FFT MAE | Other-seven-subject mean MAE | Pearson r |
|---|---:|---:|---:|---:|
| 1 | 364 | 8.369 | 28.778 | 0.054 |
| 2 | 364 | 33.517 | 5.920 | -0.027 |
| 3 | 364 | 13.915 | 2.811 | 0.242 |
| 4 | 364 | 37.994 | 14.832 | 0.004 |
| 5 | 364 | 21.509 | 3.237 | 0.066 |
| 6 | 364 | 15.896 | 7.089 | 0.026 |
| 7 | 364 | 29.973 | 9.248 | -0.095 |
| 8 | 364 | 31.444 | 8.585 | -0.020 |

Subject macro FFT MAE: 24.077 bpm; constant: 10.062 bpm. Mean within-record r: 0.024 (arithmetic descriptive mean).
The constant uses only reference HR from other seven subjects. Each test subject has 364 valid out of 368 overlapping windows. Full-record totals are 2,912 valid / 2,944 windows, distinct from earlier matched-interior scores.

## Evidence and reproduction

All source hashes, per-window candidates and harmonic energies, candidate control hits, per-record metrics and measured local timings are retained. The two small spectral arrays contain window PSDs, not raw ADC. The large source complex archives remain local and in the prior Kaggle paired output.
Three synthetic spectral checks pass. Fixed HR predictions and reference labels exactly match the prior paired audit. JSON, spectral CRC/shape/finite values and 32-record/eight-subject coverage are checked.

```cmd
setup_local_uv.cmd
uv run --locked --no-sync python validate_bgt60_spectral_attribution.py
uv run --locked --no-sync python analyze_bgt60_spectral_attribution.py
uv run --locked --no-sync python analyze_bgt60_subject_baseline.py
uv run --locked --no-sync python analyze_bgt60_peak_controls.py
uv run --locked --no-sync python summarize_bgt60_spectral_audit.py --output-dir tmp/spectral_archive_reproduction
```

Spectral commands require paired NPZ files at the default input paths. The subject reaudit reads the already committed compressed direct-phase output. Script options permit alternate paths.

## Decision and limits

Do not introduce large decomposition models or tune harmonic suppression using p03 based on this diagnostic alone. Respiratory-harmonic attribution remains unproven; broad constant-baseline failure remains. No CycleFormer or domain adaptation training resumed.
p03 is exploratory and all eight subjects were examined previously. Neither stage is independent validation. The assumed 30 Hz frame rate and shared start remain unverified; the reference is 1 Hz HR/RR, not raw ECG.