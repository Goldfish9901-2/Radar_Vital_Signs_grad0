# RDA Front-End Algorithm Pool

> **Status (2026-09):** The NN / backbone exploration is **closed** (see
> `experiments/mmwave_selection/mmwave_nn_bench.json` — no backbone beats the 12.23
> linear probe under the pre-registered rule). The active research axis is now
> **RDA conversion → representation information ceiling**. This module is the
> machinery for that axis: a *replaceable pool* of candidate algorithms per RDA
> stage, instead of a single `convert_adc_to_rda_vN`.

---

## 1. Design principle

Do **not** hunt for "one论文算法 to rule them all". Instead:

- Each RDA processing step is a *stage* with several *candidate methods*, each
  registered by name.
- A single `RDAConfig` selects one method per stage. Swapping a candidate is a
  **one-line change** (`clutter="mti"` → `clutter="pca"`), not a new code path.
- We then run the candidate through the existing
  `representation validator → linear probe → (only if promising) backbone` chain
  to decide whether it actually raised the information ceiling.

This keeps the RDA work causally linked to the already-built infrastructure, and
means a negative result ("candidate X didn't move the probe") costs almost nothing.

---

## 2. Stages and the shape contract

```
ADC ── Range FFT ──┬─ clutter        (C0..C4)
                   ├─ localization   (L0..L3)
                   ├─ range_selection(R0..R3)
                   ├─ beamforming     (B0..B2)
                   │        │
                   │        ▼  complex RDA cube
                   └─ phase          (P0..P2)  ──► 1-D phase trace
                            │
                            ▼
                   EDACM / HR-AdaVMD  (representation layer — NOT owned here)
```

**Cube shape contract** `(F, D, A, R)` — `complex64`:

| axis | meaning | PhysDrive | FTU (raw ADC) | mmwave-897 |
|------|---------|-----------|---------------|------------|
| F | frames (time) | 600 | 1200 | 400–600 |
| D | doppler bins | 8 | — (needs Doppler FFT) | **none → 1** |
| A | angle / RX-spatial | 16 | 4 (RX) | 8 (virtual ant.) |
| R | range bins | 8 | 250 (ADC→Range FFT) | 64 |

- **PhysDrive** already arrives as `(600, 8, 16, 8)` — it matches the contract
  exactly and is the **primary research target** (worst bottleneck: MAE ≈ 10.75 vs
  a ≈ 10.94 constant baseline).
- **mmwave-897** arrives as `(F, 8, 64)` — *no Doppler axis* (chirps were averaged
  at storage). Use `ensure_rda_4d(cube)` (or just pass it to `apply_rda_pipeline`,
  which inserts `D=1` automatically) → `(F, 1, 8, 64)`. mmwave-897 is an
  **external reference**, not the ADC-vs-RDA testbed: use it to check that a new
  RDA idea generalizes, not as proof that ADC→RDA itself improved.
- **FTU / BGT60** are the true ADC→RDA targets (raw ADC; need Range + Doppler FFT
  before they enter this pool).

**Layer boundary (important):** the four *cube* stages
(`clutter, localization, range_selection, beamforming`) each return a cube of the
**same** `(F, D, A, R)` shape (enforced in `apply_rda_pipeline` — a stage that
changes shape raises `ValueError`). The *phase* stage is **cross-layer**: it
consumes the cube and returns a 1-D `(F,)` phase trace. It is part of the search
pool (so we can test whether EDACM excluded a simpler, stabler phase
representation) but is **not** the EDACM implementation — EDACM lives in
`src/data/build_training_dataset.py` (`edacm_phase`, `target_edacm_signal`,
`hr_adavmd_decompose`).

---

## 3. Candidate matrix

| ID | stage | method (registry name) | input→output | labels | complexity | notes |
|----|-------|------------------------|--------------|--------|------------|-------|
| C0 | clutter | `none` | cube→cube | no | — | baseline, cube assumed already static-removed |
| C1 | clutter | `mean_subtraction` | cube→cube | no | low | zero slow-time mean per cell |
| C2 | clutter | `temporal_highpass` | cube→cube | no | low | 1-pole IIR HP; `cutoff` Hz (uses `fs`) |
| C3 | clutter | `mti` | cube→cube | no | low | 2-pulse frame difference (zero-Doppler cancel) |
| C4 | clutter | `pca` | cube→cube | no | medium | drop top slow-time PCA mode |
| C5 | clutter | `rpca` | cube→cube | no | medium | low-rank + sparse (inexact ALM) |
| L0 | localization | `energy` | cube→cube | no | low | global energy-max → soft Gaussian window |
| L1 | localization | `spatial_consistency` | cube→cube | no | medium | per-frame peak, temporally-consistent centroid |
| L2 | localization | `hr_band` | cube→cube | no | medium | **HR-band phase-energy max** (fixes PhysDrive trap) |
| L3 | localization | `temporal_consistency` | cube→cube | no | medium | track loudest cell across frames |
| R0 | range_selection | `global_energy` | cube→cube | no | low | current baseline: global energy center ± `half_width` |
| R1 | range_selection | `peak` | cube→cube | no | low | single peak-energy range |
| R2 | range_selection | `weighted_center` | cube→cube | no | low | energy-weighted centroid |
| R3 | range_selection | `hr_band` | cube→cube | no | low | **HR-band phase-energy center** (the important one) |
| B0 | beamforming | `fft` | cube→cube | no | low | identity (baseline) |
| B1 | beamforming | `bartlett` | cube→cube | no | medium | conventional / Bartlett steered filter |
| B2 | beamforming | `mvdr` | cube→cube | no | medium | MVDR / Capon adaptive (suppresses incoherent clutter) |
| P0 | phase | `p0_single` | cube→trace | no | very low | arctan phase of one bin, unwrapped + detrended |
| P1 | phase | `p1_diff` | cube→trace | no | very low | inter-frame phase diff (conjugate multiplication) |
| P2 | phase | `p2_fusion` | cube→trace | no | low | equal-weight fusion over `top_bins` bins |

Adding a new candidate = implement `f(cube, **kwargs) -> cube` (or `-> trace` for
phase) and register it in the stage module's `register_*_methods()`; no runner
changes required.

---

## 4. `fs` (frame rate) handling

`fs` is a **dataset property**, never a per-stage constant. `RDAConfig.fs`
(default `20.0`, matching FTU / the codebase-wide `DEFAULT_SAMPLING_RATE_HZ`)
is injected into every cube stage that accepts an `fs` keyword. A per-stage
override (`RDAConfig.clutter_kwargs = {"fs": 5.0}`) wins over the config-wide
value.

- FTU / codebase convention → `fs=20.0`
- **mmwave-897 → `fs=10.0`** (set `RDAConfig(fs=10.0)`)

The old code hard-coded `20.0` inside `clutter_temporal_highpass`,
`localization_hr_band`, `localization_temporal_consistency`, and
`range_selection._range_profile`; all now read `fs` from the config.

---

## 5. Why localization / range-selection are the priority

PhysDrive: `best MAE ≈ 10.75` vs `constant baseline ≈ 10.94`. The model barely
beats guessing → the **spatial region was almost certainly mis-localized**
upstream. Seat frames, clothing, and car structure dominate *total* amplitude but
carry **no heartbeat micro-motion**. So:

> **total energy max ≠ vital-sign-bearing max.**

`L2` (HR-band localization) and `R3` (HR-band range selection) directly attack
this by localizing on *HR-band phase energy* instead of total magnitude. These
two are the first candidates to run through the validator.

---

## 6. The diagnostic gate

The point is **not** "which RDA has the lowest MAE". It is "**did we raise the
representation information ceiling?**" The gate, reuse-ready from the closed
NN line:

```
RDA candidate
   → EDACM + HR-AdaVMD  (representation; src/data/build_training_dataset.py)
   → mmwave_probe.py    (Ridge linear-probe validator, subject-level GroupKFold(5))
   → linear probe (Ridge on 71-d bandpass-phase spectrum)
```

**Decision rule:** a candidate is worth promoting to the NN/backbone stage only
if the **Ridge linear-probe MAE drops below the 12.23 reference** (and, for the
PhysDrive split specifically, below ~10.94). Until then, stay in the CPU /
representation-validator phase — do **not** burn GPU on a backbone search over a
cube that still loses the heartbeat to static clutter.

Phase 0 (no training, CPU-only) sanity still applies per candidate: range
profile, Doppler profile, angle profile, HR-band energy, spectral SNR, phase
variance, phase continuity.

> Note: a more elaborate `experiments/.../validate_representation.py` and a
> dedicated `src/features/edacm.py` exist in `audit/` (pre-promotion prototypes).
> In the active repo the validator is `src/data/mmwave_probe.py` and EDACM is in
> `src/data/build_training_dataset.py`.

---

## 7. Usage

```python
from src.radar import RDAConfig, apply_rda_pipeline, list_stage_methods

# inspect the pool
list_stage_methods()

# mmwave-897 (10 Hz, 3-D cube -> D=1 inserted automatically)
cfg = RDAConfig(
    clutter="mti",
    localization="hr_band",     # L2: HR-band localization
    range_selection="hr_band",  # R3: HR-band range
    beamforming="mvdr",         # B2
    fs=10.0,                    # mmwave-897 frame rate
)
cube = loader.load_radar_data(...)          # (F, 8, 64) complex64
out  = apply_rda_pipeline(cube, cfg)        # (F, 1, 8, 64) complex64

# cross-layer phase trace (P1), no EDACM
trace = apply_rda_pipeline(
    cube,
    RDAConfig(localization="energy", range_selection="global_energy",
              phase="p1_diff", fs=10.0),
    return_phase=True,
)  # (F,)
```

Run order recommended by `AGENTS.md` §8–§11:

1. **Phase 0** — CPU diagnostic profiles per candidate.
2. **Phase 1** — representation validator + linear probe (CPU/GPU-light).
3. **Phase 2** — only the best 2–3 candidates × one backbone (e.g. CycleFormer) on
   PhysDrive/FTU/BGT60.
4. **Factorial** — only after a candidate clearly moved the ceiling (e.g.
   PhysDrive 10.75 → 7.8) do you multiply by the 4-backbone grid.

---

## 8. Files

| file | responsibility |
|------|----------------|
| `src/radar/config.py` | `RDAConfig` dataclass + `StageNotFoundError` |
| `src/radar/pipeline.py` | `RDA_STAGE_REGISTRY`, `STAGE_ORDER`, `apply_rda_pipeline`, `ensure_rda_4d`, `DEFAULT_FS` |
| `src/radar/clutter.py` | C0–C4 |
| `src/radar/localization.py` | L0–L3 (+ `_gaussian_mask`) |
| `src/radar/range_selection.py` | R0–R3 |
| `src/radar/beamforming.py` | B0–B2 |
| `src/radar/phase.py` | P0–P2 (cross-layer) |
| `src/radar/__init__.py` | registers all candidates on import |

These were promoted from the `audit/` prototype into the active repo during the
2026-09 RDA-pool scaffold (option "a"): migrate + reconcile `fs` to a config field
and add the 3-D→4-D adapter for mmwave-897.

---

## 9. Round-1 screening results (mmwave-897, full 440 sessions)

Harness: `experiments/mmwave_selection/phase0_rda_screen.py`, CPU-only, run on
Kaggle. 110 participants × 4 sessions; one stage swapped at a time; downstream is
the canonical 71-d log1p HR-band spectrum + subject-level `GroupKFold(5)` Ridge
with inner alpha CV (identical to `mmwave_probe`, whose full-440 reference is
**MAE 12.23 / R² 0.176** for `single_bin`). fs = 10 Hz. Margins: PROMOTE ≤ −0.5
BPM, DROP ≥ +0.5 BPM.

| Stage | Candidate | MAE | ΔMAE | R² | r | Status |
|-------|-----------|----:|-----:|---:|---:|--------|
| — | baseline_cube (identity → `single_bin`) | 12.41 | — | 0.156 | — | BASELINE |
| — | baseline_phase (canonical single-bin trace) | **12.23** | — | 0.176 | — | BASELINE |
| C | mean_subtraction | 12.46 | +0.05 | 0.144 | 0.380 | HOLD |
| C | mti | 12.56 | +0.15 | 0.138 | 0.371 | HOLD |
| C | pca | 12.44 | +0.03 | 0.165 | 0.406 | HOLD |
| C | rpca | 12.55 | +0.14 | 0.127 | 0.358 | HOLD |
| C | temporal_highpass | 12.39 | −0.02 | 0.148 | 0.385 | HOLD |
| L | hr_band | 12.34 | −0.07 | 0.157 | 0.397 | HOLD |
| L | spatial_consistency | 12.58 | +0.17 | 0.138 | 0.372 | HOLD |
| L | temporal_consistency | 12.57 | +0.16 | 0.136 | 0.369 | HOLD |
| R | hr_band | 12.41 | +0.00 | 0.156 | 0.396 | **invalid (no-op)** |
| R | peak | 12.41 | +0.00 | 0.156 | 0.396 | **invalid (no-op)** |
| R | weighted_center | 12.41 | +0.00 | 0.156 | 0.396 | **invalid (no-op)** |
| B | bartlett | 12.38 | −0.03 | 0.157 | 0.396 | HOLD |
| B | mvdr | 12.44 | +0.03 | 0.152 | 0.390 | HOLD |
| P | p0_single | 14.11 | +1.88 | 0.008 | 0.116 | DROP |
| P | p1_diff | 14.23 | +2.00 | 0.003 | 0.096 | DROP |
| P | p2_fusion | — | — | — | — | crashed (see below) |

### Reading of round 1

**Every single-stage swap is flat at 12.4 ± 0.2 BPM.** The whole candidate pool
moves MAE by less than the ±0.5 promotion margin, and the best candidate
(`C/temporal_highpass`, 12.39) still does not beat the 12.23 baseline. This is the
second null result in a row (the first was the 6-backbone benchmark, which also
failed to beat the 12.23 linear probe). Working hypothesis: on mmwave-897 the
ceiling is set by the *representation* (one z-scored HR-band spectrum per session
→ Ridge), not by which classical filter/beamformer/localizer sits in front of it.

### Three harness defects found by round 1 (fixed for round 2)

1. **`P/p2_fusion` crashed** — `src/radar/phase.py` iterated
   `np.unravel_index(...)` directly, which yields one array *per axis*, so the
   `(d, a, r)` unpack read axis-index arrays and indexed `cube[:, 5, ...]` on a
   `D=1` cube. Fixed by `zip(*np.unravel_index(...))`.
2. **The R (range-selection) screen was a provable no-op.** Both
   `build_features(rep='single_bin')` and `single_bin_phase_trace` keep only the
   single argmax-energy range bin. `range_selection` implements selection by
   *zeroing* rejected bins, and the argmax bin always lies inside every candidate
   window → the trace is bit-identical → all three R rows are the same number.
   Conclusion: **you cannot study range selection through a single-bin
   representation.** Round 2 re-screens R with `--rep spec_mean` (which averages
   the per-bin normalized spectra, so the mask actually changes the feature).
3. **The P (phase) comparison was confounded with the band-pass.** The baseline
   trace is `unwrap → butter_bandpass(HR band)`, while P0/P1 were fed to the
   71-d spectrum unfiltered, so respiration dominated the normalized spectrum.
   The ~1.9 BPM gap is mostly "band-pass or not", not "which phase extraction".
   Round 2 re-screens P with `--phase-bp` (pool traces band-passed too).

### Round-2 re-screen (after fixing the three defects)

Two targeted passes (kernel v7, 8 min): `--stages range_selection --rep spec_mean`
and `--stages phase --phase-bp`.

R under `spec_mean` (baseline `spec_mean` cube = 12.50 / R² 0.150) — the mask
now actually changes the feature, and the candidates separate for the first time:

| Candidate | MAE | ΔMAE | R² | r |
|-----------|----:|-----:|---:|---:|
| R/hr_band | **12.35** | **−0.15** | 0.165 | 0.406 |
| R/weighted_center | 12.50 | +0.00 | 0.150 | 0.388 |
| R/peak | 12.55 | +0.15 | 0.141 | 0.375 |

The ordering is physically sensible (HR-band-weighted > energy-weighted centre >
single peak) and `hr_band` is the largest improvement anywhere in the pool, but
−0.15 BPM is still far below the ±0.5 promotion margin → HOLD, not PROMOTE.

P with the band-pass confound removed (baseline phase trace = 12.23):

| Candidate | MAE | ΔMAE | R² | r |
|-----------|----:|-----:|---:|---:|
| P/p0_single | 12.67 | +0.44 | 0.119 | 0.346 |
| P/p2_fusion | 12.75 | +0.52 | 0.121 | 0.348 |
| P/p1_diff | 12.96 | +0.73 | 0.098 | 0.315 |

Round 1's +1.88 / +2.00 was mostly the missing band-pass; the honest gap is
+0.44…+0.73. All three still lose to the canonical single-bin trace, i.e. **no
pool phase extractor beats the current EDACM-style path**.

### Where the ceiling actually is (the real reason round 1/2 are flat)

Two earlier measurements already localised the bottleneck, and round 1/2 confirm
that no classical swap touches it:

* `experiments/mmwave_baseline/summary.json` (P001–P010, paper Fig.5 FFT-peak
  baseline): **oracle range bin MAE 0.18 / 0.43 / 0.25 / 1.32 BPM** vs
  max-energy bin 2.97 / 12.46 / 11.55 / 28.09 BPM (Lying/Rest, Lying/Post-ex,
  Sitting/Rest, Sitting/Post-ex). The HR is fully present in the data.
* `mmwave_split_half_results.json` (440 sessions): oracle over all bins 1.64 BPM,
  but pick the bin on the first half and evaluate on the second → **18.32 BPM**,
  good-bin Jaccard 0.08.

So the ~12.4 plateau is a **range-bin selection** plateau, not a filter or
backbone plateau. Next step is `experiments/mmwave_selection/phase0_bin_oracle_diag.py`,
which measures, for every bin, the FFT-peak HR error against ECG plus a battery
of *unsupervised* bin-quality scores, and reports the honest MAE of "pick the top
bin by score X" next to the oracle. That decides whether hand-designed bin
selection can be fixed at all, or whether the search must move to learned /
time-varying selection.

### Round-3: where the 12.23 baseline actually reads from

`experiments/mmwave_selection/phase0_bin_oracle_diag.py`, 440 sessions × 62 range
bins, classical FFT-peak HR per bin, ECG used only to *measure* error:

| rule | MAE | median | %<3 BPM | %<3 (harmonic-tolerant) |
|---|---:|---:|---:|---:|
| oracle (min error over bins) | **1.64** | 0.29 | **90.7** | 90.7 |
| oracle, target replaced by uniform random draw | 10.19 | — | 58.0 | — |
| PC1 of the HR-band bin ensemble | 15.31 | 9.94 | 22.7 | 23.2 |
| **max energy / probe `single_bin` (bin 2)** | **15.74** | 9.53 | 29.5 | 30.7 |
| best single score (`spec_entropy`) | 16.12 | 11.28 | 24.8 | 25.0 |
| worst (`band_frac`) | 21.12 | 15.10 | 14.3 | 15.9 |

Three facts that reframe everything above:

1. **The canonical pipeline always reads bin 2 (0.63 m).** Both energy criteria
   (`E|z|²` and probe's `(E|z|)²`) give argmax ∈ {2, 3} in 100% of sessions.
   The oracle bin is somewhere else entirely (median 27, range 2–63). This is why
   every `range_selection` candidate was a bit-identical no-op under
   `--rep single_bin`: all of their keep-windows still contain bin 2.
2. **The oracle is real, not selection overfitting.** Re-running the same
   min-over-62-bins selection against a uniformly random HR target gives 10.19 /
   58%, versus 1.64 / 90.7% against the true HR — a 6× gap. There genuinely is a
   good bin per session; we simply have no way to find it without labels.
3. **No unsupervised rule even matches "always use bin 2".** Every one of the 15
   scores lands at 15.3–21.1 versus 15.74 for bin 2, all Spearman
   |ρ(score, −err)| ≤ 0.30 (best: `temp_cont` +0.296, `spec_entropy` +0.294), and
   several are *negative* (`hr_power` −0.174: bins with more HR-band power are
   worse). Decile separation is ~1–2 BPM at the extremes. Harmonic tolerance moves
   the hit rate 29.5% → 30.7%, so the failures are not "right bin, wrong harmonic".

Combined with the earlier split-half result (pick the bin on half A, evaluate on
half B → 18.32 BPM, i.e. worse than just using bin 2): the good bin is **not
stable in time**, and per-bin FFT-peak estimation is close to a coin flip outside
~30% of easy sessions. See §10 for what this implies.

### Validity limits (do not over-read round 1)

* mmwave-897 is `(F, 8, 64)` with **no Doppler axis**; `D=1` is a compatibility
  adapter. Anything Doppler-flavoured (notably B/MVDR) is under-tested here.
* The screen is single-stage-at-a-time by design. A null here does not rule out
  *combinations* (e.g. clutter × HR-band localization), only that no single
  classical swap is a free win.
* All numbers are Ridge-on-71-d-spectrum, i.e. they measure the *linear*
  information ceiling. A candidate could still help a non-linear model.

---

## 10. Open question: how much headroom does bin selection have?

Everything above is measured with the FFT-peak estimator per bin. The project's
go/no-go metric is different — the subject-level `GroupKFold(5)` Ridge on the 71-d
log1p HR-band spectrum (12.23 for `single_bin`). A spatial combination can be
useless for picking one spectral peak and still denoise the spectrum the Ridge
reads, so the two questions have to be measured separately:

1. **Headroom**: feed the *oracle* bin's spectrum to the Ridge. This is a
   supervised upper bound (not a method): if it still lands near 12.23, perfect
   range-bin selection buys nothing and the plateau is intrinsic to the
   representation. If it lands much lower, bin selection has headroom worth
   chasing with a learned selector.
2. **Unsupervised spatial combination**: Ridge on the spectrum of PC1 / PC2 of the
   HR-band bin ensemble, and on the mean HR-band trace — data-driven spatial
   filters instead of picking one bin.

Both are measured by `experiments/mmwave_selection/phase0_rep_probe.py` under the
exact same CV. Decision rule: only a representation that moves Ridge from
**12.23 → ~10.x** counts as progress; anything inside ±0.5 is noise.

**Answer (measured):** perfect bin selection buys nothing.

| representation | MAE | Δ vs `single_bin` | R² | r |
|---|---:|---:|---:|---:|
| `single_bin` (bin 2) | 12.23 | +0.00 | 0.176 | 0.420 |
| **`oracle_bin` (supervised upper bound)** | **12.28** | **+0.06** | 0.173 | 0.416 |
| `pc2` | 12.45 | +0.22 | 0.143 | 0.378 |
| `spec_mean` | 12.46 | +0.23 | 0.140 | 0.378 |
| `pc1` | 12.59 | +0.36 | 0.142 | 0.377 |
| `mean_bp` | 12.76 | +0.53 | 0.116 | 0.342 |

The oracle bin's FFT-peak estimate is 1.64 BPM vs 15.74 for bin 2 — a 10× gap —
yet feeding its *spectrum* to the Ridge changes nothing. So the 12.23 Ridge is
**not reading spectral peak position**; it reads a bin-agnostic property of the
whole band shape. The information the oracle demonstrates is present is exactly
the information this linear readout throws away.

---

## 11. Respiration harmonics: real phenomenon, unfixable by notching

`experiments/mmwave_selection/phase0_resp_harmonic_probe.py` (440 sessions,
canonical bin 2). Breathing f_r is estimated from the sub-HR band (0.1–0.6 Hz);
harmonics k·f_r for k = 2..8 are then either excluded from peak picking or
removed by least-squares harmonic regression.

| variant | MAE |
|---|---:|
| plain FFT peak | 15.72 |
| peak excluding ±0.08 Hz around k·f_r | 17.47 |
| harmonic regression removed, then peak | 17.47 |

Ridge on the 71-d spectrum: 12.23 (plain) vs 12.26 (harmonics removed).

**The mechanism is real**: the HR-band peak sits on a respiration harmonic in
**61.4%** of sessions (73.6% for Lying/Post-exercise — about 2× the chance level
for that band width). But the fix splits by true HR:

| true-HR decile | mean HR | plain | harmonics excluded |
|---|---:|---:|---:|
| 1 | 52.1 | 8.51 | 21.22 |
| 2 | 58.9 | 4.58 | 9.52 |
| 5 | 70.0 | 7.57 | 11.88 |
| 7 | 79.2 | 17.19 | **15.80** |
| 8 | 84.3 | 19.62 | **15.33** |
| 10 | 118.7 | 51.31 | **48.11** |

Low HR is destroyed, high HR improves. Reason: at rest f_r ≈ 0.19 Hz, so
5f_r ≈ 57 BPM and 6f_r ≈ 69 BPM — **the heartbeat itself lies on a respiration
harmonic** (respiratory–cardiac coupling). Notching the harmonics removes the true
peak more often than it removes a false one. Frequency-domain separation of
heartbeat from breathing harmonics is therefore not available on this dataset.

---

## 12. Consolidated checkpoint (2026-09-10)

Four independent axes, all measured on the full 440 sessions with the same
subject-level `GroupKFold(5)` Ridge, all landing inside ±0.5 BPM of each other:

| axis | best result | verdict |
|---|---|---|
| backbone (6 models) | 11.66 (TCN) vs 12.23 Ridge | null (CI overlaps) |
| RDA front-end pool (18 candidates) | 12.35 (`R/hr_band` under `spec_mean`) | null |
| range-bin selection (oracle upper bound) | 12.28 vs 12.23 | **no headroom** |
| spatial combination (PC1 / mean) | 12.59 / 12.76 | null |
| respiration-harmonic removal | 12.26 | null |

Reference points: constant prediction 14.28, classical FFT peak 15.7, Ridge
`single_bin` 12.23 (R² 0.176, r 0.42).

The 12.2–12.5 plateau is reached by every route tried. (§13 moves the same probe to
the other three datasets — the plateau turns out to be a *best case*, not a ceiling.)

The one number that
escapes it (oracle FFT peak 1.64 BPM) is **unreachable**: no unsupervised rule
finds that bin (best 15.3), split-half shows the good bin is not even stable
within one session (18.32), and when handed to the Ridge it changes nothing
(12.28). Conclusion: with a single ~50 s window at 10 Hz, a whole-session
spectral representation, and subject-level CV over 110 participants, the
linear-information ceiling is ~12.2 BPM. Further gains need a different problem
setup (longer/multi-window sessions, within-subject calibration, or a genuinely
different target), not another front-end filter.

---

## 13. Cross-dataset probe: the plateau is a best case, not a ceiling

`experiments/mmwave_selection/cross_dataset_ridge_probe.py` moves the **identical
probe protocol** (subject/session-level `GroupKFold(5)`, inner alpha CV,
standardize + y-centering on train folds, clip [30, 200] BPM) to PhysDrive /
FTU / BGT60, using `training_exports` (12994 windows; fs 20 Hz, HR band
0.75–2.5 Hz, `group_key` per dataset).

| dataset | groups | n | constant | Ridge (grouped) | gain | **r** | Ridge (random folds) | within-group σ of HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmwave-897 | 110 | 440 | 14.28 | 12.23 | **+2.05** | **+0.420** | — | **9.88** |
| PhysDrive | 48 | 5026 | 10.29 | 10.35 | −0.06 | **−0.270** | 9.81 (+0.47) | 6.76 |
| FTU | 10 | 3520 | 8.71 | 8.77 | −0.05 | **−0.644** | 7.31 (+1.41) | 3.55 |
| BGT60 | 8 | 4448 | 11.34 | 11.39 | −0.06 | **−0.772** | 8.96 (+2.38) | 3.80 |

(three datasets shown with the `phase71` representation = mmwave-897's own recipe
applied to `sum(x_time)`; the 903-d `x_freq` EDACM/VMD representation gives the
same numbers to ±0.02 r. mmwave-897's row is its 71-d `single_bin` spectrum.)

Three controls, all run before believing the negative correlations:

1. **Permuted labels** → r = −0.002 / −0.021 / −0.034, gain ≈ 0. The probe is not
   buggy; the negative r is real anti-signal in the data, not an artifact.
2. **Subject count** (`phase0_groupcount_ablation.py`, mmwave-897 subsampled,
   5 random subsets per N): N=8 → r +0.346, N=20 → +0.368, N=48 → +0.400,
   N=110 → +0.420. At matched group counts mmwave-897 stays *positive* while
   FTU/BGT60 are −0.6/−0.8, so subject count is not the explanation.
3. **Representation**: `phase71` (mmwave-897's recipe: band-pass → Hann → FFT →
   interpolate onto the 71-point 0.5–4.0 Hz grid → log1p → L2-normalize) applied
   to the other three reproduces their negative r. Representation is not the
   explanation either.

**The driver is label structure.** Under subject/session-level CV the learnable
signal is bounded by how far the label moves *inside* a group:

| dataset | total σ | between-group σ | **within-group σ** | within/total | grouped r |
|---|---:|---:|---:|---:|---:|
| mmwave-897 | 19.38 | 15.36 | **9.88** | 0.51 | +0.420 |
| PhysDrive | 12.05 | 9.98 | 6.76 | 0.56 | −0.270 |
| BGT60 | 11.74 | 11.11 | 3.80 | 0.32 | −0.772 |
| FTU | 9.04 | 8.32 | 3.55 | 0.39 | −0.644 |

Within-group HR spread ranks the datasets exactly as the probe's transferability
does. mmwave-897 is the only one with a large HR-driven within-subject range
(Rest vs Post-exercise: P001 = 64.0 / 86.9 / 67.8 / 87.3 BPM). On FTU the
within-subject variation that *is* present is deliberately geometry (Distance /
Orientation / Angle scenarios), i.e. nuisance, so a cross-subject linear model
fits nuisance and anti-generalizes.

**Consequences**

* The ~12.2 plateau is not a universal ceiling: it is the *best* of the four.
  On the other three the same protocol sits at or below the constant baseline.
* Reported low MAEs on these datasets (e.g. FTU ≈ 3.7, BGT60 ≈ 3.1) must be read
  together with Pearson: `experiments/results.json` already documents prediction
  collapse (pred_mean ≈ 74.7 for every target, in-domain Pearson −0.054). A model
  that outputs the cohort mean scores a low MAE wherever the label distribution is
  narrow — that is not HR estimation.
* Evaluating "cross-subject radar HR regression" is only meaningful on datasets
  that contain within-subject HR variation. Otherwise subject-level CV removes
  essentially all the label variance and leaves nuisance.

---

## 14. Final summary by family

### What was actually run

The `src/radar` pool registers **19 methods** (C5 + L4 + R4 + B3 + P3), three of
which *are* the current baseline (`localization/energy`, `range_selection/global_energy`,
`beamforming/fft`). Round 1 therefore screened **16 non-baseline candidates plus 2
baseline rows = 18 rows**; round 2 re-ran 6 of them (R×3 under `spec_mean`, P×3 with
the band-pass fix). Everything below also folds in the ablations that live outside
the pool.

| # | family | in pool | outside the pool | best result | Δ vs reference | verdict |
|---|---|---:|---|---:|---:|---|
| 1 | **Clutter suppression** | 5 | 4 methods × 2 selectors × 3 datasets (`rda_clutter_x_range`) | `temporal_highpass` 12.39 | −0.02 | null |
| 2 | **Localization** | 4 | — | `hr_band` 12.34 | −0.07 | null |
| 3 | **Range-bin selection** | 4 | 10 selectors (`select_range_bin`) + oracle + split-half | `hr_band` 12.35; **oracle 12.28** | −0.15; **+0.06** | **no headroom** |
| 4 | **Beamforming / spatial combination** | 3 | PC1/PC2/mean-of-bins, 6 spectrum reps | `bartlett` 12.38 | −0.03 | null |
| 5 | **Phase extraction** | 3 | 5 methods × 3 datasets (`rda_phase_extraction`) | `p0_single` 12.67 | +0.44 | worse |
| 6 | **Respiration harmonics** | — | 3 variants (plain / exclude / notch) | harmonic-removed 12.26 | +0.03 | null, mechanism unfixable |
| 7 | **EDACM isolation** (cross-dataset) | — | 4 variants × 3 datasets | probe R² −0.59…−3.79 | — | independent confirmation of §13 |

Reference for Δ: Ridge on `single_bin` = **12.23** (constant 14.28, FFT peak 15.7).
Promotion bar was ±0.5 BPM; **nothing clears it.**

### Per-family findings

**1. Clutter suppression.** mmwave-897: 12.39–12.56, all within ±0.15 of the
12.41 baseline row; `temporal_highpass` (12.39) is best and still loses to 12.23.
Cross-dataset (FFT-peak MAE, `rda_clutter_x_range`, n = 440/32/1681):

| dataset | current | mti | highpass | pca |
|---|---:|---:|---:|---:|
| FTU (440) | **18.27** | 18.61 | 19.38 | 19.14 |
| PhysDrive (1681) | 18.83 | 18.77 | 18.68 | **18.36** |
| BGT60 (32) | **19.56** | 31.40 | 30.21 | 30.29 |

FTU and PhysDrive are flat (< 0.5 BPM spread). BGT60 is the only dataset with a
strong result, and it is a *negative* one: MTI / high-pass / PCA cost +10 BPM there.
⚠️ `experiments/rda_clutter/` (clutter only) ran on **n = 2 sessions** — do not cite it.

**2. Localization.** `hr_band` 12.34 (−0.07) < `temporal_consistency` 12.57 <
`spatial_consistency` 12.58. The direction is right (HR-band beats total energy) but
the size is 0.07 BPM. Decisive fact: **the pipeline always reads bin 2 (0.63 m)** —
argmax ∈ {2, 3} in 100% of sessions under both energy criteria (§9).

**3. Range-bin selection.** Under `--rep single_bin` all three candidates are
bit-identical at 12.41 (their keep-windows always contain bin 2). Re-screened under
`--rep spec_mean`: `hr_band` **12.35** (−0.15) < `weighted_center` 12.50 < `peak`
12.55. The supervised upper bound settles it: **oracle bin → Ridge 12.28 vs 12.23**.
Perfect bin selection buys nothing, even though the same oracle bin gives the
FFT-peak estimator 1.64 vs 15.74. The good bin is also not stable in time
(split-half 18.32).

**4. Beamforming / spatial combination.** `bartlett` 12.38 / `mvdr` 12.44 vs 12.41 —
no effect (and mmwave-897 has no Doppler axis, so this family is under-tested here).
Spatial *combination* is consistently worse: `pc2` 12.45, `pc1` 12.59, `mean_bp`
12.76, `spec_mean` 12.46 → **one bin ≈ the full spatial spectrum for a linear probe**.

**5. Phase extraction.** After removing the band-pass confound: `p0_single` 12.67,
`p2_fusion` 12.75, `p1_diff` 12.96 — all worse than the canonical 12.23. Round 1's
apparent +1.88/+2.00 was mostly the missing band-pass. Cross-dataset (FFT-peak MAE):

| dataset | angle | detrend | multi_bin | conj_ref | phase_diff |
|---|---:|---:|---:|---:|---:|
| FTU (440) | **18.27** | 18.27 | 19.41 | 21.54 | 33.45 |
| PhysDrive (1681) | 18.83 | 18.83 | **18.63** | 19.03 | 31.12 |
| BGT60 (32) | **20.48** | 20.48 | 23.11 | 23.82 | 35.48 |

`angle` / `detrend` win everywhere; `phase_diff` is catastrophic (+13…+15 BPM).

**6. Respiration harmonics.** The mechanism is real — the HR-band peak sits on a
respiration harmonic in **61.4%** of sessions (73.6% post-exercise, ~2× chance) — but
removing them makes things *worse* (exclusion 17.47, harmonic regression 17.47, vs
plain 15.72) because at rest f_r ≈ 0.19 Hz puts 5f_r ≈ 57 and 6f_r ≈ 69 BPM, i.e.
**the heartbeat is on a harmonic**. Low-HR deciles are destroyed (8.51 → 21.22),
high-HR deciles improve (19.62 → 15.33). The Ridge is unaffected (12.26).

**7. EDACM isolation (cross-dataset, independent).** Probe R²: FTU −0.645 … −0.783,
BGT60 −0.593 … −0.696, PhysDrive −2.355 … −3.789; `hr_corr_peak` ≈ −0.03 … +0.16.
This reproduces §13 (grouped r = −0.64 / −0.77 / −0.27) with a different code path
and a different representation, which is why the negative-transfer finding is not
attributed to a bug in the probe.

### One-paragraph version

19 pool methods plus 6 families of outside-pool controls were measured against a
fixed linear probe on mmwave-897 (440 sessions, subject-level `GroupKFold(5)` Ridge,
12.23 reference): clutter ±0.15, localization −0.07, range selection −0.15 with a
12.28 oracle ceiling, beamforming −0.03, spatial combination +0.2…+0.5, phase
extraction +0.4…+0.7, respiration-harmonic removal +0.03. Nothing moves the probe by
more than ±0.5 BPM, and the one number that looks large (oracle FFT peak 1.64 BPM) is
unreachable: no unsupervised rule finds that bin (best 15.3), it is not stable within
a session (18.32), and it is worth nothing to the linear readout (12.28). Moving the
same probe to PhysDrive / FTU / BGT60 shows 12.2 is the *best* of the four, not a
ceiling — on the other three the probe sits at or below the constant baseline with
negative Pearson, because their HR variance is mostly between subjects.
