# FTU P0 gates before supervised retraining

Run `uv run --locked --no-sync python -m src.data.audit_ftu_p0` for the local
three-record and synthetic-signal diagnostics. The archived E2a feature source
must match before execution. The diagnostic instrumented implementation must
produce exactly the same arrays as the original implementation.

`kaggle/prepare_ftu_p0_audit.py` packages a private CPU-only audit of all existing
E2a training windows and the five raw FTU datasets' small timing sidecars. No ADC
processing and no training occurs. After downloading its results, run
`uv run --locked --no-sync python summarize_ftu_p0_audit.py` locally.

## P0: mode integrity

Inspect pre-weighting and weighted modes, all 21 pairwise correlations, numerical
rank, per-mode power centroids and peaks, actual weights, clipping counts and
convergence history. Check the saved training arrays independently of the new
diagnostic code. A three-tone signal deliberately has only three independent
components: seven orthogonal outputs are not a requirement. Exact repeated arrays,
large clipping-driven growth and failure to converge are failure evidence.

The diagnostic `sequential_sum_diagnostic` only refreshes the residual's mode sum.
It is an isolated numerical intervention, not a released VMD fix. Before a fixed
implementation is accepted, compare against a vetted VMD reference with the same
analytic-spectrum convention, initialization, bandwidth penalty, boundary
extension and stopping rule. Check signal reconstruction before per-mode
normalization; normalization otherwise conceals amplitude failures. Cover single
tones, multitone mixtures, zero signals and noise, then real window-local inputs.
Do not silently change the front end of existing checkpoints.

## P0: timing integrity

Distinguish configured frame cadence, measured per-frame timestamps, HR CSV elapsed
times, wall-clock headers and cross-device synchronization. A radar configuration
of 20 Hz does not prove no missing frames or that HR and radar share a time origin.
Do not infer radar cadence from the HR reference length. Do not optimize a time lag
on held-out HR to maximize correlation.

Audit raw sidecar hashes and dataset/sample identity; compare raw HR CSV arrays to
exported labels. The current HR endpoint-to-frame `linspace` mapping must be
reported separately from a nominal `arange(n)/fs` sensitivity calculation. Raw
integer-second wall-clock offsets are evidence to investigate, not automatic lag
corrections. P0 timing passes only once cadence, start-offset policy, coverage,
interpolation and missing/duplicate-sample handling are documented and checked.
If independent synchronization cannot be established, dynamic-tracking claims
remain unverified and P1 stays pending under the user's gate.

## P1: minimal supervised comparison (conditional; not launched)

After BOTH P0 gates pass, version and regenerate the corrected representation and
label export. Preserve participant separation and the declared split; record all
excluded windows and coverage decisions. Freeze export/checkpoint/source hashes.

Train a small TCN and a small Transformer on Kaggle and compare with CycleFormer
trained on the SAME corrected export. An old-checkpoint score on the old export is
historical context, not a fair architecture comparison. Use common training/val/test
data, target normalization, early stopping and compute budgets; select on validation
only. Begin with one fixed seed, then use three fixed seeds for the comparison.

Report train-mean and train-median constant baselines, MAE/RMSE, prediction/label
standard deviations, variance ratio, subject and session correlations, pooled
within-subject correlation, trajectories and error distributions. Aggregate by
subject/session as well as windows; overlapping windows are not independent
replicates. A modest MAE gain without within-group tracking does not clear the
source-signal-use question. Do not launch SFDA on that basis.

Current results: [P0 report](../experiment_runs/ftu_p0_audit_2026-10-14/REPORT.md).

## Official-package follow-up (2026-10-16)

Use the temporary dependency command below, without adding sktime to project dependencies:

```powershell
uv run --locked --no-sync --with sktime==1.2.0 python -m src.data.experiment_ftu_sktime
uv run --locked --no-sync python experiment_ftu_timing_sensitivity.py
```

This uses the unmodified `sktime.libs.vmdpy.VMD`, uniform initialization and a
separately reported HR-band scoring step. It is a standard-VMD baseline, not a
reproduction of the old physiological initialization. The archived vmdpy README
points to sktime as the maintained package. No custom solver is substituted.

The Kaggle CPU experiment selects the first window of each FTU record before
looking at labels, reconstructs window-local EDACM, and compares both solvers on
that same signal. Historical feature mismatches and target-bin mismatches remain
explicit; they are not hidden by increasing a tolerance or excluding records.
The timing experiment uses a fixed offset grid and header-based sensitivity
assumptions. It cannot establish independent clock synchronization.

Results: [sktime and timing report](../experiment_runs/ftu_sktime_baseline_2026-10-16/REPORT.md).
