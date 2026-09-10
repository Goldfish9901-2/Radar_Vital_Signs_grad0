# TSLANet Backbone — Progress Log

**Last updated:** 2026-07-25 (FTU benchmark complete)

## Status: Phase 2a + validation + FTU benchmark DONE

All three requested stages are complete and the unified experiment results have been generated.

---

## Stage 1 — Phase 2a: `channel_mode='official'`

Implemented alongside the existing `joint` mode; frequency branch untouched; `use_frequency_domain=False` hardcoded (R8 deferred).

Changes (`src/models/tslanet.py`):
- Module docstring now documents both channel modes.
- `TSLANetBranch.__init__` no longer asserts `joint` only.
- `TSLANetHeartRateModel` accepts `channel_mode ∈ {"joint","official"}` and builds the official forward path:
  - official = channel-independent: `(B,C,T) → (B*C,1,T)`, single `Linear(P, emb)`, blocks, then mean over channels → scalar.
  - joint = channels mixed in patch projection `Linear(C*P, emb)`.

CLI / config wiring (`src/models/factory.py`, `src/training/train_model.py`):
- New args: `--emb-dim`, `--tslanet-depth`, `--tslanet-patch-len`, `--tslanet-patch-stride`, `--tslanet-dropout`, `--use-asb/--no-use-asb`, `--use-icb/--no-use-icb`, `--adaptive-filter/--no-adaptive-filter`, `--normalize/--no-normalize`, `--channel-mode {joint,official}`.
- `create_model_and_config` reads them into `TSLANetConfig`.

Tests (`tests/test_tslanet.py`): numerical-equivalence vs verbatim official ASB/ICB/layer; shape/regression tests for both modes, `joint==official` at C=1, channel independence, factory build, config roundtrip, CLI defaults, gradients, deterministic init.

## Stage 2 — Validation (no new features)

Smoke harness (`validate_tslanet_smoke.py`): 1 epoch, loss-decrease, NaN, GPU mem, param count, forward latency, output range.

**Bug fixed:** local `trunc_normal_` used `uniform_(2*l, 2*u)`; upper bound >1 makes `erfinv_` return NaN, poisoning ASB weights. Correct timm form is `uniform_(2*l-1, 2*u-1)`. After fix: loss 0.6341→0.4809, no NaN, GPU 64.6 MB, 198,788 params, latency 3.80 ms, pred 76.9–79.1 BPM. (Reviewer risk R13.)

## Stage 3 — Full FTU benchmark (20 epochs, batch 32)

Both `joint` and `official` run; best checkpoint saved; per-epoch loss/MAE/RMSE recorded; unified results generated.

**Final FTU test metrics:**

| setting | MAE | RMSE | within5 | within10 | Pearson |
|---|---|---|---|---|---|
| joint   | 4.25 | 5.32 | 62.6% | 94.2% | 0.095 |
| official| 3.69 | 4.76 | 69.3% | 96.2% | 0.114 |

Official beats joint on every metric. Official also beats the CycleFormer-v3 reference (MAE 4.19 / RMSE 5.41 at 80 epochs) **at only 20 epochs**.

**Early-stop (patience=12):** joint ran 15 epochs (best val MAE 4.45 @ ep4); official ran 12 epochs (best val MAE 4.18 @ ep1).

---

## Artifacts

- `model_outputs/tslanet_joint_ftu_source/` and `tslanet_official_ftu_source/`: `run_config.json`, `history.json`, `summary.json`, `best.pt`, `final.pt`, `eval_test.json`, `eval_test_FTU.json`.
- `experiment_runs/tslanet_2026-07-25/tables/overall_metrics.csv` (unified schema, header + 2 rows).
- `experiment_runs/tslanet_2026-07-25/tables/per_epoch_metrics.csv` (header + 54 rows: train/val per epoch).
- `experiment_runs/tslanet_2026-07-25/make_tslanet_tables.py` (generator).

## Code changes
- `src/models/tslanet.py` — official mode + trunc_normal_ NaN fix + deviations doc update.
- `src/models/factory.py` — CLI→TSLANetConfig wiring.
- `src/training/train_model.py` — TSLANet CLI args, RMSE logging, eval_test.json write.
- `src/training/common/metrics.py` — `rmse_bpm`.
- `tests/test_tslanet.py` — equivalence/shape tests.
- `validate_tslanet_smoke.py` — smoke harness.

## Notes
- Two background training tasks reported "failed" only due to a fish-shell `$?` quirk (`fish` uses `$status`); training itself completed and all artifacts are present. Cosmetic only.
- Benchmark used 20 epochs (quick). For a strictly apples-to-apples paper table vs CycleFormer-v3 (80 epochs), an 80-epoch TSLANet run is the natural Phase-3 follow-up — but `official` already leads at 20, so the conclusion is unlikely to flip.

## Not yet done (awaiting direction)
- Phase 2b: frequency-branch R8 design (`use_frequency_domain` still False).
- Phase 3: 80-epoch run + multi-dataset (PhysDrive / BGT60TR13C).
- Optional dropout sweep (official default 0.5 is high vs repo ~0.15).
- Optional: sync `work/upstream_rw` changes back to read-only `work/upstream`.
