# FTU D preprocessing: frozen real-label controls

This experiment is authorized after the synthetic factorial selected D. It does not reuse synthetic weights or synthetic success thresholds.

The frozen runner is `src/training/run_ftu_real_d.py`; `protocol.json` specifies the choices. The immutable prior capture split is `experiment_runs/ftu_record_learnability/record_split.csv`. Local preflight verified all 216 capture feature hashes and the 108/54/54 capture split, using only the original six training subjects.

Twelve Kaggle runs pair seeds 42/43/44 with correct labels and within-record shuffle seeds 101/202/303. All validation/evaluation labels remain correct. Training uses the unchanged MLP and 60-epoch maximum, without a per-bin scaler, masking outside the 29 bin centers in 0.5–2.75 Hz after original window normalization. No renormalization follows the mask.

Checkpoint selection uses validation record-centered RMSE only. Report raw and centered metrics, centered R², six subjects and their equal-weight average, paired shuffled-control differences, training-global constants and a fixed in-band maximum-bin baseline. The evidence of interest is held-out true-label advantage with positive centered R² across subjects; shuffled controls are diagnostic, not a formal permutation significance test.

Scope is same-subject, same-visit, cross-triggered-capture. Some p03 radar-start logs are missing, and this experiment does not certify synchronization or cross-session independence.

Kaggle: https://www.kaggle.com/code/goldfish9901/ftu-real-d-controls (version 1). This setup was committed while the task was queued, before observing training results. The final report, when available, is `REPORT.md`.
