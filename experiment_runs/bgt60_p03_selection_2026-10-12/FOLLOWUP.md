# Frozen selection follow-up (local uv)

No rule changes, threshold tuning or label-based selection. p03 exploratory cases only; no new subjects.
Tests: independent per-channel constant phase rotations; selection causality against future-signal modification.
Paired errors use identical zero-offset valid windows. No overlapping-window significance tests.

| Record | Strategy | Full MAE | Full r | Improved / worsened / tied windows | Selection counts |
|---|---|---:|---:|---|---|
| 0.3m | fixed | 4.862 | 0.395 | 0/0/91 | {'RX2/bin14': 92} |
| 0.3m | energy | 4.516 | 0.382 | 6/5/80 | {'RX1/bin15': 4, 'RX2/bin14': 50, 'RX1/bin14': 38} |
| 0.3m | quality | 5.493 | 0.446 | 4/8/79 | {'RX1/bin15': 1, 'RX0/bin12': 5, 'RX2/bin14': 45, 'RX1/bin14': 39, 'RX2/bin13': 2} |
| 0.6m | fixed | 20.683 | 0.126 | 0/0/91 | {'RX2/bin22': 92} |
| 0.6m | energy | 17.720 | -0.001 | 29/21/41 | {'RX2/bin23': 30, 'RX0/bin23': 3, 'RX1/bin24': 1, 'RX1/bin21': 15, 'RX1/bin23': 1, 'RX2/bin21': 2, 'RX1/bin22': 17, 'RX2/bin22': 23} |
| 0.6m | quality | 16.765 | -0.023 | 35/18/38 | {'RX1/bin23': 5, 'RX2/bin23': 9, 'RX0/bin23': 1, 'RX0/bin20': 12, 'RX2/bin24': 1, 'RX1/bin24': 5, 'RX2/bin22': 21, 'RX0/bin24': 3, 'RX2/bin21': 9, 'RX1/bin21': 7, 'RX0/bin21': 2, 'RX0/bin22': 3, 'RX1/bin22': 14} |

## Interpretation

A lower MAE without preserved within-record correlation is not reliable HR tracking. The quality heuristic may switch more than energy selection despite its switch penalty; this does not justify tuning the penalty on these reference labels.
Do not interpret power movement as chest displacement. These two records were selected after screening; no cross-subject validation or confidence claim is made.
The independent-window estimator is invariant to constant channel phase rotations, so a constant inter-channel phase offset alone does not explain changes in this frequency estimator. Time-varying channel artifacts remain possible.
Raw arrays are not included in the Git summary; input hashes are in the original experiment artifacts. No model training or Kaggle submission was performed for this follow-up.