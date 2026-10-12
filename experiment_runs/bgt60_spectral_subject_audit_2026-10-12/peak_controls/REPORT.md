# Candidate-peak controls (local follow-up)

This is diagnostic use of reference labels, not a practical peak selector or oracle MAE. No estimation rule is changed.
Top3 proximity tolerance stays one FFT bin; exact common windows across offsets. Negative offsets are fixed, not optimized.

| Record | Group | Windows | Hit at 0 | -60 | +60 | -120 | +120 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.3m | all_matched | 55 | 0.891 | 0.745 | 0.709 | 0.727 | 0.782 |
| 0.3m | wrong_matched | 7 | 0.286 | 0.429 | 0.429 | 0.429 | 0.429 |
| 0.3m | nonoverlap_matched | 27 | 0.889 | 0.704 | 0.667 | 0.704 | 0.741 |
| 0.6m | all_matched | 55 | 0.600 | 0.545 | 0.600 | 0.564 | 0.545 |
| 0.6m | wrong_matched | 47 | 0.553 | 0.511 | 0.574 | 0.511 | 0.511 |
| 0.6m | nonoverlap_matched | 27 | 0.519 | 0.444 | 0.556 | 0.556 | 0.593 |

Reference-assisted RR agreement strata do not authorize discarding windows in the algorithm. Nonoverlap reduces shared samples but is not independent-subject validation.
Slow HR, coarse frequency bins and multiple candidate peaks can yield high recall without synchronous HR tracking. Harmonic and half-order masks have different edge coverage; coincidences do not prove source.
Do not choose new thresholds or a best time shift using these post hoc p03 labels.