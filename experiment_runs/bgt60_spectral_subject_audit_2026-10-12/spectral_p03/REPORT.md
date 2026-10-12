# p03 spectral peak attribution (frozen diagnostic)

Radar peaks are extracted before reference use. Reference HR/RR are used only to audit coincidences; no estimator, bin or threshold is changed.
Respiration search 0.1–0.6 Hz; HR 0.75–2.5 Hz; top three local peaks, including flagged band endpoints. Orders 2–6; tolerance one FFT bin (0.078125 Hz = 4.6875 bpm). Wrong means absolute HR error >5 bpm.

| Distance | Error group | Windows | Radar-harmonic match | Half-order control | Reference HR near radar harmonic | Prediction near reference-RR harmonic | HR in top3 (diagnostic) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.3m | wrong | 18 | 0.722 | 0.611 | 0.444 | 0.611 | 0.167 |
| 0.3m | within_5 | 73 | 0.836 | 0.329 | 0.822 | 0.562 | 0.986 |
| 0.6m | wrong | 64 | 0.719 | 0.688 | 0.250 | 0.312 | 0.531 |
| 0.6m | within_5 | 27 | 0.556 | 0.815 | 0.222 | 0.519 | 0.926 |

Harmonic masks and shifted masks have different boundary coverage; their grid coverage is preserved per window and in summary.json. Neither control is a statistical null test. Matching is not proof of respiratory origin.
Reference HR often coincides with an estimated respiratory multiple, including in accurate windows. Removing such peaks could remove real heartbeat information. Top3 recall is label-assisted diagnosis, not deployable oracle accuracy.
Respiration frequency resolution is coarse and the lowest/upper search bins can contain drift or noise. Reference RR is an existing 1Hz column under unverified shared-start synchronization; no raw ECG is available.
Both p03 cases are exploratory; overlapping windows are not independent experiments. Do not use these labels to choose harmonic subtraction parameters.