# BGT60 acquisition configuration search

Search date: 2026-10-10. The exact file name
`BGT60TR13C_settings_20250423-163757.json`, raw file names, HeartTimeMixer,
and publication/repository terms were searched. No downloadable acquisition
JSON was located. The existing Kaggle mirror's file listing contains the ADC
and reference files, but no settings JSON.

The strongest candidate original publication is [A time–frequency feature fusion
method for contactless heart rate estimation using mmWave FMCW radar](https://www.sciencedirect.com/science/article/abs/pii/S1746809426005884),
DOI 10.1016/j.bspc.2026.110034. Its acquisition dimensions, distances and long
recording duration match this mirror, but file-level provenance is not established.

Its Experiment setup section reports:

| Parameter | Reported value |
| --- | --- |
| Radar | BGT60TR13C, 1 Tx / 3 Rx |
| Start / end frequency | 58 / 63.5 GHz |
| Sweep bandwidth | 5.5 GHz |
| Frame frequency | 30 Hz |
| Chirps per frame | 16 |
| Samples per chirp | 512 |
| ADC sampling frequency | 2 MHz |
| Chirp repetition interval | 400 µs |
| Nominal range resolution | about 2.7 cm |

The 400 µs interval is not established as the active sweep duration. The ADC
capture spans 256 µs (512 / 2 MHz), which is also not independently established
as the full ramp duration. Neither should be silently substituted for ramp time.
Without slope or verified sampled sweep bandwidth, a native FFT bin cannot yet
be assigned a verified metric distance. Native and historical nonuniformly
selected ADC grids must not share the same integer ROI without conversion.

Diagnostics therefore default to bin coordinates and mark physical distance
ROI as unavailable. A calibration JSON may provide positive `sample_rate_hz`,
`chirp_slope_hz_per_s`, and a `source` identifying the actual acquisition settings.
The published nominal resolution can guide subsequent checks, but does not
validate the current mirror's distance-axis calibration.
