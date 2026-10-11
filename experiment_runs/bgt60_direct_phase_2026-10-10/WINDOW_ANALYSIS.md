# Direct phase window-level analysis

## Findings

The native local range peaks increase consistently with acquisition distance:
0.3 m bins 11–14; 0.6 m 20–25; 0.9 m 28–37; 1.2 m 42–47.
This supports a distance-dependent reflector, but does not identify chest motion
or establish a verified metric range axis.

Zero-Doppler unwrap predicts a mean 61.294 bpm against reference 83.648 bpm on
matched windows. Its lowest permitted FFT bin (46.875 bpm) occurs in 29.6% of
windows. Its mean within-record Pearson correlation is only 0.042; the pooled
correlation is 0.055. Peak-at-band-boundary bias is consistent with residual
low-frequency contamination, but its source is not established by this file.

One exploratory exception is p03_0.3m: MAE 3.015 bpm, Pearson r 0.602 across
55 overlapping matched windows. The same subject at 0.6 m has MAE 24.99 bpm
and r −0.026. This motivates targeted signal inspection, not a claim of stable
cross-subject recovery. The eight-subject macro MAE is 24.319 bpm, compared with
10.054 bpm for the matched-window LOSO training-mean constant.

Zero-Doppler unwrap is 0.264–0.393 bpm better at zero offset than the four fixed
shift controls. The paired subject bootstrap intervals are positive before
multiple-comparison adjustment; these are weak exploratory indications, not
absence of all synchronous information. Slow HR changes and systematic
underestimation limit interpretation of an MAE-based shift control.

All reference sequences start at 1 s, end at 600 s, and contain 600 rows. The
32 initial windows have incomplete label coverage and are excluded rather than
extrapolated. Absolute start synchronization and the assumed 30 Hz radar rate
remain unverified. No raw ECG is present in this analysis.

176,640 rows represent 2,944 physical windows × 12 representations × 5 reference offsets. Overlapping windows are not independent subjects.

All offset comparisons below use the exact intersection of valid sample/window keys. CIs resample eight subject-level paired differences; exploratory, no multiplicity correction.

| Representation | Matched windows | Subject macro MAE | Pooled r | Lowest HR-bin fraction |
|---|---:|---:|---:|---:|
| first_chirp/complex_real/0 | 1760 | 24.100 | 0.123 | 0.220 |
| first_chirp/complex_imag/0 | 1760 | 25.180 | 0.122 | 0.241 |
| first_chirp/unwrap/0 | 1760 | 24.253 | 0.057 | 0.294 |
| first_chirp/edacm/0 | 1760 | 24.243 | 0.059 | 0.299 |
| zero_doppler/complex_real/0 | 1760 | 24.157 | 0.118 | 0.222 |
| zero_doppler/complex_imag/0 | 1760 | 25.177 | 0.127 | 0.244 |
| zero_doppler/unwrap/0 | 1760 | 24.319 | 0.055 | 0.296 |
| zero_doppler/edacm/0 | 1760 | 24.470 | 0.044 | 0.305 |
| adjacent_doppler/complex_real/0 | 1760 | 23.993 | 0.149 | 0.069 |
| adjacent_doppler/complex_imag/0 | 1760 | 23.260 | 0.160 | 0.049 |
| adjacent_doppler/unwrap/0 | 1760 | 27.377 | -0.017 | 0.327 |
| adjacent_doppler/edacm/0 | 1760 | 27.555 | -0.021 | 0.326 |

Matched-window LOSO training-subject mean constant macro MAE: 10.054 bpm.

Pooled correlation can reflect between-subject heart-rate differences; inspect per-record and per-subject correlations in window_analysis.json.
Phase variance of complex real/imaginary amplitude is in ADC units, and cannot be compared directly with phase variance in radians.
Acquisition slope, frame rate and absolute radar/reference synchronization remain unverified. Reference is a 1 Hz HR series, not raw ECG.

| Representation | Offset seconds | Shift − zero MAE | Subject bootstrap 95% CI | Subjects zero better |
|---|---:|---:|---|---:|
| first_chirp/complex_real | -60 | 0.165 | [0.013974982984135007, 0.3400339466441742] | 7 |
| first_chirp/complex_real | 60 | 0.097 | [-0.11668210671164998, 0.28935857599431913] | 5 |
| first_chirp/complex_real | -120 | 0.153 | [-0.04971590909090895, 0.35209102746211957] | 6 |
| first_chirp/complex_real | 120 | 0.061 | [-0.3391565200343301, 0.43841648910984854] | 5 |
| first_chirp/complex_imag | -60 | 0.069 | [-0.1408599668560624, 0.30090801816998036] | 4 |
| first_chirp/complex_imag | 60 | 0.066 | [-0.15720491906368378, 0.27324014559659204] | 5 |
| first_chirp/complex_imag | -120 | 0.040 | [-0.11533661813447171, 0.20584324692234812] | 4 |
| first_chirp/complex_imag | 120 | 0.057 | [-0.3049221709280303, 0.4092061730587102] | 5 |
| first_chirp/unwrap | -60 | 0.390 | [0.2192931758996175, 0.5755862482244312] | 7 |
| first_chirp/unwrap | 60 | 0.273 | [0.054245235558711746, 0.5273254024621217] | 6 |
| first_chirp/unwrap | -120 | 0.295 | [0.0806902521306816, 0.532903349905302] | 7 |
| first_chirp/unwrap | 120 | 0.359 | [0.032827444365531866, 0.6714464222301133] | 6 |
| first_chirp/edacm | -60 | 0.389 | [0.21813343394886253, 0.5866064823035032] | 8 |
| first_chirp/edacm | 60 | 0.276 | [0.06583688446969882, 0.5281624348958325] | 6 |
| first_chirp/edacm | -120 | 0.276 | [0.0581605113636372, 0.5043180486505647] | 6 |
| first_chirp/edacm | 120 | 0.322 | [-0.06064462372750904, 0.6841665187026508] | 6 |
| zero_doppler/complex_real | -60 | 0.170 | [0.027581528172348047, 0.33535940459280134] | 6 |
| zero_doppler/complex_real | 60 | 0.094 | [-0.12427542021780535, 0.29293249881628647] | 5 |
| zero_doppler/complex_real | -120 | 0.143 | [-0.06220747514204428, 0.3415213660037866] | 6 |
| zero_doppler/complex_real | 120 | 0.064 | [-0.33814067678740184, 0.44036447236032544] | 5 |
| zero_doppler/complex_imag | -60 | 0.095 | [-0.11463173606179064, 0.3219344778349897] | 4 |
| zero_doppler/complex_imag | 60 | 0.089 | [-0.12664705033735654, 0.2913162619850856] | 5 |
| zero_doppler/complex_imag | -120 | 0.061 | [-0.08872055516098643, 0.2173442049893465] | 5 |
| zero_doppler/complex_imag | 120 | 0.084 | [-0.2816309666489127, 0.4310502485795451] | 5 |
| zero_doppler/unwrap | -60 | 0.393 | [0.2356399443655277, 0.5732385401870239] | 8 |
| zero_doppler/unwrap | 60 | 0.264 | [0.028284338748817094, 0.529858942205256] | 6 |
| zero_doppler/unwrap | -120 | 0.272 | [0.05704796993371386, 0.5166398851799252] | 6 |
| zero_doppler/unwrap | 120 | 0.350 | [0.01289232658617454, 0.6667677038944153] | 6 |
| zero_doppler/edacm | -60 | 0.380 | [0.22412588408499082, 0.5518631628787865] | 8 |
| zero_doppler/edacm | 60 | 0.271 | [0.0630636282256123, 0.5232515092329539] | 6 |
| zero_doppler/edacm | -120 | 0.251 | [0.05519353693181828, 0.46972508285984826] | 7 |
| zero_doppler/edacm | 120 | 0.323 | [-0.039933556758998415, 0.6653998579545406] | 6 |
| adjacent_doppler/complex_real | -60 | 0.273 | [0.005761518998578433, 0.6266740648674247] | 5 |
| adjacent_doppler/complex_real | 60 | 0.371 | [0.06917804509942897, 0.7110341205018962] | 6 |
| adjacent_doppler/complex_real | -120 | 0.319 | [0.11358321496212122, 0.5356106474905302] | 7 |
| adjacent_doppler/complex_real | 120 | 0.307 | [0.06539802320075827, 0.5673315207741512] | 6 |
| adjacent_doppler/complex_imag | -60 | 0.308 | [0.16839754971590892, 0.4518045691287873] | 7 |
| adjacent_doppler/complex_imag | 60 | 0.136 | [0.009508289683946371, 0.2677748653527416] | 7 |
| adjacent_doppler/complex_imag | -120 | 0.379 | [0.15109052068536588, 0.5804076260653378] | 7 |
| adjacent_doppler/complex_imag | 120 | 0.283 | [0.08402565696022313, 0.5041147608901512] | 6 |
| adjacent_doppler/unwrap | -60 | 0.143 | [0.01884442323626521, 0.28004241758404635] | 5 |
| adjacent_doppler/unwrap | 60 | 0.051 | [-0.0981110802852798, 0.19394752456202294] | 5 |
| adjacent_doppler/unwrap | -120 | -0.024 | [-0.17872630726207814, 0.1453125961766101] | 3 |
| adjacent_doppler/unwrap | 120 | 0.050 | [-0.2606196732954593, 0.3388350793087138] | 5 |
| adjacent_doppler/edacm | -60 | 0.021 | [-0.06380415482954405, 0.12956359123460842] | 3 |
| adjacent_doppler/edacm | 60 | 0.018 | [-0.14780110677083735, 0.1817221716678515] | 4 |
| adjacent_doppler/edacm | -120 | -0.022 | [-0.12657680812026528, 0.08637818122632736] | 3 |
| adjacent_doppler/edacm | 120 | 0.013 | [-0.320713652639681, 0.3302931167140173] | 5 |
