# BGT60 direct phase and timing audit

No VMD or model training. No label-based ROI or offset selection.
Physical distance calibration and absolute synchronization remain unverified; see records.json and distance_calibration.json.
Negative controls use matched interior windows, no circular wrapping. Slow reference HR may make offsets weak controls.

| Coefficient / phase / reference offset seconds | Subject macro MAE |
| --- | ---: |
| first_chirp/complex_real/0 | 24.100432350852273 |
| first_chirp/complex_real/-60 | 24.265413263494317 |
| first_chirp/complex_real/60 | 24.197064393939392 |
| first_chirp/complex_real/-120 | 24.25377988873106 |
| first_chirp/complex_real/120 | 24.161393968986744 |
| first_chirp/complex_imag/0 | 25.179883700284094 |
| first_chirp/complex_imag/-60 | 25.249320993134468 |
| first_chirp/complex_imag/60 | 25.24567589962121 |
| first_chirp/complex_imag/-120 | 25.219808830492425 |
| first_chirp/complex_imag/120 | 25.236981090198864 |
| first_chirp/unwrap/0 | 24.252817530776518 |
| first_chirp/unwrap/-60 | 24.642833658854165 |
| first_chirp/unwrap/60 | 24.525599846117423 |
| first_chirp/unwrap/-120 | 24.548041844223484 |
| first_chirp/unwrap/120 | 24.61230868252841 |
| first_chirp/edacm/0 | 24.243362926136363 |
| first_chirp/edacm/-60 | 24.632850526751895 |
| first_chirp/edacm/60 | 24.518999171401518 |
| first_chirp/edacm/-120 | 24.518960700757575 |
| first_chirp/edacm/120 | 24.565377456202654 |
| zero_doppler/complex_real/0 | 24.157309126420454 |
| zero_doppler/complex_real/-60 | 24.326828983191287 |
| zero_doppler/complex_real/60 | 24.25165127840909 |
| zero_doppler/complex_real/-120 | 24.300549242424243 |
| zero_doppler/complex_real/120 | 24.22120546283144 |
| zero_doppler/complex_imag/0 | 25.17731534090909 |
| zero_doppler/complex_imag/-60 | 25.272411665482952 |
| zero_doppler/complex_imag/60 | 25.26627574573864 |
| zero_doppler/complex_imag/-120 | 25.23809747869318 |
| zero_doppler/complex_imag/120 | 25.261507013494317 |
| zero_doppler/unwrap/0 | 24.31939364346591 |
| zero_doppler/unwrap/-60 | 24.712790749289773 |
| zero_doppler/unwrap/60 | 24.583810073390154 |
| zero_doppler/unwrap/-120 | 24.591855172821973 |
| zero_doppler/unwrap/120 | 24.669363310842805 |
| zero_doppler/edacm/0 | 24.47043294270833 |
| zero_doppler/edacm/-60 | 24.8500078420928 |
| zero_doppler/edacm/60 | 24.741380208333332 |
| zero_doppler/edacm/-120 | 24.720980113636365 |
| zero_doppler/edacm/120 | 24.793133433948864 |
| adjacent_doppler/complex_real/0 | 23.993296934185608 |
| adjacent_doppler/complex_real/-60 | 24.266773348721593 |
| adjacent_doppler/complex_real/60 | 24.36412760416667 |
| adjacent_doppler/complex_real/-120 | 24.312104048295453 |
| adjacent_doppler/complex_real/120 | 24.3001679391572 |
| adjacent_doppler/complex_imag/0 | 23.26016364820076 |
| adjacent_doppler/complex_imag/-60 | 23.568457475142047 |
| adjacent_doppler/complex_imag/60 | 23.3965166311553 |
| adjacent_doppler/complex_imag/-120 | 23.63954930160985 |
| adjacent_doppler/complex_imag/120 | 23.543494762073863 |
| adjacent_doppler/unwrap/0 | 27.376809895833333 |
| adjacent_doppler/unwrap/-60 | 27.5197314453125 |
| adjacent_doppler/unwrap/60 | 27.4277178030303 |
| adjacent_doppler/unwrap/-120 | 27.352449396306817 |
| adjacent_doppler/unwrap/120 | 27.4269189453125 |
| adjacent_doppler/edacm/0 | 27.554954427083334 |
| adjacent_doppler/edacm/-60 | 27.576302527225376 |
| adjacent_doppler/edacm/60 | 27.573422999526514 |
| adjacent_doppler/edacm/-120 | 27.533274147727276 |
| adjacent_doppler/edacm/120 | 27.56780702533144 |