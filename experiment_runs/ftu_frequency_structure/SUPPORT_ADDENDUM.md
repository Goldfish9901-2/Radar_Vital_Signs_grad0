Input-support exception discovered during the first local run (before inspecting new scores):
p03, p04 and p05 reference timestamps stop at59s. Circular queries modulo60 include
the unobserved59--60s interval. Intersecting all offsets and all controls therefore
leaves zero scoring windows for these three records at both frozen STFT lengths.

The frozen control-based gate is unchanged. These records are unevaluable, not
evidence of failed correspondence. No extrapolation, missing-label filling,
alternative circular period, or best-offset selection is introduced.

All six records retain their full spectral/quality diagnostics. A separately named
descriptive_true_only.csv uses the intersection of the three correct-label offsets,
without shifted controls. It cannot satisfy or replace the expansion gate.
This is a disclosed post-protocol descriptive addition motivated by support, not scores.
