# p03 frozen ROI/RX selection comparison

All strategies share the same independent-window unwrap FFT. Quality parameters were frozen before execution; no label-based selection, no model training.
Exploratory p03 evaluation only. Absolute synchronization remains unverified. See protocol.json for the explicit heuristic.

| Record | Strategy | Full valid/total | Full MAE | Full r | Switches |
|---|---|---:|---:|---:|---:|
| 0.3m | fixed | 91/92 | 4.862425595238094 | 0.39505516274973496 | 0 |
| 0.3m | energy | 91/92 | 4.51584249084249 | 0.382027304570811 | 8 |
| 0.3m | quality | 91/92 | 5.49251945970696 | 0.4462992810668636 | 11 |
| 0.6m | fixed | 91/92 | 20.68309294871795 | 0.12625592398802435 | 0 |
| 0.6m | energy | 91/92 | 17.720163690476188 | -0.0006175697569470656 | 25 |
| 0.6m | quality | 91/92 | 16.764823717948715 | -0.023193891224184606 | 41 |

Per-block results and fixed offsets are in summary.json. Bin23 early and RX late controls are in prespecified_controls.json; no winner is chosen from their HR error.