# mmwave-897: does the Ridge probe depend on the number of subjects?

- features: 71-d log1p HR-band spectrum (`single_bin`)
- protocol: subject-level GroupKFold(5) + inner alpha CV; 5 random participant subsets per N

| N subjects | sessions | grouped MAE | grouped r | gain vs constant | random r | random gain |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 14.70±1.20 | +0.346 | +1.94 | +0.389 | +1.57 |
| 10 | 40 | 16.29±3.71 | +0.148 | -1.75 | +0.255 | +0.10 |
| 20 | 80 | 12.02±1.55 | +0.368 | +1.36 | +0.399 | +1.49 |
| 48 | 192 | 12.43±1.25 | +0.400 | +2.07 | +0.424 | +2.09 |
| 110 | 440 | 12.23±0.00 | +0.420 | +2.20 | +0.439 | +2.21 |

Compare against the cross-dataset probe: FTU (10 groups) r=−0.59, BGT60 (8 groups) r=−0.77, PhysDrive (48 groups) r=−0.26.