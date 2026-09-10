# CycleFormer Experiment Run

Date: 2026-06-23

Container: `radar_dev`

Project path in container: `/Radar_Vital_Signs`

## Scope

This run follows the revised CycleFormer plan:

- Main model: CycleFormer
- Baselines: FFT, STFT, TCN, Transformer, PatchTST, TimesNet
- Transfer settings: Source Only and Pseudo-label Adaptation
- Experiments: within-dataset, cross-dataset, source-free pseudo-label adaptation, and CycleFormer ablations

## Directory Layout

- `commands/`: exact commands used for each experiment.
- `logs/`: stdout/stderr logs captured from container runs.
- `results/`: copied or generated JSON result files.
- `tables/`: CSV files intended for paper tables and plotting.
- `figures/`: reserved for generated plots.
- `configs/`: copied run configs and experiment metadata.
- `notes/`: manual notes and status records.

## Current Execution Order

1. Smoke test model training paths with small debug runs.
2. Evaluate FFT and STFT baselines.
3. Train CycleFormer on FTU.
4. Evaluate Source Only transfer to PhysDrive and BGT60TR13C.
5. Run Pseudo-label Adaptation for CycleFormer.
6. Add TCN, Transformer, PatchTST, and TimesNet baselines.
