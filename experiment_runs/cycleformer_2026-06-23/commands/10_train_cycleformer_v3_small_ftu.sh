#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

python3 src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/deep_models/cycleformer_v3_small_ftu_source \
  --epochs 80 \
  --batch-size 32 \
  --num-workers 2 \
  --d-model 32 \
  --d-ff 64 \
  --num-layers 1 \
  --nhead 4 \
  --hidden-channels 48 \
  --num-blocks 4 \
  --kernel-size 7 \
  --dropout 0.15
