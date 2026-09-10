#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

python3 src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/deep_models/cycleformer_ftu_source \
  --epochs 80 \
  --batch-size 32 \
  --num-workers 2 \
  --d-model 64 \
  --d-ff 128 \
  --num-layers 2 \
  --nhead 4 \
  --dropout 0.15
