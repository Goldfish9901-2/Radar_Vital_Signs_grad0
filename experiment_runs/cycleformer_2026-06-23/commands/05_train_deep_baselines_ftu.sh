#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

common_args=(
  --datasets FTU
  --export-dir training_exports
  --epochs 80
  --batch-size 32
  --num-workers 2
  --dropout 0.15
)

python3 src/training/train_model.py \
  --model tcn \
  "${common_args[@]}" \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/deep_models/tcn_ftu_source \
  --hidden-channels 48 \
  --num-blocks 4 \
  --kernel-size 7

python3 src/training/train_model.py \
  --model transformer \
  "${common_args[@]}" \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/deep_models/transformer_ftu_source \
  --d-model 64 \
  --d-ff 128 \
  --num-layers 2 \
  --nhead 4

python3 src/training/train_model.py \
  --model patchtst \
  "${common_args[@]}" \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/deep_models/patchtst_ftu_source \
  --d-model 64 \
  --d-ff 128 \
  --num-layers 2 \
  --nhead 4 \
  --patch-len 16 \
  --patch-stride 8

python3 src/training/train_model.py \
  --model timesnet \
  "${common_args[@]}" \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/deep_models/timesnet_ftu_source \
  --d-model 64 \
  --d-ff 128 \
  --num-blocks 3
