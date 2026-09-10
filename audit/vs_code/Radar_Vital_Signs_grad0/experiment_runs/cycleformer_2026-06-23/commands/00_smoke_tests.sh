#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

python3 src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/smoke_cycleformer \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --d-model 32 \
  --d-ff 64 \
  --num-layers 1 \
  --nhead 4 \
  --limit-batches 2

python3 src/training/train_model.py \
  --model patchtst \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/smoke_patchtst \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --d-model 32 \
  --d-ff 64 \
  --num-layers 1 \
  --nhead 4 \
  --limit-batches 2

python3 src/training/train_model.py \
  --model timesnet \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir experiment_runs/cycleformer_2026-06-23/results/smoke_timesnet \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --d-model 32 \
  --d-ff 64 \
  --num-blocks 1 \
  --limit-batches 2
