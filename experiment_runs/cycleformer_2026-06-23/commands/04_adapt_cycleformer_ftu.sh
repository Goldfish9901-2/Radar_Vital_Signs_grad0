#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

source_model_dir="experiment_runs/cycleformer_2026-06-23/results/deep_models/cycleformer_ftu_source"

for dataset in PhysDrive BGT60TR13C; do
  dataset_slug="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
  python3 src/training/adapt_source_free.py \
    --source-model-dir "${source_model_dir}" \
    --export-dir training_exports \
    --target-datasets "${dataset}" \
    --adapt-split train \
    --eval-split test \
    --output-dir "experiment_runs/cycleformer_2026-06-23/results/pseudo_adaptation/cycleformer_ftu_to_${dataset_slug}" \
    --epochs 20 \
    --batch-size 64 \
    --num-workers 2 \
    --lr 0.0001 \
    --temporal-window 5
done
