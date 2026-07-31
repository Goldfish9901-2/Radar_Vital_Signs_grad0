#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

model_dir="experiment_runs/cycleformer_2026-06-23/results/deep_models/cycleformer_ftu_source"

for dataset in FTU PhysDrive BGT60TR13C; do
  dataset_slug="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
  python3 src/training/evaluate_model.py \
    --model-dir "${model_dir}" \
    --export-dir training_exports \
    --target-datasets "${dataset}" \
    --split test \
    --batch-size 128 \
    --output-json "experiment_runs/cycleformer_2026-06-23/results/source_only/cycleformer_ftu_to_${dataset_slug}_test.json"
done
