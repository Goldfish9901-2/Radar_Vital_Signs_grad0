#!/usr/bin/env bash
set -euo pipefail

cd /Radar_Vital_Signs

mkdir -p experiment_runs/cycleformer_2026-06-23/results/signal_baselines

for dataset in FTU PhysDrive BGT60TR13C; do
  dataset_slug="$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')"
  python3 src/training/evaluate_signal_baselines.py \
    --method fft \
    --export-dir training_exports \
    --target-datasets "${dataset}" \
    --split test \
    --output-json "experiment_runs/cycleformer_2026-06-23/results/signal_baselines/fft_${dataset_slug}_test.json"

  python3 src/training/evaluate_signal_baselines.py \
    --method stft \
    --export-dir training_exports \
    --target-datasets "${dataset}" \
    --split test \
    --output-json "experiment_runs/cycleformer_2026-06-23/results/signal_baselines/stft_${dataset_slug}_test.json"
done
