#!/usr/bin/env bash
# Direct Transfer baseline (0号基线):
#   Train HeartTimeMixer on FTU (source), evaluate on PhysDrive + BGT60 (targets).
# Run AFTER the data build (training_exports) is complete.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPORT_DIR=training_exports
OUT=model_outputs/direct_transfer_ftu_source

echo "=== [1/3] Train HeartTimeMixer on FTU (source) ==="
python -u src/training/train_model.py \
  --model heart_timemixer \
  --datasets FTU \
  --export-dir "$EXPORT_DIR" \
  --output-dir "$OUT"

echo "=== [2/3] Evaluate on SOURCE test split (FTU, in-domain reference) ==="
python -u src/training/evaluate_model.py \
  --model-dir "$OUT" \
  --export-dir "$EXPORT_DIR" \
  --target-datasets FTU \
  --split test

echo "=== [3/3] Evaluate on TARGET test splits (Direct Transfer) ==="
python -u src/training/evaluate_model.py \
  --model-dir "$OUT" \
  --export-dir "$EXPORT_DIR" \
  --target-datasets PhysDrive \
  --split test
python -u src/training/evaluate_model.py \
  --model-dir "$OUT" \
  --export-dir "$EXPORT_DIR" \
  --target-datasets BGT60TR13C \
  --split test

echo "=== Done. Eval JSONs saved under $OUT/eval_test_*.json ==="
