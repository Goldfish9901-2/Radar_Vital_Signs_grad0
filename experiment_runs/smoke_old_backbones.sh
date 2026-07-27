#!/usr/bin/env bash
# Smoke test the 7 "older" backbones under the CURRENT code/representation to
# confirm they (a) build, (b) forward+backward under dual time+freq, and
# (c) emit predictions_test.csv. Tiny batch + 2 epochs + limit-batches=8 so this
# finishes in minutes and cannot OOM the 4 GB GPU. Used to lock batch sizes before
# the full 12x3 queue.
set -u
cd "$(dirname "$0")/.."
EXPORT=/home/agent-dev-radar/radar/work/run/upstream/training_exports
SMOKE=model_outputs/_smoke
RUN="uv run python src/training/train_model.py"

MODELS="cycleformer heart_timemixer patchtst tcn timesnet transformer tslanet"

for m in $MODELS; do
  odir="$SMOKE/$m"
  rm -rf "$odir"
  mkdir -p "$odir"
  echo "================ [$(date +%H:%M:%S)] SMOKE $m ================"
  $RUN --model "$m" --datasets FTU --epochs 2 --batch-size 8 \
       --seed 42 --lr 3e-4 --weight-decay 1e-4 \
       --export-dir "$EXPORT" --output-dir "$odir" --dump-predictions --limit-batches 8 \
       > "$odir/smoke.log" 2>&1 \
    && echo "================ [$(date +%H:%M:%S)] OK    $m ================" \
    || echo "################ [$(date +%H:%M:%S)] FAIL  $m (see $odir/smoke.log)"
done
echo "SMOKE DONE at $(date)"
