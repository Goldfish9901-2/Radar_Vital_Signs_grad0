#!/usr/bin/env bash
# Headline queue: train ALL 12 backbones on ALL 3 datasets (FTU, BGT60TR13C,
# PhysDrive) under the canonical FTU->FTU protocol:
#   80 epochs, AdamW lr=3e-4/wd=1e-4, SmoothL1Loss(beta=0.5),
#   ReduceLROnPlateau(patience=4), early-stop patience=12, seed=42,
#   dual time+freq, --dump-predictions --resume.
#
# Output layout (uniform, read by make_benchmark_v3.py):
#   model_outputs/<model>/<DATASET>/{best.pt,final.pt,run_config.json,
#                                     history.json,summary.json,predictions_test.csv}
#
# Batch sizes are set per-model ONLY to avoid OOM on the 4 GB T600. They affect
# throughput, never the model or final quality. Already-completed runs (a dir
# containing both best.pt and predictions_test.csv) are skipped, so the queue is
# safe to re-run / resume after interruption.
#
# Launch (unattended, survives disconnect):
#   systemctl --user start radar-benchmark.service
set -u
cd "$(dirname "$0")/.."   # work/upstream_rw

EXPORT=/home/agent-dev-radar/radar/work/run/upstream/training_exports
OUT=model_outputs
EPOCHS=80
SEED=42
RUN="uv run python src/training/train_model.py"

# model -> batch size (4 GB T600, tuned for headroom)
declare -A BS=(
  [contiformer]=32
  [dlinear]=64
  [nlinear]=64
  [tsmixer]=64
  [mamba]=16
  [cycleformer]=16
  [heart_timemixer]=16
  [patchtst]=32
  [tcn]=32
  [timesnet]=16
  [transformer]=16
  [tslanet]=32
  [xlstm]=16
  [frets]=32
)

DATASETS="FTU BGT60TR13C PhysDrive"
MODELS="contiformer cycleformer dlinear heart_timemixer mamba nlinear patchtst tcn timesnet transformer tslanet tsmixer xlstm frets"

train() {
  local model="$1"; local ds="$2"; local bs="$3"
  local odir="$OUT/$model/$ds"
  if [ -f "$odir/predictions_test.csv" ] && [ -f "$odir/best.pt" ]; then
    echo "================ SKIP $model/$ds (already trained) ================"
    return 0
  fi
  mkdir -p "$odir"
  echo "================ [$(date +%H:%M:%S)] START $model on $ds (bs=$bs) ================"
  $RUN --model "$model" --datasets "$ds" --epochs "$EPOCHS" --batch-size "$bs" \
       --seed "$SEED" --lr 3e-4 --weight-decay 1e-4 \
       --export-dir "$EXPORT" --output-dir "$odir" --dump-predictions --resume \
       > "$odir/train.log" 2>&1 \
    && echo "================ [$(date +%H:%M:%S)] DONE  $model/$ds ================" \
    || { echo "################ [$(date +%H:%M:%S)] FAILED $model/$ds (see $odir/train.log)"; return 1; }
}

mkdir -p "$OUT"
FAIL=0
for ds in $DATASETS; do
  for model in $MODELS; do
    train "$model" "$ds" "${BS[$model]}" || FAIL=1
  done
done

echo "ALL DONE at $(date) (failures=${FAIL})"

# Optional: if a v3 benchmark already exists, regenerate it from the new layout.
if [ -f experiment_runs/make_benchmark_v3.py ]; then
  echo "---------------- regenerating v3 benchmark ----------------"
  uv run python experiment_runs/make_benchmark_v3.py >> "$OUT/_queue.log" 2>&1 \
    && echo "v3 benchmark updated" \
    || echo "################ v3 benchmark step failed (see $OUT/_queue.log)"
fi
