#!/usr/bin/env bash
# Train the 5 newly-integrated backbones under the EXACT same FTU->FTU protocol
# used by the June baselines: --datasets FTU, 80 epochs, lr=3e-4, seed=42,
# SmoothL1Loss(beta=0.5), early-stop patience 12. All run in dual (time+freq)
# mode to match the proposed HR-AdaVMD representation.
#
# Batch sizes are adjusted only where the 4 GB GPU would otherwise OOM:
#   dlinear/nlinear/tsmixer -> 64 (as in the protocol)
#   contiformer            -> 32 (L=256 attention)
#   mamba                  -> 16 (pure-PyTorch scan materializes (B,L,d_inner,d_state))
#
# Per-sample predictions are dumped (--dump-predictions) for the unified benchmark
# scatter / histogram / pred-vs-label plots and overall Pearson.
set -u

cd "$(dirname "$0")/.."   # work/upstream_rw

EXPORT=/home/agent-dev-radar/radar/work/run/upstream/training_exports
OUT=model_outputs
EPOCHS=80
SEED=42
RUN="uv run python src/training/train_model.py"

train() {
  local model="$1"; local bs="$2"; local odir="$3"
  if [ -f "$odir/predictions_test.csv" ] && [ -f "$odir/best.pt" ]; then
    echo "================ SKIP $model (already trained) ================"
    return 0
  fi
  mkdir -p "$odir"
  echo "================ [$(date +%H:%M:%S)] START $model (bs=$bs) ================"
  $RUN --model "$model" --datasets FTU --epochs "$EPOCHS" --batch-size "$bs" \
       --seed "$SEED" --lr 3e-4 --weight-decay 1e-4 \
       --export-dir "$EXPORT" --output-dir "$odir" --dump-predictions --resume \
       > "$odir/train.log" 2>&1 \
    && echo "================ [$(date +%H:%M:%S)] DONE  $model ================" \
    || { echo "################ [$(date +%H:%M:%S)] FAILED $model (see $odir/train.log)"; return 1; }
  # Generate this model's independent results immediately (incremental benchmark).
  # NOTE: benchmark scripts are standalone entry points, NOT args to train_model.py.
  echo "---------------- [$(date +%H:%M:%S)] benchmark $model ----------------"
  uv run python experiment_runs/benchmark_inference.py >> "$odir/train.log" 2>&1 \
    && uv run python experiment_runs/make_benchmark_v2.py >> "$odir/train.log" 2>&1 \
    && echo "---------------- [$(date +%H:%M:%S)] benchmark updated (incl. $model) ----------------" \
    || echo "################ benchmark step failed for $model (see $odir/train.log)"
}

mkdir -p "$OUT"
train dlinear     64 "$OUT/dlinear"
train nlinear     64 "$OUT/nlinear"
train tsmixer     64 "$OUT/tsmixer"
train contiformer 32 "$OUT/contiformer"
train mamba       16 "$OUT/mamba"

echo "ALL DONE at $(date)"
