#!/usr/bin/env bash
# Representation (input-feature) ablation: factorial of BACKBONE x REPRESENTATION.
#
# Answers "does performance come from the backbone or the representation?":
#   5 representations (REPRESENTATION_ABLATION_CHOICES) x 4 backbone archetypes
#   spanning the major paradigms (linear / conv / attention / recurrent).
#
# Each representation gets its OWN window export (built from the raw exports with
# --representation X, FTU only — the within-dataset question). Models are auto-built
# to the representation's channel count via train_model.py's representation-derived
# --time-channels/--freq-channels. Output layout:
#   model_outputs_ablation/<model>/<rep>/FTU/{best.pt,predictions_test.csv,...}
#
# Skips already-done (best.pt + predictions_test.csv present) runs, so it is safe
# to re-run / resume. NOTE: the GPU is typically busy with the headline 12x3 queue;
# run this as a SEPARATE queue AFTER that finishes (e.g. its own systemd service),
# or interleave behind it.
#
# Usage:
#   bash experiment_runs/run_representation_ablation.sh
set -u
cd "$(dirname "$0")/.."

RAW_EXPORTS=/home/agent-dev-radar/radar/work/exports
EXPORT_ROOT=/home/agent-dev-radar/radar/work/run/upstream/training_exports_ablation
OUT=model_outputs_ablation
EPOCHS=80
SEED=42
RUN="uv run python src/training/train_model.py"
BUILD="uv run python src/data/build_training_dataset.py"

REPS="proposed edacm_only raw_logmag raw_real_imag edacm_vmd_fixed"
# 4-backbone sanity matrix (per research plan, Message-5): linear / attention / attention /
# attention. contiformer + cycleformer cover the temporal-attention paradigms; dlinear is the
# linear baseline. xlstm and frets are DEFERRED to last (run separately) and intentionally
# excluded here so the first ablation pass follows the named sanity set.
MODELS="dlinear contiformer cycleformer transformer"

# per-model batch size (4 GB T600)
declare -A BS=(
  [dlinear]=64 [contiformer]=16 [cycleformer]=16 [transformer]=16
)

train() {
  local model="$1"; local rep="$2"; local bs="$3"; local exp="$4"
  local odir="$OUT/$model/$rep/FTU"
  if [ -f "$odir/predictions_test.csv" ] && [ -f "$odir/best.pt" ]; then
    echo "================ SKIP $model/$rep/FTU (already trained) ================"
    return 0
  fi
  mkdir -p "$odir"
  echo "================ [$(date +%H:%M:%S)] START $model on $rep (bs=$bs) ================"
  $RUN --model "$model" --datasets FTU --epochs "$EPOCHS" --batch-size "$bs" \
       --seed "$SEED" --lr 3e-4 --weight-decay 1e-4 \
       --export-dir "$exp" --output-dir "$odir" --dump-predictions --resume \
       > "$odir/train.log" 2>&1 \
    && echo "================ [$(date +%H:%M:%S)] DONE  $model/$rep/FTU ================" \
    || { echo "################ [$(date +%H:%M:%S)] FAILED $model/$rep (see $odir/train.log)"; return 1; }
}

mkdir -p "$OUT"
for rep in $REPS; do
  exp="$EXPORT_ROOT/$rep"
  if [ ! -f "$exp/build_config.json" ]; then
    echo "---------------- building window export for rep=$rep ----------------"
    $BUILD --exports-dir "$RAW_EXPORTS" --output-dir "$exp" --datasets FTU \
           --representation "$rep" --overwrite > "$exp/build.log" 2>&1 \
      && echo "export built: $exp" \
      || { echo "################ export FAILED for $rep (see $exp/build.log)"; continue; }
  fi
  for model in $MODELS; do
    train "$model" "$rep" "${BS[$model]}" "$exp" || true
  done
done

echo "ABLATION DONE at $(date)"
