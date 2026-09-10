#!/usr/bin/env bash
# One-time: migrate the 5 existing FTU-trained backbones into the uniform
# model_outputs/$model/$dataset/ layout used by the 12x3 queue + v3 benchmark.
# Also remove smoke-test scratch dirs. Safe to re-run (mv is a no-op if target exists).
set -u
cd "$(dirname "$0")/.."
ROOT=model_outputs

for m in dlinear nlinear tsmixer contiformer mamba; do
  src="$ROOT/$m"
  dst="$ROOT/$m/FTU"
  if [ -f "$src/predictions_test.csv" ] && [ ! -d "$dst" ]; then
    mkdir -p "$dst"
    # move everything except a possible FTU subdir already there
    for f in "$src"/*; do
      base=$(basename "$f")
      [ "$base" = "FTU" ] && continue
      mv "$f" "$dst/" 2>/dev/null || true
    done
    echo "moved $m -> $m/FTU"
  else
    echo "skip $m (no predictions or already migrated)"
  fi
done

# remove smoke scratch
rm -rf "$ROOT/_smoke" "$ROOT/_smoke_ds"
echo "cleaned smoke dirs"
ls "$ROOT"
