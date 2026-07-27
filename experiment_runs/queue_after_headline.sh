#!/usr/bin/env bash
# Chained queue that runs AFTER the headline 12x3 finishes.
#
# Stage 1: queue_all.sh -- now contains all 12 headline backbones PLUS the two
#   new backbones (xlstm, frets). Because of skip-done logic, the 12 headline
#   cells are skipped (already trained) and only the 6 xlstm/frets cells run.
#   It also regenerates the v3 benchmark (now with 14 backbones).
# Stage 2: run_representation_ablation.sh -- the factorial representation
#   ablation (4 backbones x 5 representations x FTU = 20 cells). It builds each
#   representation's window export inline, then trains. After it finishes, run
#   experiment_runs/make_ablation_v3.py to get the backbone-vs-representation
#   variance decomposition.
#
# Launch (unattended):
#   systemctl --user restart radar-benchmark.service
# (the service ExecStart points here)
set -u
cd "$(dirname "$0")/.."

echo "================ [$(date +%H:%M:%S)] STAGE 1: xlstm + frets (12 headline skipped) ================"
bash experiment_runs/queue_all.sh

echo "================ [$(date +%H:%M:%S)] STAGE 2: representation ablation (4x5 factorial) ================"
bash experiment_runs/run_representation_ablation.sh

echo "================ [$(date +%H:%M:%S)] STAGE 3: ablation variance decomposition ================"
uv run python experiment_runs/make_ablation_v3.py

echo "ALL NEXT-QUEUE DONE at $(date)"
