#!/usr/bin/env bash
# Batch 3 (2026-11-13): layer 2 decides on its first ~12 input spikes, when a third of the windows have reported, and
# learns slowly (see new_architecture_testing/snn_multi_layered_unshared/layer2_analysis/README.md). These runs make it
# wait for more of the code (--l2-drive below the default 1.2) and/or imprint more per win (--l2-lr-mult).
# Same regime as batch 2 (6 epochs, layer-2 theta decay tau = 30k examples, tie-break in both layers, snapshots every
# 200 batches), so batch 2's control (65.1 %) and single-layer reference (85.9 %) are the baselines; they are not rerun.
# Run from anywhere:  bash unity/submit_multi_layer.sh
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
COMMON="--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --seed 0"
sbatch -t 05:00:00 -J ml7_drv0.6           unity/run.sbatch $S $COMMON --l2-drive 0.6                       # decide on ~2/3 of the code
sbatch -t 05:00:00 -J ml7_drv0.36          unity/run.sbatch $S $COMMON --l2-drive 0.36                      # ... on ~5/6 of the code
sbatch -t 05:00:00 -J ml7_lr3              unity/run.sbatch $S $COMMON --l2-lr-mult 3                       # learning rate alone
sbatch -t 05:00:00 -J ml7_drv0.6_lr6       unity/run.sbatch $S $COMMON --l2-drive 0.6 --l2-lr-mult 6        # both: fewer winner spikes per presentation -> larger rate
sbatch -t 05:00:00 -J ml7_drv0.36_lr10     unity/run.sbatch $S $COMMON --l2-drive 0.36 --l2-lr-mult 10
sbatch -t 05:00:00 -J ml7_rd17_drv0.6_lr6  unity/run.sbatch $S $COMMON --l1-readout-inhibition 17.5 --l2-drive 0.6 --l2-lr-mult 6   # on the best code so far
squeue --me
# afterwards, on the login node:
#   conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py
