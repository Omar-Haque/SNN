#!/usr/bin/env bash
# Batch 2 (2026-11-06): does the slow layer-2 climb come from the low within-class overlap of the layer-1 code?
# All stage-B runs: 6 epochs with layer-2 theta decay tau = 30k examples (the stationary regime of test2), tie-break in both
# layers, snapshots every 200 batches. Run from anywhere:  bash unity/submit_multi_layer.sh
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
COMMON="--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200"
sbatch -t 05:00:00 -J ml6_wta_s0   unity/run.sbatch $S $COMMON --seed 0                                   # control: time alone (+ stationary theta)
sbatch -t 05:00:00 -J ml6_ref      unity/run.sbatch $S --no-l1 --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --seed 0   # single-layer reference, same regime
sbatch -t 05:00:00 -J ml6_f8       unity/run.sbatch $S $COMMON --features 8 --seed 0                      # coarser partition per window
sbatch -t 05:00:00 -J ml6_f4       unity/run.sbatch $S $COMMON --features 4 --seed 0
sbatch -t 05:00:00 -J ml6_rd17     unity/run.sbatch $S $COMMON --l1-readout-inhibition 17.5 --seed 0      # softer readout: a runner-up feature fires too
sbatch -t 05:00:00 -J ml6_f8_rd17  unity/run.sbatch $S $COMMON --features 8 --l1-readout-inhibition 17.5 --seed 0
squeue --me
# afterwards, on the login node:
#   conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py
#   python new_architecture_testing/snn_multi_layered_unshared/replay_layer1.py results/<run>   # code statistics of any finished run
