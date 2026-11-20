#!/usr/bin/env bash
# Batch 4 (2026-11-20): build on batch 3's best run (--l2-drive 0.36 --l2-lr-mult 10: 82.3 %; single layer 85.9 %).
# Six jobs: a second seed of it, the 17.5 mV layer-1 readout added to it (two seeds), 12 epochs instead of 6, one more
# step of the drive / learning-rate trend, and 2x2 pooling for shift tolerance. Same regime as batches 2-3 (6 epochs,
# layer-2 theta decay tau = 30k examples, tie-break in both layers, snapshots every 200 batches); the baselines are not
# rerun. Runs go to new_architecture_testing/snn_multi_layered_unshared/results_2/ (its README.md lists the questions).
# This is the first batch with the stage-A accuracy code, so check it once on tiny settings first:
#   bash unity/submit_multi_layer.sh smoke     # one short job -> results/smoke_tests/; check its log in unity/logs/
#   bash unity/submit_multi_layer.sh           # the six jobs
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
BEST="--l2-drive 0.36 --l2-lr-mult 10"          # batch 3's best layer-2 setting

if [ "${1:-}" = "smoke" ]; then
    sbatch -p gpu-preempt -t 00:30:00 -J ml7_smoke unity/run.sbatch $S --smoke --l1-wta --l2-wta --theta-tau 30000 $BEST --results-dir results/smoke_tests
    squeue --me
    exit 0
fi

COMMON="--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --results-dir results_2"
sbatch -t 05:00:00 -J ml7_best_s1       unity/run.sbatch $S $COMMON $BEST --seed 1                                # is 82.3 % real?
sbatch -t 05:00:00 -J ml7_best_rd17     unity/run.sbatch $S $COMMON $BEST --l1-readout-inhibition 17.5 --seed 0   # do the two improvements add up?
sbatch -t 05:00:00 -J ml7_best_rd17_s1  unity/run.sbatch $S $COMMON $BEST --l1-readout-inhibition 17.5 --seed 1   # ... on a second seed
sbatch -t 08:00:00 -J ml7_best_ep12     unity/run.sbatch $S $COMMON $BEST --epochs 12 --seed 0                    # is the gap training time? (~4 h)
sbatch -t 05:00:00 -J ml7_drv0.24_lr15  unity/run.sbatch $S $COMMON --l2-drive 0.24 --l2-lr-mult 15 --seed 0     # one more step of the trend
sbatch -t 05:00:00 -J ml7_best_pool2    unity/run.sbatch $S $COMMON $BEST --pool2 --seed 0                        # shift tolerance through pooling
squeue --me
# afterwards, on the login node (from the repo root):
#   conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
#       --results new_architecture_testing/snn_multi_layered_unshared/results_2
