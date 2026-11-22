#!/usr/bin/env bash
# Batch 5 (2026-11-22): layer 1. Batch 4 put every layer-2 variant at 82-84 % (single layer 85.9 %), all with the same
# ~24-point loss on shifted digits, so what is left points at layer 1's code: each window outputs the feature that crosses
# threshold first, the features are near-copies of each other, and the race between them is often decided by Poisson
# noise or a 1-px shift (layer1_analysis/README.md, sections 6-7). Six layer-1 changes on top of batch 4's layer-2 setting
# (--l2-drive 0.36 --l2-lr-mult 10), one seed, same regime as batches 2-4. Runs go to
# new_architecture_testing/snn_multi_layered_unshared/results_2/06_batch5_layer1_code_stability_6ep_20261122/
# (its README.md lists the questions).
#   bash unity/submit_multi_layer.sh smoke     # optional: one short job on tiny settings (stride 2) -> results/smoke_tests/
#   bash unity/submit_multi_layer.sh           # the six jobs
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
BEST="--l2-drive 0.36 --l2-lr-mult 10"          # batch 3/4's best layer-2 setting
OUT=results_2/06_batch5_layer1_code_stability_6ep_20261122

if [ "${1:-}" = "smoke" ]; then
    sbatch -p gpu-preempt -t 00:30:00 -J ml7_smoke unity/run.sbatch $S --smoke --l1-wta --l2-wta --theta-tau 30000 $BEST --stride 2 --results-dir results/smoke_tests
    squeue --me
    exit 0
fi

COMMON="--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --seed 0 $BEST --results-dir $OUT"
# --l1-wmax is scaled with --l1-norm so templates learn the same shape and only the drive changes
sbatch -t 05:00:00 -J ml7_l1norm6   unity/run.sbatch $S $COMMON --l1-norm 6 --l1-wmax 0.75      # layer-1 drive x0.5: races decided on more input
sbatch -t 05:00:00 -J ml7_l1norm8   unity/run.sbatch $S $COMMON --l1-norm 8 --l1-wmax 1.0       # layer-1 drive x0.67
sbatch -t 05:00:00 -J ml7_rd8       unity/run.sbatch $S $COMMON --l1-readout-inhibition 8       # softer readout than 17.5 mV
sbatch -t 05:00:00 -J ml7_rd50      unity/run.sbatch $S $COMMON --l1-readout-inhibition 50      # harder readout than 17.5 mV
sbatch -t 05:00:00 -J ml7_l1tp0.03  unity/run.sbatch $S $COMMON --l1-theta-plus 0.03            # conscience /3: fewer near-copy features
sbatch -t 08:00:00 -J ml7_stride2   unity/run.sbatch $S $COMMON --stride 2                      # 121 windows instead of 64: more overlap
squeue --me
# afterwards, on the login node (from the repo root):
#   conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
#       --results new_architecture_testing/snn_multi_layered_unshared/results_2
