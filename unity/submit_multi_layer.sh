#!/usr/bin/env bash
# Batch 6 (2026-11-22): batch 5's two winners. Stride 2 (+2.5 points held-out, +2 shifted) and 8 mV layer-1 readout
# inhibition (+2.1 held-out) each brought the two-layer network to within ~1-1.5 points of the single layer, one seed
# each. This batch combines them (two seeds), repeats each on a second seed, pushes the readout to 4 mV, and tries 2x2
# pooling on the 8 mV code (pooling failed in batch 4 on the 200 mV code, whose windows flip under a 1-px shift; at 8 mV
# the code is as shift-stable as pixels). Layer-2 setting and regime as in batches 4-5. Runs go to
# new_architecture_testing/snn_multi_layered_unshared/results_2/07_batch6_stride2_readout8_6ep_20261122/
# (its README.md lists the questions).
#   bash unity/submit_multi_layer.sh smoke     # optional: one short job (stride 2 + 8 mV) on tiny settings -> results/smoke_tests/
#   bash unity/submit_multi_layer.sh           # the six jobs
#   bash unity/submit_multi_layer.sh ml7_rd4 ml7_pool2_rd8   # only the jobs named (e.g. to resubmit failed ones)
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
BEST="--l2-drive 0.36 --l2-lr-mult 10"          # the layer-2 setting of batches 3-5
OUT=results_2/07_batch6_stride2_readout8_6ep_20261122

if [ "${1:-}" = "smoke" ]; then
    sbatch -p gpu-preempt -t 00:30:00 -J ml7_smoke unity/run.sbatch $S --smoke --l1-wta --l2-wta --theta-tau 30000 $BEST --stride 2 --l1-readout-inhibition 8 --results-dir results/smoke_tests
    squeue --me
    exit 0
fi

COMMON="--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 $BEST --results-dir $OUT"
WANT=" $* "; DONE=" "
job() {   # job NAME TIME ARGS...: submit unless job names were given on the command line and NAME is not one of them
    local name=$1 t=$2; shift 2
    if [ -z "${WANT// /}" ] || [[ "$WANT" == *" $name "* ]]; then sbatch -t "$t" -J "$name" unity/run.sbatch $S "$@"; DONE="$DONE$name "; fi
}
job ml7_s2_rd8     05:00:00 $COMMON --stride 2 --l1-readout-inhibition 8 --seed 0   # do the two gains add up?
job ml7_s2_rd8_s1  05:00:00 $COMMON --stride 2 --l1-readout-inhibition 8 --seed 1   # ... on a second seed
job ml7_s2_s1      05:00:00 $COMMON --stride 2 --seed 1                             # stride 2 alone, second seed
job ml7_rd8_s1     05:00:00 $COMMON --l1-readout-inhibition 8 --seed 1              # 8 mV alone, second seed
job ml7_rd4        05:00:00 $COMMON --l1-readout-inhibition 4 --seed 0              # softer still (same layer 1 as 8 / 17.5 / 50 / 200 mV)
job ml7_pool2_rd8  05:00:00 $COMMON --pool2 --l1-readout-inhibition 8 --seed 0      # can pooling buy shift tolerance on a stable code?
for n in $*; do [[ "$DONE" == *" $n "* ]] || echo "no job named $n in this batch"; done
squeue --me
# --pool2 needs an even window grid: it works at stride 3 (8x8 windows), not at stride 2 (11x11).
# afterwards, on the login node (from the repo root):
#   conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
#       --results new_architecture_testing/snn_multi_layered_unshared/results_2
