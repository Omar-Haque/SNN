#!/usr/bin/env bash
# results_for_cnn_replication, batch 2 (2026-11-28), three-layer part (the two-layer part: unity/submit_multi_layer.sh; both
# write to new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication/02_batch2_parts_and_k_6ep_20261128/,
# whose README.md lists the questions). (1) k = 2 / 3 winners per image in layer 3 (new --l3-k), on the best three-layer network
# (batch 3's tl3_a2r4_b3s1: 86.0 %, one seed) and that network's second seed. (2) Local receptive fields at the top: layer 2
# with large blocks (15-23 px) and many features per block -- a local layer whose neurons can only learn parts -- scored on its
# own at the end of stage A2 ("layer 2 alone"), with layer 3 trained on top as usual. Three-layer batch 3's version of this
# script: git show 22255be:unity/submit_three_layer.sh
#
#   bash unity/submit_three_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_three_layer.sh smoke t2_lrf7      # ... only the jobs named
#   bash unity/submit_three_layer.sh smoke-unity        # the default configuration's smoke run as one short Unity job
#   bash unity/submit_three_layer.sh                    # the 10 jobs on Unity
#   bash unity/submit_three_layer.sh t2_k2 t2_lrf9      # only the jobs named (e.g. to resubmit failed ones)
#
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python;
# SMOKE_DEVICE=cpu forces the CPU (default: MPS on a Mac, else CUDA, else CPU). Each writes a log and a run folder (figures
# included) to three_layer_architecture/results/smoke_tests/.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=three_layer_architecture/three_layer.py
OUT=../new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication/02_batch2_parts_and_k_6ep_20261128   # relative to three_layer_architecture/

MODE=unity
if [ "${1:-}" = "smoke-unity" ]; then
    sbatch -p gpu-preempt -t 00:30:00 -J tl_smoke unity/run.sbatch $S --smoke --results-dir results/smoke_tests
    squeue --me
    exit 0
fi
if [ "${1:-}" = "smoke" ]; then
    MODE=local; shift
    PY=${PYTHON:-python}
    command -v "$PY" >/dev/null 2>&1 || PY=python3
    if [[ "$PY" == */* ]]; then PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"; fi    # absolute: the runs cd into the script's folder
    if ! "$PY" -c "import torch, torchvision, snntorch" >/dev/null 2>&1; then
        echo "torch / torchvision / snntorch cannot be imported with '$PY'. Activate the project's environment, or run"
        echo "  PYTHON=/path/to/python bash unity/submit_three_layer.sh smoke"
        exit 1
    fi
    LOGS=three_layer_architecture/results/smoke_tests/logs; mkdir -p "$LOGS"
    echo "local smoke runs with $("$PY" -c 'import sys, torch; print(sys.executable, "| torch", torch.__version__)')"
fi

COMMON="--results-dir $OUT"                     # everything else is three_layer.py's defaults (see its README)
WANT=" $* "; DONE=" "; PASSED=" "; FAILED=" "
job() {   # job NAME TIME ARGS...: run / submit unless job names were given on the command line and NAME is not one of them
    local name=$1 t=$2; shift 2
    if [ -n "${WANT// /}" ] && [[ "$WANT" != *" $name "* ]]; then return; fi
    DONE="$DONE$name "
    if [ "$MODE" = unity ]; then
        sbatch -t "$t" -J "$name" unity/run.sbatch $S "$@"
        return
    fi
    local log="$LOGS/$name.log" t0=$SECONDS
    printf "smoke %-20s ... " "$name"
    # the job's own flags, then --smoke and the smoke results folder (later flags win, so --results-dir is overridden)
    if (cd three_layer_architecture && "$PY" -u three_layer.py "$@" --smoke --results-dir results/smoke_tests ${SMOKE_DEVICE:+--device "$SMOKE_DEVICE"}) \
            > "$log" 2>&1 && grep -q "run finished" "$log"; then
        echo "ok ($((SECONDS - t0)) s)"; PASSED="$PASSED$name "
    else
        echo "FAILED after $((SECONDS - t0)) s -- last lines of $log:"; tail -n 15 "$log" | sed 's/^/    /'; FAILED="$FAILED$name "
    fi
}
T=10:00:00                                      # batch 3: 2.7-3.7 h per job
B3="--l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-stride 1"   # batch 3's best: layer 2 learns on the 200 mV layer-1 code, overlapping blocks, read at 4 mV
# (1) k winners per image in layer 3
job t2_best_s1  $T $COMMON $B3 --seed 1                                  # the best network's second seed (seed 0: 86.0 %)
job t2_k2       $T $COMMON $B3 --l3-k 2 --seed 0
job t2_k2s1     $T $COMMON $B3 --l3-k 2 --seed 1
job t2_k3       $T $COMMON $B3 --l3-k 3 --seed 0
job t2_k3s1     $T $COMMON $B3 --l3-k 3 --seed 1
# (2) local receptive fields at the top: large layer-2 blocks with many features; "layer 2 alone" is the local top layer's score
LRF="--l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-examples 60000"
job t2_lrf5     $T $COMMON $LRF --l2-block 5 --l2-stride 2 --l2-features 25 --seed 0   # 4x4 blocks of 15x15 px x 25 = 400 neurons
job t2_lrf7     $T $COMMON $LRF --l2-block 7 --l2-stride 2 --l2-features 48 --seed 0   # 3x3 blocks of 19x19 px x 48 = 432
job t2_lrf7f96  $T $COMMON $LRF --l2-block 7 --l2-stride 2 --l2-features 96 --seed 0   # the same with 96 features per block (864)
job t2_lrf9     $T $COMMON $LRF --l2-block 9 --l2-stride 1 --l2-features 48 --seed 0   # 3x3 blocks of 23x23 px x 48 = 432
job t2_lrf7k2   $T $COMMON $LRF --l2-block 7 --l2-stride 2 --l2-features 48 --l3-k 2 --seed 0   # + k = 2 in layer 3
for n in $*; do [[ "$DONE" == *" $n "* ]] || echo "no job named $n in this batch"; done
if [ "$MODE" = local ]; then
    echo; echo "passed:${PASSED% }"; echo "failed:${FAILED% }"
    echo "logs and run folders: three_layer_architecture/results/smoke_tests/"
    [ -z "${FAILED// /}" ] && [ -n "${PASSED// /}" ]
    exit
fi
squeue --me
# afterwards, on the login node (from the repo root):
#   conda activate snn && python three_layer_architecture/summarize_three_layer.py --results three_layer_architecture/results
