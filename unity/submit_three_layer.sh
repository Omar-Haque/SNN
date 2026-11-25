#!/usr/bin/env bash
# Three-layer batch 2 (2026-11-25): 19 jobs. Batch 1 (73-80 % vs the two-layer network's 85-87 %) showed that layer 2 learns
# sharp, specific stroke combinations (F6, spike-triggered averages) but hands layer 3 only ~20 channels per image (one
# feature per block) where the two-layer top layer heard 343, and accuracy followed the channel count. This batch gives layer 3
# more channels -- softer layer-2 readout, overlapping blocks, more features per block, smaller blocks, a skip connection from
# layer 1 -- and tests training layer 2 on the sharper one-feature-per-window layer-1 code. Runs go to
# three_layer_architecture/results/02_batch2_more_channels_6ep_20261125/ (its README.md lists the questions).
#
#   bash unity/submit_three_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_three_layer.sh smoke tl2_skip     # ... only the jobs named
#   bash unity/submit_three_layer.sh smoke-unity        # the default configuration's smoke run as one short Unity job
#   bash unity/submit_three_layer.sh                    # the 19 jobs on Unity
#   bash unity/submit_three_layer.sh tl2_f64 tl2_skip   # only the jobs named (e.g. to resubmit failed ones)
#
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python;
# SMOKE_DEVICE=cpu forces the CPU (default: MPS on a Mac, else CUDA, else CPU). Each writes a log and a run folder (figures
# included) to three_layer_architecture/results/smoke_tests/.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=three_layer_architecture/three_layer.py
OUT=results/02_batch2_more_channels_6ep_20261125

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
T=10:00:00                                      # batch 1 took ~3 h; the largest layer 2 here has 2,592 neurons
# control: batch 1's default with the new code (STA, skip option): should reproduce batch 1's seed-0 run
job tl2_control          $T $COMMON --seed 0
# A. more layer-2 channels: softer layer-2 readout and overlapping blocks (batch 1's two best changes) and both together
job tl2_b3s1_rd8         $T $COMMON --l2-stride 1 --l2-readout-inhibition 8 --seed 0
job tl2_b3s1_rd8_s1      $T $COMMON --l2-stride 1 --l2-readout-inhibition 8 --seed 1
job tl2_rd4              $T $COMMON --l2-readout-inhibition 4 --seed 0
job tl2_b3s1_rd4         $T $COMMON --l2-stride 1 --l2-readout-inhibition 4 --seed 0
# B. a finer layer-2 vocabulary: more features per block
job tl2_f32              $T $COMMON --l2-features 32 --seed 0
job tl2_f64              $T $COMMON --l2-features 64 --seed 0
job tl2_f32_tp0.3        $T $COMMON --l2-features 32 --l2-theta-plus 0.3 --seed 0     # stronger conscience: keep the extra features in use
job tl2_f32_rd8          $T $COMMON --l2-features 32 --l2-readout-inhibition 8 --seed 0
# C. smaller blocks: 2x2 layer-1 windows (9x9 px) at stride 1 -> 10x10 = 100 blocks
job tl2_b2s1             $T $COMMON --l2-block 2 --l2-stride 1 --seed 0
job tl2_b2s1_rd8         $T $COMMON --l2-block 2 --l2-stride 1 --l2-readout-inhibition 8 --seed 0
# D. skip connection: layer 3 reads layer 1 (343 channels per image) as well as layer 2
job tl2_skip             $T $COMMON --l3-skip --seed 0
job tl2_skip_s1          $T $COMMON --l3-skip --seed 1
job tl2_skip_rd8         $T $COMMON --l3-skip --l2-readout-inhibition 8 --seed 0
job tl2_skip_b3s1_rd8    $T $COMMON --l3-skip --l2-stride 1 --l2-readout-inhibition 8 --seed 0
# E. layer 2 learns on the sharper one-feature-per-window layer-1 code (200 mV); layer 1 is read at 8 mV in stage B
job tl2_a2inh200         $T $COMMON --l1-a2-inhibition 200 --seed 0
job tl2_a2inh200_rd8     $T $COMMON --l1-a2-inhibition 200 --l2-readout-inhibition 8 --seed 0
# F. combinations: many channels and a finer vocabulary
job tl2_b3s1_f32_rd8     $T $COMMON --l2-stride 1 --l2-features 32 --l2-readout-inhibition 8 --seed 0
job tl2_b2s1_f32_rd8     $T $COMMON --l2-block 2 --l2-stride 1 --l2-features 32 --l2-readout-inhibition 8 --seed 0
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
