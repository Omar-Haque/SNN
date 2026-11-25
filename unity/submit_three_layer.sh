#!/usr/bin/env bash
# Three-layer batch 1 (2026-11-25): first runs of three_layer_architecture/three_layer.py -- layer 1 = the two-layer study's
# best (stride 2, 8 mV readout), layer 2 = NEW local mid-level layer on blocks of layer-1 windows, layer 3 = the D&C classifier
# (drive 0.36, lr x10). Default layer 2: 3x3 blocks at stride 2 (25 blocks x 16 = 400 neurons). Six jobs: the default on two
# seeds, then one change each: layer-2 readout at 8 mV, more block overlap (stride 1), bigger blocks (5x5), faster layer-2
# learning. Runs go to three_layer_architecture/results/01_batch1_first_three_layer_6ep_20261125/ (its README.md lists the
# questions).
#
#   bash unity/submit_three_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_three_layer.sh smoke tl_b5s2      # ... only the jobs named
#   bash unity/submit_three_layer.sh smoke-unity        # the default configuration's smoke run as one short Unity job
#   bash unity/submit_three_layer.sh                    # the six jobs on Unity
#   bash unity/submit_three_layer.sh tl_l2rd8 tl_b5s2   # only the jobs named (e.g. to resubmit failed ones)
#
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python;
# SMOKE_DEVICE=cpu forces the CPU (default: MPS on a Mac, else CUDA, else CPU). Each writes a log and a run folder (figures
# included) to three_layer_architecture/results/smoke_tests/.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=three_layer_architecture/three_layer.py
OUT=results/01_batch1_first_three_layer_6ep_20261125

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
    printf "smoke %-15s ... " "$name"
    # the job's own flags, then --smoke and the smoke results folder (later flags win, so --results-dir is overridden)
    if (cd three_layer_architecture && "$PY" -u three_layer.py "$@" --smoke --results-dir results/smoke_tests ${SMOKE_DEVICE:+--device "$SMOKE_DEVICE"}) \
            > "$log" 2>&1 && grep -q "run finished" "$log"; then
        echo "ok ($((SECONDS - t0)) s)"; PASSED="$PASSED$name "
    else
        echo "FAILED after $((SECONDS - t0)) s -- last lines of $log:"; tail -n 15 "$log" | sed 's/^/    /'; FAILED="$FAILED$name "
    fi
}
job tl_default     08:00:00 $COMMON --seed 0                              # the default three-layer network
job tl_default_s1  08:00:00 $COMMON --seed 1                              # ... second seed (the two-layer seeds differed by 2 points)
job tl_l2rd8       08:00:00 $COMMON --l2-readout-inhibition 8 --seed 0    # layer 1's 8 mV lesson applied to layer 2
job tl_b3s1        08:00:00 $COMMON --l2-stride 1 --seed 0                # 9x9 = 81 overlapping blocks (1,296 neurons): layer 1's stride lesson
job tl_b5s2        08:00:00 $COMMON --l2-block 5 --seed 0                 # 5x5 windows = 15x15 px blocks, 4x4 = 16 blocks (256 neurons)
job tl_l2lr5       08:00:00 $COMMON --l2-lr-mult 5 --seed 0               # layer 2 learns 5x faster (its inputs are sparser than pixels)
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
