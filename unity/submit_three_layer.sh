#!/usr/bin/env bash
# Three-layer batch 3 (2026-11-26): 13 jobs around batch 2's best layer 2 -- trained on layer 1's one-feature-per-window
# code (layer 1 at 200 mV during stage A2) and read out at 8 mV: 84.2 % with only 400 layer-2 neurons, one seed (the
# many-channel runs plateaued at ~83 %; the skip connection gave 85.1 / 84.7 %). This batch replicates it (two more seeds),
# asks why it works (layer 1 at 200 mV throughout; layer 2 trained on the 17.5 mV code), and combines it with batch 2's other
# levers (4 mV readout, overlapping / smaller blocks, skip connection, 32 features, faster layer-2 learning). Runs go to
# three_layer_architecture/results/03_batch3_sparse_trained_layer2_6ep_20261126/ (its README.md lists the questions).
#
#   bash unity/submit_three_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_three_layer.sh smoke tl3_a2r8_skip  # ... only the jobs named
#   bash unity/submit_three_layer.sh smoke-unity        # the default configuration's smoke run as one short Unity job
#   bash unity/submit_three_layer.sh                    # the 13 jobs on Unity
#   bash unity/submit_three_layer.sh tl3_a2r4 tl3_a2r8_s1   # only the jobs named (e.g. to resubmit failed ones)
#
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python;
# SMOKE_DEVICE=cpu forces the CPU (default: MPS on a Mac, else CUDA, else CPU). Each writes a log and a run folder (figures
# included) to three_layer_architecture/results/smoke_tests/.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=three_layer_architecture/three_layer.py
OUT=results/03_batch3_sparse_trained_layer2_6ep_20261126

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
T=10:00:00                                      # batch 2 took ~3 h per job (two jobs on a slow node took 6.6 / >10 h)
A2R8="--l1-a2-inhibition 200 --l2-readout-inhibition 8"   # batch 2's best: layer 2 learns on the 200 mV layer-1 code, read out at 8 mV
# replication (batch 2, seed 0: 84.2 %)
job tl3_a2r8_s1          $T $COMMON $A2R8 --seed 1
job tl3_a2r8_s2          $T $COMMON $A2R8 --seed 2
# why it works
job tl3_l1r200_r8        $T $COMMON --l1-readout-inhibition 200 --l2-readout-inhibition 8 --seed 0   # layer 1 at 200 mV throughout: no switch
job tl3_a2inh17_r8       $T $COMMON --l1-a2-inhibition 17.5 --l2-readout-inhibition 8 --seed 0      # layer 2 learns on the 17.5 mV code (~2 features / window)
# combined with batch 2's other levers
job tl3_a2r4             $T $COMMON --l1-a2-inhibition 200 --l2-readout-inhibition 4 --seed 0
job tl3_a2r8_b3s1        $T $COMMON $A2R8 --l2-stride 1 --seed 0                         # 81 overlapping blocks
job tl3_a2r4_b3s1        $T $COMMON --l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-stride 1 --seed 0
job tl3_a2r8_b2s1        $T $COMMON $A2R8 --l2-block 2 --l2-stride 1 --seed 0            # 100 blocks of 2x2 windows
job tl3_a2r8_skip        $T $COMMON $A2R8 --l3-skip --seed 0                              # + layer 1 straight to layer 3
job tl3_a2r8_skip_s1     $T $COMMON $A2R8 --l3-skip --seed 1
job tl3_a2r8_b3s1_skip   $T $COMMON $A2R8 --l2-stride 1 --l3-skip --seed 0
job tl3_a2r8_f32         $T $COMMON $A2R8 --l2-features 32 --seed 0                       # trained on a sparser code, extra features may survive
job tl3_a2r8_l2lr5       $T $COMMON $A2R8 --l2-lr-mult 5 --seed 0
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
