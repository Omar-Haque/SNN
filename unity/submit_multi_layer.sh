#!/usr/bin/env bash
# results_for_cnn_replication, batch 1 (2026-11-27): several winners per image in layer 2. In the best two-layer network
# (batch 6: stride 2, layer 1 read at 8 mV, drive 0.36, lr x10; 85.3 / 87.3 %), layer 2 is a whole-image layer where one
# neuron wins each image, so its neurons learn whole-digit prototypes. A CNN's upper layers learn parts instead. This batch lets
# several layer-2 neurons share each image -- softer lateral inhibition (50 / 20 / 8 mV), or exactly k winners per image (new
# --l2-k: 2 / 3 / 5 / 10) -- and records what each neuron fires for (F16, class purity, distinct neurons per image). MNIST,
# same regime as batches 6-8. Runs go to
# new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication/01_batch1_several_winners_6ep_20261127/ (its
# README.md lists the questions; the README one folder up explains the aim). Batch 8's version of this script:
# git show 1d7b3b8:unity/submit_multi_layer.sh
#
#   bash unity/submit_multi_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_multi_layer.sh smoke c1_k3        # ... only the jobs named
#   bash unity/submit_multi_layer.sh                    # the 11 jobs on Unity
#   bash unity/submit_multi_layer.sh c1_k3 c1_k5s1      # only the jobs named (e.g. to resubmit failed ones)
#
# Fashion-MNIST is downloaded to <repo>/data/FashionMNIST on first use. Unity compute nodes may have no internet, so in Unity
# mode this script downloads it first, here on the login node (with the snn environment active), before submitting.
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
SDIR=$(dirname "$S")
OUT=results_for_cnn_replication/01_batch1_several_winners_6ep_20261127

PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
if [[ "$PY" == */* ]]; then PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"; fi      # absolute: local runs cd into the script's folder
MODE=unity
if [ "${1:-}" = "smoke" ]; then
    MODE=local; shift
    if ! "$PY" -c "import torch, torchvision, snntorch" >/dev/null 2>&1; then
        echo "torch / torchvision / snntorch cannot be imported with '$PY'. Activate the project's environment, or run"
        echo "  PYTHON=/path/to/python bash unity/submit_multi_layer.sh smoke"
        exit 1
    fi
    LOGS=$SDIR/results/smoke_tests/logs; mkdir -p "$LOGS"
    echo "local smoke runs with $("$PY" -c 'import sys, torch; print(sys.executable, "| torch", torch.__version__)')"
fi
# Fashion-MNIST present? (both modes: locally it avoids a download inside the first smoke run; on Unity, compute nodes may be offline)
if [ ! -f data/FashionMNIST/raw/t10k-labels-idx1-ubyte ]; then
    echo "downloading Fashion-MNIST to data/FashionMNIST ..."
    if ! "$PY" -c "from torchvision import datasets; [datasets.FashionMNIST('data', train=t, download=True) for t in (True, False)]"; then
        echo "could not download Fashion-MNIST with '$PY'. On Unity: module load conda/latest && conda activate snn, then rerun."
        exit 1
    fi
fi

COMMON="--theta-tau 30000 --epochs 6 --snapshot-interval 200 --results-dir $OUT"   # MNIST, no input normalisation (as batch 6)
SINGLE="--no-l1 --l2-wta"                                                       # batch 2's single-layer reference
BEST="--stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10"   # batch 6's best two-layer network
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
    printf "smoke %-16s ... " "$name"
    # the job's own flags, then --smoke and the smoke results folder (later flags win, so --results-dir is overridden)
    if (cd "$SDIR" && "$PY" -u "$(basename "$S")" "$@" --smoke --results-dir results/smoke_tests) > "$log" 2>&1 && grep -q "run finished" "$log"; then
        echo "ok ($((SECONDS - t0)) s)"; PASSED="$PASSED$name "
    else
        echo "FAILED after $((SECONDS - t0)) s -- last lines of $log:"; tail -n 15 "$log" | sed 's/^/    /'; FAILED="$FAILED$name "
    fi
}
T=06:00:00                                      # batch 8's MNIST two-layer run took 2.4 h
# reference: the best two-layer network unchanged (one winner per image); batch 6 gave 85.3 / 56.5 % on seed 0
job c1_base   $T $COMMON $BEST --seed 0
# softer layer-2 lateral inhibition: the winner's spikes no longer silence everyone, so runner-up neurons can fire too
job c1_inh50  $T $COMMON $BEST --inhibition 50 --seed 0
job c1_inh20  $T $COMMON $BEST --inhibition 20 --seed 0
job c1_inh8   $T $COMMON $BEST --inhibition 8 --seed 0
# exactly k winners per image: the first k neurons to reach threshold share the image, everyone else is silenced
job c1_k2     $T $COMMON $BEST --l2-k 2 --seed 0
job c1_k3     $T $COMMON $BEST --l2-k 3 --seed 0
job c1_k5     $T $COMMON $BEST --l2-k 5 --seed 0
job c1_k10    $T $COMMON $BEST --l2-k 10 --seed 0
job c1_k3tp1  $T $COMMON $BEST --l2-k 3 --theta-plus 1.0 --seed 0   # each neuron now wins ~3x as many images: a stronger threshold rise per spike makes frequent winners give way sooner
# second seeds of a middle setting from each family
job c1_i20s1  $T $COMMON $BEST --inhibition 20 --seed 1
job c1_k5s1   $T $COMMON $BEST --l2-k 5 --seed 1
for n in $*; do [[ "$DONE" == *" $n "* ]] || echo "no job named $n in this batch"; done
if [ "$MODE" = local ]; then
    echo; echo "passed:${PASSED% }"; echo "failed:${FAILED% }"
    echo "logs and run folders: $SDIR/results/smoke_tests/"
    [ -z "${FAILED// /}" ] && [ -n "${PASSED// /}" ]
    exit
fi
squeue --me
# afterwards, on the login node (from the repo root):
#   conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
#       --results new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication
