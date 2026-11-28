#!/usr/bin/env bash
# results_for_cnn_replication, batch 2 (2026-11-28), two-layer part (the three-layer part: unity/submit_three_layer.sh; both
# write to the same folder). Batch 1 found that k = 2-3 winners per image in layer 2 gives 89.2 / 88.7 % (baseline 85.3, one
# seed) but no part-like features: every neuron still learns a whole digit. This batch (1) replicates k = 2 / 3 on seeds 1-2,
# (2) tries k = 2 / 3 on Fashion-MNIST with input normalisation, and (3) tests two mechanisms aimed at parts in layer 2:
# explaining away (new --l2-explain, with k winners) and learned lateral inhibition (new --l2-lat-lr). Runs go to
# new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication/02_batch2_parts_and_k_6ep_20261128/ (its
# README.md lists the questions). Batch 1's version of this script: git show 22255be:unity/submit_multi_layer.sh
#
#   bash unity/submit_multi_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_multi_layer.sh smoke c2_ea3       # ... only the jobs named
#   bash unity/submit_multi_layer.sh                    # the 18 jobs on Unity
#   bash unity/submit_multi_layer.sh c2_ea3 c2_k2s1     # only the jobs named (e.g. to resubmit failed ones)
#
# Fashion-MNIST is downloaded to <repo>/data/FashionMNIST on first use. Unity compute nodes may have no internet, so in Unity
# mode this script downloads it first, here on the login node (with the snn environment active), before submitting.
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
SDIR=$(dirname "$S")
OUT=results_for_cnn_replication/02_batch2_parts_and_k_6ep_20261128

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
T=06:00:00                                      # batch 1: 1.9-3.4 h per job
TE=10:00:00                                     # explaining away does extra work every millisecond
# (1) replication on MNIST: batch 1 (seed 0) gave 85.3 (one winner), 89.2 (k = 2), 88.7 (k = 3); batch 6 seed 1: 87.3 (one winner)
job c2_bs2    $T $COMMON $BEST --seed 2
job c2_k2s1   $T $COMMON $BEST --l2-k 2 --seed 1
job c2_k2s2   $T $COMMON $BEST --l2-k 2 --seed 2
job c2_k3s1   $T $COMMON $BEST --l2-k 3 --seed 1
job c2_k3s2   $T $COMMON $BEST --l2-k 3 --seed 2
# (2) Fashion-MNIST with input normalisation: batch 8 gave 72.9 / 73.3 % with one winner (seeds 0 / 1)
job c2_fk2    $T $COMMON $BEST --dataset fmnist --input-norm --l2-k 2 --seed 0
job c2_fk2s1  $T $COMMON $BEST --dataset fmnist --input-norm --l2-k 2 --seed 1
job c2_fk3    $T $COMMON $BEST --dataset fmnist --input-norm --l2-k 3 --seed 0
job c2_fk3s1  $T $COMMON $BEST --dataset fmnist --input-norm --l2-k 3 --seed 1
# (3a) explaining away: each winner claims the input its template covers; the next winner must match what is left
job c2_ea2    $TE $COMMON $BEST --l2-k 2 --l2-explain --seed 0
job c2_ea3    $TE $COMMON $BEST --l2-k 3 --l2-explain --seed 0
job c2_ea5    $TE $COMMON $BEST --l2-k 5 --l2-explain --seed 0
job c2_ea5s1  $TE $COMMON $BEST --l2-k 5 --l2-explain --seed 1
job c2_ea10   $TE $COMMON $BEST --l2-k 10 --l2-explain --seed 0
# (3b) learned lateral inhibition: neurons that fire on the same images come to inhibit each other (starting from soft inhibition,
#      where ~7 neurons share each digit; batch 1's 20 mV run gave 85.5 %)
job c2_lat5   $T $COMMON $BEST --inhibition 20 --l2-lat-lr 5 --seed 0
job c2_lat20  $T $COMMON $BEST --inhibition 20 --l2-lat-lr 20 --seed 0
job c2_l20i8  $T $COMMON $BEST --inhibition 8 --l2-lat-lr 20 --seed 0      # start softer: more co-firing to learn from
job c2_l20p5  $T $COMMON $BEST --inhibition 20 --l2-lat-lr 20 --l2-lat-target 0.005 --seed 0   # sparser target: ~2 neurons per image
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
