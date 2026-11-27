#!/usr/bin/env bash
# Batch 8 (2026-11-26): per-image input normalisation (--input-norm). On Fashion-MNIST (batch 7) the best two-layer network
# lost 6 points to the single layer (63.5 / 64.2 vs 69.4 / 69.6 %), mostly on sandals (recall 22 vs ~70 %): light images make
# few layer-1 spikes (sandals ~500 per image vs 1,782 on average), so layer 2, calibrated to the average image, forms almost no
# sandal neurons. --input-norm scales every image to the training set's mean intensity. Both networks, unchanged otherwise, on
# Fashion-MNIST (two seeds each) and on MNIST (one seed each: does it cost anything there?). Same regime as batches 2-7. Runs go
# to new_architecture_testing/snn_multi_layered_unshared/results_2/09_batch8_input_norm_6ep_20261126/ (its README.md lists the
# questions).
#
#   bash unity/submit_multi_layer.sh smoke              # ON YOUR OWN COMPUTER, no queue: each job's exact configuration on
#                                                       # tiny settings, one after another -> pass / fail per job
#   bash unity/submit_multi_layer.sh smoke ml8_fm_best_in   # ... only the jobs named
#   bash unity/submit_multi_layer.sh                    # the six jobs on Unity
#   bash unity/submit_multi_layer.sh ml8_fm_best_in_s1  # only the jobs named (e.g. to resubmit failed ones)
#
# Fashion-MNIST is downloaded to <repo>/data/FashionMNIST on first use. Unity compute nodes may have no internet, so in Unity
# mode this script downloads it first, here on the login node (with the snn environment active), before submitting.
# Local smoke runs use the python on your PATH (activate the project's environment first) or PYTHON=/path/to/python.
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
SDIR=$(dirname "$S")
OUT=results_2/09_batch8_input_norm_6ep_20261126

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

COMMON="--theta-tau 30000 --epochs 6 --snapshot-interval 200 --input-norm --results-dir $OUT"
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
job ml8_fm_single_in     05:00:00 $COMMON --dataset fmnist $SINGLE --seed 0    # Fashion-MNIST, normalised
job ml8_fm_single_in_s1  05:00:00 $COMMON --dataset fmnist $SINGLE --seed 1
job ml8_fm_best_in       05:00:00 $COMMON --dataset fmnist $BEST --seed 0
job ml8_fm_best_in_s1    05:00:00 $COMMON --dataset fmnist $BEST --seed 1
job ml8_mn_single_in     05:00:00 $COMMON --dataset mnist $SINGLE --seed 0     # MNIST, normalised: does it cost anything?
job ml8_mn_best_in       05:00:00 $COMMON --dataset mnist $BEST --seed 0
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
#       --results new_architecture_testing/snn_multi_layered_unshared/results_2
