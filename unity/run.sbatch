#!/bin/bash
# Generic Unity job: runs ONE of this project's python scripts on one GPU. Submit from the REPO ROOT:
#   sbatch unity/run.sbatch new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py --l1-wta --l2-wta --seed 0
#   sbatch -p gpu-preempt -t 01:00:00 -J stdp unity/run.sbatch new_architecture_testing/STDP_testing/stdp_demo.py
# Options on the sbatch command line override the #SBATCH lines below. Logs: unity/logs/<jobname>-<jobid>.out
#SBATCH -p gpu                 # non-preemptable, up to 48 h. -p gpu-preempt: jobs under 2 h, shorter queue, can be killed
#SBATCH --gpus=1               # any GPU; this workload needs < 2 GB of VRAM, so no --constraint (shortest queue)
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o unity/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
module purge
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate snn
cd "$SLURM_SUBMIT_DIR"                          # the repo root (submit from there)
script="$1"; shift
cd "$(dirname "$script")"                       # every script resolves its repo root and results/ from its own location
echo "node $(hostname) | gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) | start $(date)"
python -u "$(basename "$script")" "$@"
echo "end $(date)"
