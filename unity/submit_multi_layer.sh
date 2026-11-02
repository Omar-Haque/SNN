#!/usr/bin/env bash
# Submits every variant of snn_multi_layered_unshared/run_multi_layer_unshared.sh as its own job, so they run in parallel
# instead of one after another. Run from anywhere:  bash unity/submit_multi_layer.sh
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
S=new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py
sbatch -J ml_wta_s0      unity/run.sbatch $S --l1-wta --l2-wta --seed 0
sbatch -J ml_wta_s1      unity/run.sbatch $S --l1-wta --l2-wta --seed 1
sbatch -J ml_ref_wta     unity/run.sbatch $S --no-l1 --l2-wta --seed 0
sbatch -J ml_l2wta_only  unity/run.sbatch $S --l2-wta --seed 0
sbatch -J ml_wta_stride7 unity/run.sbatch $S --l1-wta --l2-wta --stride 7 --seed 0
sbatch -J ml_wta_pool2   unity/run.sbatch $S --l1-wta --l2-wta --pool2 --seed 0
sbatch -J ml_wta_f32     unity/run.sbatch $S --l1-wta --l2-wta --features 32 --seed 0
squeue --me
# afterwards, on the login node:  conda activate snn && python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py
