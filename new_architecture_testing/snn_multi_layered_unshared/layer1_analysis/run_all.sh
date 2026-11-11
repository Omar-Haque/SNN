#!/usr/bin/env bash
# Reproduces every number and figure in README.md (~1.5 h on 2 CPU cores). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p cache figs
R=unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133
python3 validate.py $R                                   # numpy replay vs the run's own accuracy
python3 census.py && python3 ablate.py && python3 info.py && python3 mechanism.py && python3 needed.py
python3 stability.py ../results/$R                       # Jaccard version of the stability numbers
# stage A, numpy replica and interventions
python3 train_l1.py --examples 120000 --frames 0,320,1280,3200,9600,30000,60000,120000 --out cache/l1_base120k.npz
python3 train_l1.py --theta-plus 0.3 --out cache/l1_tp03.npz
python3 train_l1.py --features 8 --out cache/l1_f8.npz
python3 train_l1.py --features 4 --out cache/l1_f4.npz
python3 train_l1.py --shift-aug 2 --out cache/l1_aug2.npz
python3 extract.py cache/l1_base120k.npz cache/eval_base30k.npz 30000; python3 extract.py cache/l1_base120k.npz cache/eval_base120k.npz
for v in tp03 f8 f4 aug2; do python3 extract.py cache/l1_$v.npz cache/eval_$v.npz; done
# the same harness on every layer 1
python3 code_eval.py pixels --tag pixels
python3 code_eval.py run:../results/$R --tag torch_f16
python3 code_eval.py run:../results/$R --inh 17.5 --tag torch_f16_read17.5
python3 code_eval.py run:../results/unshared_p7s3f32_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-154659 --tag torch_f32
for v in base30k base120k tp03 f8 f4 aug2; do python3 code_eval.py npz:cache/eval_$v.npz --tag np_$v; done
python3 figures_layer1.py
