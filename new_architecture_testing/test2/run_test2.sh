#!/usr/bin/env bash
# test2: make the fast variant (theta_plus 0.5) stationary. All runs are 3 epochs (~40 min each on MPS).
# First pass = one seed of everything (~4 h), then a summary; second pass = second seeds of the informative ones (~2 h).
# Run from this folder:  bash run_test2.sh      (summarize_test2.py can be run at any time on whatever has finished)
set -e
cd "$(dirname "$0")"

# ---- first pass: one seed of everything ----
# (a) no decay: does the collapse happen, and when?
python test2.py --theta-plus 0.5 --seed 0
python test2.py --theta-plus 1.0 --seed 0
# (b) theta decay, tau in training examples (30k is the estimate that puts the equilibrium at the baseline's theta level)
python test2.py --theta-plus 0.5 --theta-tau 15000 --seed 0
python test2.py --theta-plus 0.5 --theta-tau 30000 --seed 0
python test2.py --theta-plus 0.5 --theta-tau 60000 --seed 0
# (c) the clamp is NOT run: its accuracy cost (2-3 points, monotone in the cap) is already known from rounds 1/2.
#     If you ever want the theta-spread picture (S3 bottom panel), one epoch is enough:
#     python test2.py --theta-plus 0.5 --theta-max 0 --epochs 1 --seed 0
# (d) annealing: 0.5 while the reservoir drains, then the baseline 0.05
python test2.py --theta-plus 0.5 --theta-plus-after 15000 --seed 0

python summarize_test2.py

# ---- second pass: second seeds ----
python test2.py --theta-plus 0.5 --seed 1
python test2.py --theta-plus 0.5 --theta-tau 30000 --seed 1
python test2.py --theta-plus 0.5 --theta-plus-after 15000 --seed 1

python summarize_test2.py
