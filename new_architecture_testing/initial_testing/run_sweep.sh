#!/usr/bin/env bash
# Planned sweep, run sequentially. ~13 min per epoch on MPS: 10 one-epoch runs + 2 three-epoch baselines + 2 imprint
# runs ~ 3.5 h. Run from this folder:  bash run_sweep.sh      (trim as needed; summarize_sweep.py picks up whatever finished)
set -e
cd "$(dirname "$0")"

# 1. the baseline, with its plateau and spike (3 epochs, so the ceiling is the one we know: ~85%)
python sweep.py --theta-plus 0.05 --epochs 3 --seed 0
python sweep.py --theta-plus 0.05 --epochs 3 --seed 1

# 2. imprint at zero: no reservoir, nothing else changed -> the reference ceiling for a no-plateau network
python sweep.py --init imprint --seed 0
python sweep.py --init imprint --seed 1

# 3. the theta_plus sweep (1 epoch each: a fast-convergence variant must have converged by 60k)
for tp in 0.1 0.2 0.5 1.0; do
  for seed in 0 1; do
    python sweep.py --theta-plus $tp --seed $seed
  done
done

python summarize_sweep.py

# 4. (after looking at S2-S4) widen around the winner, e.g.:
# python sweep.py --theta-plus 0.2 --inhibition 120 --seed 0
# python sweep.py --theta-plus 0.05 --theta-init -48 --seed 0
