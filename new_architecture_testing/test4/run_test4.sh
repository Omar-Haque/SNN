#!/usr/bin/env bash
# test4: "use it or lose it". Uniform init, STDP only, no gate. ~30 min per run on MPS (2 epochs per phase).
# Run from this folder:  bash run_test4.sh      (summarize_test4.py can be run at any time on whatever has finished)
set -e
cd "$(dirname "$0")"

# ---- A. i.i.d. stream: recycling should be dormant (conscience keeps every neuron in use) ----
python test4.py --phases iid --phase-epochs 2 --seed 0                               # reference
python test4.py --phases iid --phase-epochs 2 --pool 50 --seed 0                     # recycling available: does it ever trigger?

# ---- B. class-incremental 0-4 -> 5-9 ----
python test4.py --seed 0                                                             # no recycling (test3's V1, without imprint)
python test4.py --pool 50  --seed 0                                                  # pool 50, idle >= 10k
python test4.py --pool 100 --seed 0                                                  # pool 100
python test4.py --pool 50  --recycle-idle 20000 --seed 0                             # stricter idle criterion
python test4.py --pool 50  --theta-tau 30000 --seed 0                                # with the forgetting clock

python summarize_test4.py

# ---- C. second seeds of the informative ones ----
python test4.py --seed 1
python test4.py --pool 50 --seed 1
python test4.py --pool 100 --seed 1

python summarize_test4.py

# ---- later ----
# python test4.py --neurons 100 --seed 0 ; python test4.py --neurons 100 --pool 20 --seed 0     # remove redundancy so forgetting shows in accuracy
# python test4.py --phases "0-1,2-3,4-5,6-7,8-9" --pool 50 --seed 0                               # five arrivals
