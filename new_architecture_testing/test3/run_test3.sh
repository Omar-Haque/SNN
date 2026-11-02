#!/usr/bin/env bash
# test3: the reservoir as a plasticity reserve. Stream = digits 0-4 for 2 epochs, then 5-9 for 2 epochs (~122k examples,
# ~30 min per run on MPS). First pass = one seed of everything (~3.5 h) + summary; second pass = second seeds (~2 h).
# Run from this folder:  bash run_test3.sh      (summarize_test3.py can be run at any time on whatever has finished)
set -e
cd "$(dirname "$0")"

# ---- first pass, seed 0 ----
python test3.py --init imprint --reserve 0   --seed 0                          # V1  no reserve
python test3.py --init imprint --reserve 100 --seed 0                          # V2  fixed reserve, K = 100
python test3.py --init imprint --reserve 200 --seed 0                          # V2  K = 200 (does new-class accuracy scale with K?)
python test3.py --init uniform --theta-plus 0.05 --seed 0                      # baseline: the original network, reserve by accident
python test3.py --init imprint --reserve 0   --theta-tau 30000 --seed 0        # V1 with theta decay (decay = a forgetting clock)
python test3.py --init imprint --reserve 100 --theta-tau 30000 --seed 0        # V2 with theta decay
python test3.py --init imprint --reserve 100 --replenish --seed 0              # V3  replenished reserve

python summarize_test3.py

# ---- second pass, seed 1 ----
python test3.py --init imprint --reserve 0   --seed 1
python test3.py --init imprint --reserve 100 --seed 1
python test3.py --init imprint --reserve 100 --theta-tau 30000 --seed 1
python test3.py --init imprint --reserve 100 --replenish --seed 1

python summarize_test3.py

# ---- later: five arrivals (V3 is the one that has to survive this; V1 for contrast) ----
# python test3.py --phases "0-1,2-3,4-5,6-7,8-9" --phase-epochs 2 --init imprint --reserve 100 --replenish --seed 0
# python test3.py --phases "0-1,2-3,4-5,6-7,8-9" --phase-epochs 2 --init imprint --reserve 0 --seed 0
