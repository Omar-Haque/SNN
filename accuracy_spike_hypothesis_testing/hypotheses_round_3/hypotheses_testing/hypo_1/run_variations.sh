#!/usr/bin/env bash
# Full design for hypothesis 1, run sequentially (~35-40 min each on MPS). Trim as needed.
# Run from this folder:  bash run_variations.sh
set -e
cd "$(dirname "$0")"

python hypo_1.py --variation baseline    --seed 0
python hypo_1.py --variation sabotage      --seed 0                 # sharpest variation: our predictions differ here
python hypo_1.py --variation refill      --n-refill 50  --seed 0
python hypo_1.py --variation pseudo_refill --n-refill 50  --seed 0
python hypo_1.py --variation imprint     --seed 0
python hypo_1.py --variation refill      --n-refill 25  --seed 0
python hypo_1.py --variation refill      --n-refill 100 --seed 0
python hypo_1.py --variation baseline    --seed 1
python hypo_1.py --variation sabotage      --seed 1

python summarize_hypo_1.py
