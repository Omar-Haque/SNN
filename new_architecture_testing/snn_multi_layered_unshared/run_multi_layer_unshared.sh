#!/usr/bin/env bash
# Locally connected two-layer STDP network. Stage A (layer 1) up to 30k images (~7 min), stage B 2 epochs (~50-60 min).
# Run from this folder:  bash run_multi_layer_unshared.sh
#
# The first run (2026-10-31, no tie-break anywhere) reached 69.7% but both layers ran in "volley" mode: the 16 features
# of a window see the identical input and inhibition arrives one step late, so once their templates are similar they
# cross threshold in the same millisecond and all fire once; layer 2 then receives a synchronous burst and 6-59 of its
# neurons fire once each per presentation. Nobody takes a presentation, templates average everything (top-k share
# 0.23-0.30), and every theta climbs together. The tie-break flags resolve same-step crossings: --l1-wta within a
# window, --l2-wta in layer 2 (BindsNET's DiehlAndCookNodes does the same by default, one_spike=True, picking at random;
# ours keeps the highest potential, like the WTA block of the original code).
set -e
cd "$(dirname "$0")"

python multi_layer_unshared.py --l1-wta --l2-wta --seed 0       # recommended first: tie-break in both layers
python multi_layer_unshared.py --l1-wta --l2-wta --seed 1
python multi_layer_unshared.py --no-l1 --l2-wta --seed 0        # single-layer reference with the same tie-break, same evaluation (incl. shifted test set)
python multi_layer_unshared.py --l2-wta --seed 0                # tie-break in layer 2 only: is layer-1 volleying harmful by itself?
python multi_layer_unshared.py --l1-wta --l2-wta --stride 7 --seed 0     # disjoint windows: 4x4 = 16 non-overlapping 7x7 tiles x 16 = 256 neurons
python multi_layer_unshared.py --l1-wta --l2-wta --pool2 --seed 0        # 2x2 spike-OR pooling between the layers
python multi_layer_unshared.py --l1-wta --l2-wta --features 32 --seed 0  # 32 neurons per window
# python multi_layer_unshared.py --seed 0                       # the original no-tie-break run (results/unshared_..._ep2_seed0_20261031-135637)

python summarize_multi_layer.py
