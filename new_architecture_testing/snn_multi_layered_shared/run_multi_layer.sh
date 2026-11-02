#!/usr/bin/env bash
# Two-layer STDP network. Stage A (layer 1) is a few minutes; stage B is 2 epochs of layer-2 training (~45-60 min on
# MPS at roughly 2x the single-layer cost). Run from this folder:  bash run_multi_layer.sh
set -e
cd "$(dirname "$0")"

# 0. the single-layer reference, evaluated with the same code and the same shifted test set
python multi_layer.py --no-l1 --seed 0

# 1. the default architecture: shared 7x7 templates, stride 3, 16 features, sequential training
python multi_layer.py --seed 0
python multi_layer.py --seed 1

# 2. the comparisons, one knob each
python multi_layer.py --pool2 --seed 0                  # 2x2 spike-OR pooling between the layers (256 layer-2 inputs)
python multi_layer.py --unshared --seed 0               # locally connected: every layer-1 neuron its own template
python multi_layer.py --features 32 --seed 0            # more features
python multi_layer.py --patch 5 --stride 2 --seed 0     # smaller features, 12x12 positions (2,304 layer-2 inputs)

python summarize_multi_layer.py

# later, after looking at F15 / F10-style latencies: --l1-norm, --l1-theta-plus, --l1-theta-tau, --l2-norm, --l2-wmax
