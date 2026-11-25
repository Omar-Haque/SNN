# 01 — batch 1: first three-layer runs (submitted from `unity/submit_three_layer.sh`, 2026-11-25)

## Purpose
The first runs of `../../three_layer.py`: layer 1 = the two-layer study's best (stride 2, 8 mV readout), layer 2 = the new
local mid-level layer, layer 3 = the D&C classifier (drive 0.36, lr ×10). Two questions:
1. Does the third layer keep the two-layer network's accuracy on centred digits (85–87 %)?
2. Do layer-2 parts make the network more tolerant of ±2 px shifts (stuck at 56–59 % with two layers, 60.6 % single layer)?

Each job changes one layer-2 choice from the default, so the batch also gives a first map of what matters in layer 2.

## References (not rerun; mean of the snapshots of epochs 4–6, held-out / shifted)
| network | seed | accuracy | where |
|---|---|---|---|
| single layer | 0 | 85.6 / 60.6 | `../../../new_architecture_testing/snn_multi_layered_unshared/results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |
| two layers, stride 2 + 8 mV | 0 | 85.3 / 56.5 | `../../../new_architecture_testing/snn_multi_layered_unshared/results_2/07_batch6_stride2_readout8_6ep_20261122/…seed0_l1wta_rd8…` |
| two layers, stride 2 + 8 mV | 1 | 87.3 / 57.4 | same folder, `…seed1_l1wta_rd8…` |

Stage A is the two-layer study's code with the same seed and image order, so each three-layer run should train exactly the
layer 1 of the two-layer run with its seed. Check: "layer 1 alone" at the freeze should read 67.2 / 49.3 % for seed 0 and
68.3 / 49.1 % for seed 1, as in those runs; if it does, any difference further up comes from layers 2 and 3. Two seeds of
the same configuration differed by 2 points in the two-layer study; read single differences of 1–2 points as noise.

## Configuration
Every job uses `three_layer.py`'s defaults (see `../../README.md`) plus the flags below: 6 epochs for layer 3, snapshots every
200 batches, stages A and A2 up to 30,000 images each.

| job | flags | layer 2 | question |
|---|---|---|---|
| `tl_default` | `--seed 0` | 3×3 blocks, stride 2: 25 blocks × 16 = 400 neurons, 144 inputs | the default three-layer network |
| `tl_default_s1` | `--seed 1` | same | second seed |
| `tl_l2rd8` | `--l2-readout-inhibition 8` | same, read out at 8 mV | Does layer 1's best change (let the runners-up fire) also help one layer up? |
| `tl_b3s1` | `--l2-stride 1` | 3×3 blocks, stride 1: 81 blocks × 16 = 1,296 neurons | More overlap: stride 2 was the other layer-1 win |
| `tl_b5s2` | `--l2-block 5` | 5×5 blocks (15×15 px), stride 2: 16 blocks × 16 = 256 neurons | Bigger parts |
| `tl_l2lr5` | `--l2-lr-mult 5` | default, STDP rates ×5 | Layer 2's inputs are sparser than pixels, so it may learn slower than layer 1 did; does that matter? |

Time limit 8 h each (the two-layer stride-2 runs took ~2 h; this adds a stage and a layer).

## What to watch
- **Stage A2 console / F5:** layer-2 templates still at their initial weights (should fall well below 400 — blocks that
  never get input stay), blocks with input that fired (%), first-spike latency, features per active block.
- **Layer 2 alone** (orange in F1; `run.json` → `l2.readout_accuracy_at_freeze`): how much of the digit layer 2 already
  carries, vs layer 1 alone (~67 %).
- **F3:** do layer-2 features look like stroke combinations (corners, junctions, curves), or like blurred blobs?
- **Layer 3** (blue in F1): accuracy against the references, and the held-out − shifted gap (~29 points for the two-layer
  stride 2 + 8 mV runs; 25 single layer).
- `[calibration L2]` and `[calibration L3]` lines: the input rates and weight scales chosen.

## Run and summarise
```bash
bash unity/submit_three_layer.sh smoke     # first, on your own computer: all six configurations on tiny settings
bash unity/submit_three_layer.sh           # then the six jobs on Unity
python three_layer_architecture/summarize_three_layer.py --results three_layer_architecture/results
```

## Results
(to be filled in when the runs finish)
