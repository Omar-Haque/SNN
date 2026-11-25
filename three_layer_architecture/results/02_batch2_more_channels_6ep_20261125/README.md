# 02 — batch 2: more channels for layer 3 (submitted from `unity/submit_three_layer.sh`, 2026-11-25)

## Purpose
Batch 1 (`../01_batch1_first_three_layer_6ep_20261125/`) put the three-layer network at 73–80 % against the two-layer
network's 85–87 %, and found:
- **Layer 2 learns good features.** The spike-triggered averages (F6: the average pixel patch each layer-2 neuron fired for)
  are sharp, specific stroke combinations — arcs, corners, bars, loops, junctions. The blobs in F3 are the drawing method
  (weights drawn back through overlapping, slightly fuzzy layer-1 templates), not the features.
- **Layer 3 is starved of channels.** Each block reports one feature at 200 mV, so layer 3 hears ~20 channels per image
  where the two-layer top layer heard 343, and accuracy followed the channel count: ~15 → 73.4 %, ~20 → 75.7, ~69 (blocks
  at stride 1) → 79.4, ~76 (layer-2 readout at 8 mV) → 80.3.

So this batch keeps the features and gives layer 3 more of them: softer readout, more overlap, more features per block,
smaller blocks, a skip connection from layer 1, combinations; and one change aimed at the features themselves (training
layer 2 on the sharper one-feature-per-window layer-1 code).

## References (mean of the snapshots of epochs 4–6, held-out / shifted ±2 px)
| network | seed | accuracy |
|---|---|---|
| single layer | 0 | 85.6 / 60.6 |
| two layers, stride 2 + 8 mV | 0 / 1 | 85.3 / 56.5 · 87.3 / 57.4 |
| three layers, batch 1 default | 0 / 1 | 75.7 / 52.2 · 75.2 / 51.6 |
| three layers, batch 1, layer-2 readout 8 mV | 0 | 80.3 / 53.3 |
| three layers, batch 1, blocks at stride 1 | 0 | 79.4 / 53.9 |

## Configuration
Every job: `three_layer.py`'s defaults (layer 1: stride 2, 8 mV readout; layer 2: 3×3 blocks at stride 2, 16 features,
200 mV; layer 3: 400 neurons, drive 0.36, lr ×10, 6 epochs) plus the flags below. Seed 0 unless noted. 10-h limit (batch 1
took ~3 h; the largest layer 2 here has 3,200 neurons).

| job | flags | layer 2 | layer-3 inputs | question |
|---|---|---|---|---|
| `tl2_control` | — | 25 blocks × 16 = 400 | 400 | Does the new code reproduce batch 1's seed-0 run (75.7)? |
| **A. more channels** | | | | |
| `tl2_b3s1_rd8` | `--l2-stride 1 --l2-readout-inhibition 8` | 81 × 16 = 1,296, 8 mV | 1,296 | Do batch 1's two best changes add up? |
| `tl2_b3s1_rd8_s1` | same, `--seed 1` | | | … on a second seed |
| `tl2_rd4` | `--l2-readout-inhibition 4` | 400, 4 mV | 400 | Softer still? |
| `tl2_b3s1_rd4` | `--l2-stride 1 --l2-readout-inhibition 4` | 1,296, 4 mV | 1,296 | … with overlap |
| **B. finer vocabulary** | | | | |
| `tl2_f32` | `--l2-features 32` | 25 × 32 = 800 | 800 | More features per block: finer stroke combinations, more channels |
| `tl2_f64` | `--l2-features 64` | 25 × 64 = 1,600 | 1,600 | … more still |
| `tl2_f32_tp0.3` | `--l2-features 32 --l2-theta-plus 0.3` | 800, stronger conscience | 800 | Keep the extra features in use (layer 1 at 32 features left 41 % dead) |
| `tl2_f32_rd8` | `--l2-features 32 --l2-readout-inhibition 8` | 800, 8 mV | 800 | finer vocabulary + soft readout |
| **C. smaller blocks** | | | | |
| `tl2_b2s1` | `--l2-block 2 --l2-stride 1` | 2×2 windows (9×9 px): 100 × 16 = 1,600 | 1,600 | Smaller parts, more blocks |
| `tl2_b2s1_rd8` | same + `--l2-readout-inhibition 8` | 1,600, 8 mV | 1,600 | |
| **D. skip connection** | | | | |
| `tl2_skip` | `--l3-skip` | 400 | 1,936 + 400 = 2,336 | Layer 3 reads layer 1 too: does it recover the two-layer accuracy, and do the layer-2 parts add anything on top? |
| `tl2_skip_s1` | same, `--seed 1` | | | … second seed (compare with the two-layer seed 1: 87.3) |
| `tl2_skip_rd8` | `--l3-skip --l2-readout-inhibition 8` | 400, 8 mV | 2,336 | |
| `tl2_skip_b3s1_rd8` | `--l3-skip --l2-stride 1 --l2-readout-inhibition 8` | 1,296, 8 mV | 3,232 | |
| **E. what layer 2 learns from** | | | | |
| `tl2_a2inh200` | `--l1-a2-inhibition 200` | 400 | 400 | Layer 2 learns on the one-feature-per-window layer-1 code (sharper: a digit's code drawn back is blurred σ 1 px vs 1.5 at 8 mV); layer 1 is read at 8 mV in stage B |
| `tl2_a2inh200_rd8` | same + `--l2-readout-inhibition 8` | 400, 8 mV | 400 | |
| **F. combinations** | | | | |
| `tl2_b3s1_f32_rd8` | `--l2-stride 1 --l2-features 32 --l2-readout-inhibition 8` | 81 × 32 = 2,592, 8 mV | 2,592 | many channels and a finer vocabulary |
| `tl2_b2s1_f32_rd8` | `--l2-block 2 --l2-stride 1 --l2-features 32 --l2-readout-inhibition 8` | 100 × 32 = 3,200, 8 mV | 3,200 | |

Notes:
- With `--l1-a2-inhibition 200`, layer 2 is calibrated and trained on layer 1 at 200 mV (~657 layer-1 spikes per image), then
  in stage B layer 1 is read at 8 mV (~1,065): layer 2 receives more input than it learned on. That mismatch is part of the
  test.
- With `--l3-skip`, layer 3's calibration counts both pathways (layer 1 sends ~6× more spikes than layer 2), so at the start
  layer 3's weight is spread over all lines evenly; `run.json` → `l3_weight_share_l2` reports how much of it ends on
  layer-2 lines (start: 400 / 2,336 = 17 %).

## What to watch
- Layer-3 accuracy against the channel count it receives (`l3_input_rate_per_ms` in `run.json`, F6 counts) — does the
  batch-1 relation keep holding, and where does it level off?
- F6 (what layer-2 neurons fire for) and F3 for the 32 / 64-feature and 2×2-block runs: finer, distinct parts, or many
  neurons that never fire (`l2_never_fired_on_test` in `run.json`, black panels in F6)?
- Skip runs: accuracy vs the two-layer network and `l3_weight_share_l2` — near 17 % means layer 3 ignored layer 2.
- Shifted accuracy and the held-out − shifted gap (~24–27 in batch 1).

## Run and summarise
```bash
bash unity/submit_three_layer.sh smoke     # first, on your own computer: all 19 configurations on tiny settings
bash unity/submit_three_layer.sh           # the 19 jobs on Unity
python three_layer_architecture/summarize_three_layer.py --results three_layer_architecture/results
```

## Results
(to be filled in when the runs finish)
