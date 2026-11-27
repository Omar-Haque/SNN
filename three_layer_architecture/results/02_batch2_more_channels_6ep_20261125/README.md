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
git show 3b1887b:unity/submit_three_layer.sh   # the script now holds batch 3; this is this batch's version
python three_layer_architecture/summarize_three_layer.py --results three_layer_architecture/results
```

## Results (Unity, 2026-11-25/26)

Mean of the snapshots of epochs 4–6 (held-out / shifted ±2 px); "layer-3 input" = layer 2's calibrated output rate (spikes
per ms per presentation, both pathways with `--l3-skip`); "dead" = layer-2 neurons that never fired on the 1,000 test digits.
`tl2_b3s1_f32_rd8` hit the 10-h limit in epoch 6 (313,632 of 360,000 images; its numbers come from the log). It and
`tl2_a2inh200_rd8` (6.6 h) both ran on `gypsum-gpu018`; every other job took 2.7–2.9 h, so that node was slow, not the code.

| job | epochs 4–6 | gap | best | layer 2 alone | layer-3 input /ms | layer-3 lines | dead layer 2 |
|---|---|---|---|---|---|---|---|
| *references:* single layer · two layers s0 / s1 | *85.6 / 60.6 · 85.3 / 56.5 · 87.3 / 57.4* | | | | | | |
| `tl2_control` (= batch-1 default) | 75.7 / 52.2 | 23.4 | 78.3 | 63.2 | 0.60 | 400 | 46 / 400 |
| **A** `tl2_b3s1_rd8` / `_s1` | 82.7 / 54.0 · 82.1 / 52.7 | 28.7 · 29.4 | 84.7 · 84.9 | 66.9 · 64.6 | 3.15 · 3.12 | 1,296 | 65 · 64 |
| `tl2_rd4` | 81.9 / 55.5 | 26.3 | 84.4 | 63.2 | 1.45 | 400 | 9 |
| `tl2_b3s1_rd4` | 83.2 / 54.6 | 28.6 | 85.2 | 66.9 | 4.62 | 1,296 | 18 |
| **B** `tl2_f32` | 76.5 / 52.3 | 24.2 | 79.1 | 66.7 | 0.64 | 800 | 201 / 800 |
| `tl2_f64` | 77.1 / 52.0 | 25.0 | 79.9 | 65.9 | 0.65 | 1,600 | 739 / 1,600 |
| `tl2_f32_tp0.3` | 73.7 / 48.1 | 25.5 | 77.7 | 69.5 | 0.46 | 800 | 48 / 800 |
| `tl2_f32_rd8` | 79.9 / 53.7 | 26.3 | 82.2 | 66.7 | 1.06 | 800 | 143 / 800 |
| **C** `tl2_b2s1` | 81.0 / 55.4 | 25.6 | 83.6 | 66.0 | 2.17 | 1,600 | 213 / 1,600 |
| `tl2_b2s1_rd8` | 83.0 / 55.6 | 27.4 | 85.6 | 66.0 | 3.73 | 1,600 | 142 / 1,600 |
| **D** `tl2_skip` / `_s1` | **85.1 / 56.1 · 84.7 / 56.0** | 29.1 · 28.6 | 87.1 · 87.3 | 63.2 · 63.0 | 3.96 · 3.85 | 2,336 | 46 · 51 |
| `tl2_skip_rd8` | 84.0 / 54.9 | 29.2 | 85.9 | 63.2 | 4.35 | 2,336 | 36 |
| `tl2_skip_b3s1_rd8` | 83.8 / 54.7 | 29.1 | 85.9 | 66.9 | 6.51 | 3,232 | 65 |
| **E** `tl2_a2inh200` | 77.4 / 54.4 | 23.0 | 79.7 | 64.5 | 0.92 | 400 | 19 |
| `tl2_a2inh200_rd8` | **84.2 / 56.1** | 28.1 | 85.8 | 64.9 | 1.65 | 400 | 13 |
| **F** `tl2_b3s1_f32_rd8` (timed out) | 81.8 / 52.4 | 29.5 | 84.9 | – | 3.39 | 2,592 | – |
| `tl2_b2s1_f32_rd8` | 83.3 / 55.8 | 27.5 | 85.7 | 67.4 | 4.04 | 3,200 | 555 / 3,200 |

- **The control reproduces batch 1 exactly** (75.7 / 52.2, same snapshots), so the new code changed nothing it shouldn't.
- **More channels helps, up to ~83 %.** Overlap plus a soft readout (A, C) takes the network from 75.7 to 82–83 %, and the
  layer-3 input rate tracks it — but past ~3 spikes/ms it flattens (3.1 → 82.1–82.7, 3.7 → 83.0, 4.0 → 83.3, 4.6 → 83.2).
- **What layer 2 learns from matters most.** Layer 2 trained on layer 1's one-feature-per-window code (200 mV) and read out
  at 8 mV reaches **84.2 %** with only 400 layer-2 neurons and 1.65 spikes/ms into layer 3 — above every A/C/F run, which
  need 2–3× the channels. Read out at 200 mV it gives 77.4 (control 75.7). Its features were learned on a sparser, sharper
  input, so each layer-2 spike carries more.
- **More features per block does not help.** The extra features mostly never fire (201 of 800, 739 of 1,600 dead), and a
  block still reports one winner at 200 mV, so layer 3 gets no more input (0.64–0.65 spikes/ms). A stronger conscience keeps
  them alive (48 dead) but lowers layer 2's output further (0.46) and accuracy with it (73.7) — though it gives the best
  layer-2-alone score (69.5).
- **The skip connection gets the two-layer accuracy, not more — and layer 3 largely ignores layer 2.** 85.1 / 84.7 % vs the
  two-layer 85.3 / 87.3 on the same seeds. The share of layer 3's weight on layer-2 lines ends at 13 % (it started at 17 %);
  with the 8 mV layer 2 it ends at 21 % (start 17 %), with the stride-1 layer 2 at 46 % (start 40 %). Layer 3 moves weight
  toward layer 1, not toward the parts.
  *Batch-3 note: this reading was wrong. In every skip run the layer-2 weight share tracks layer 2's share of layer 3's
  input spikes (here 15 % of the spikes, 13 % of the weight; 23 → 21 %; 48 → 46 %), because STDP with row normalisation
  gives weight roughly in proportion to how often each line fires. The share cannot show whether layer 3 uses layer 2; an
  ablation can. See `../03_batch3_sparse_trained_layer2_6ep_20261126/`.*
- **No shift gain anywhere:** shifted accuracy 48–56 % (two layers 56.5–57.4, single layer 60.6).
- **Best three-layer runs:** 85.1 / 84.7 % with the skip connection (≈ two layers), 84.2 % without it (layer 2 trained at
  200 mV, read at 8 mV; one seed). Still under the two-layer mean (86.3) and the single layer (85.6) on centred digits.
