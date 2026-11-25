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

## Results (Unity, 2026-11-25; ~3 h per run)

"Epochs 4–6" = mean of all snapshots in epochs 4–6 (held-out / shifted ±2 px). "Layer 1 / layer 2 alone" = that layer
labelled and scored as the output at its freeze. Replay columns: the trained layers 1 + 2 simulated in numpy on 120 test
digits (`layer1_analysis/l1lib.py` + a layer-2 step) — how many layer-2 channels (block × feature) fire per image, and how
blurred the image's own layer-2 code looks when drawn back to pixels (σ of the Gaussian blur of the digit it matches best).

| run | final | epochs 4–6 | gap | layer 1 alone | layer 2 alone | layer-2 channels / img | layer-2 spikes / img | code blur σ | layer-3 input (spikes/ms) |
|---|---|---|---|---|---|---|---|---|---|
| default, seed 0 | 77.1 | 75.7 / 52.2 | 23.5 | 67.2 / 49.3 | 63.2 / 45.1 | ~20 | 189 | 2 px | 0.60 |
| default, seed 1 | 74.3 | 75.2 / 51.6 | 23.6 | 68.3 / 49.1 | 63.0 / 44.7 | – | – | – | 0.58 |
| layer-2 readout 8 mV | 80.7 | **80.3** / 53.3 | 27.0 | 67.2 / 49.3 | 63.2 / 45.1 | ~76 | 309 | 2 px | 0.99 |
| blocks at stride 1 (81 blocks) | 79.2 | 79.4 / **53.9** | 25.5 | 67.2 / 49.3 | 66.9 / 47.1 | ~69 | 618 | 2 px | 1.93 |
| 5×5 blocks (16 blocks) | 73.5 | 73.4 / 48.7 | 24.7 | 67.2 / 49.3 | 67.4 / 46.9 | ~15 | 125 | 3 px | 0.39 |
| layer-2 lr ×5 | 77.9 | 77.5 / 53.6 | 23.9 | 67.2 / 49.3 | 65.2 / 44.5 | – | – | – | 0.65 |
| *two layers, stride 2 + 8 mV (seeds 0 / 1)* | | *85.3 / 56.5 · 87.3 / 57.4* | *~29* | *67.2 · 68.3* | | *343 (layer-1 channels)* | *~1,065* | *1.5 px (layer-1 code)* | *3.3* |
| *single layer* | | *85.6 / 60.6* | *25.0* | | | | | | |

- **Sanity check passed:** layer 1 alone matches the two-layer runs exactly (67.2 / 49.3 seed 0, 68.3 / 49.1 seed 1), so
  every difference comes from layers 2 and 3.
- **The third layer costs 5–12 points** (73–80 % vs 85–87 %), and shifted digits did not improve (48.7–53.9 %, below
  the two-layer 56.5–57.4); the held-out − shifted gap is unchanged (~24–27).
- **Layer 3 is starved of channels.** Each layer-2 block reports one of 16 features (200 mV), so layer 3 hears ~20
  channels per image instead of the two-layer top layer's 343. Accuracy follows the channel count: ~15 → 73.4, ~20 →
  75.7, ~69 → 79.4, ~76 → 80.3 — the same pattern as the two-layer batches 5–6 (accuracy tracked layer-1 spikes from more
  channels). The two changes that add channels (8 mV readout, stride 1) are the two best runs.
- **Layer 2 alone (63–67 %) is no better than layer 1 alone (67 %)**: it compresses 144 input channels into one symbol per
  block and loses information on the way.
- **Why the pictures are blurry (F3, F4):**
  1. the drawing itself: a single digit's *layer-1* code drawn back already looks like the digit blurred by σ 1–1.5 px
     (each count is drawn as a whole, slightly fuzzy 7×7 template, and neighbouring windows overlap by 5 px); F4 draws
     through two such layers;
  2. drawing through two layers adds up: one digit's *layer-2* code drawn back is blurred by σ 2 px (3 px with 5×5
     blocks). My first reading — that layer-2 features are coarse blobs — was wrong: see F6 below;
  3. a template is an average over every instance its neuron wins (as in any D&C layer).
  **Added after the runs (`sta_replay.py`, F6/F7):** the spike-triggered averages — the average pixel patch each layer-2
  neuron fired for on the 1,000 test digits — are sharp, specific stroke combinations (arcs, corners, bars, loops,
  junctions), and layer-3 neurons fire for clean digit shapes. The replay reproduces each run's accuracy within 1.5
  points with the run's own labels. So the blur in F3/F4 is mostly the drawing method; the features are good, and the
  accuracy gap is the channel count, not the features.
  The layer-2 templates did not collapse onto one another: they use ~56 % of their inputs (random start 78 %), and
  features of the same block are unrelated (mean-centred correlation 0.07). They are distinct.
