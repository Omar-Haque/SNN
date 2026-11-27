# 08 — batch 7: Fashion-MNIST (submitted from `unity/submit_multi_layer.sh`, 2026-11-25)

## Purpose
Every result so far is on MNIST. This batch runs the best two-layer network and the single-layer reference, unchanged, on
Fashion-MNIST: do the two-layer study's choices (a locally connected layer 1 at stride 2, the soft 8 mV readout, layer 2
calibrated from layer 1's output) carry over to a harder dataset without retuning, and how do the two networks compare there?

Fashion-MNIST has MNIST's format — 28×28 grey images in [0, 1], 10 classes, 60,000 training / 10,000 test images — but the
images are clothing: 0 T-shirt/top, 1 trouser, 2 pullover, 3 dress, 4 coat, 5 sandal, 6 shirt, 7 sneaker, 8 bag, 9 ankle
boot. They carry roughly twice MNIST's ink (mean intensity ~0.29 vs ~0.13) and texture, and several classes look alike
(shirt, T-shirt, pullover, coat).

## References (MNIST; mean of the snapshots of epochs 4–6, held-out / shifted ±2 px)
| network | seed | MNIST | folder |
|---|---|---|---|
| single layer | 0 | 85.6 / 60.6 | `../../results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |
| two layers, stride 2 + 8 mV | 0 / 1 | 85.3 / 56.5 · 87.3 / 57.4 | `../07_batch6_stride2_readout8_6ep_20261122/…_p7s2f16_…_rd8_…` |

## Configuration
Exactly the MNIST settings plus `--dataset fmnist`, two seeds each; 6 epochs, layer-2 θ decay τ = 30k examples, snapshots
every 200 batches, shifted test set ±2 px. 5-h limit (the MNIST runs took 1.2–2.4 h).

| job | flags | network |
|---|---|---|
| `ml7_fm_single` / `ml7_fm_single_s1` | `--no-l1 --l2-wta` | single layer: 400 D&C neurons on the 784 pixels (row sum 78.4, w_max 1.0, θ⁺ 0.5) |
| `ml7_fm_best` / `ml7_fm_best_s1` | `--stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10` | two layers: 121 windows × 16 features, read out at 8 mV; 400 D&C neurons, drive 0.36, lr ×10 |

Nothing is retuned, and the extra ink reaches the two networks differently:
- **Two layers:** layer 2's weight scale is calibrated from layer 1's measured output (`[calibration]` line,
  `l2_input_rate_per_ms` in `run.json`), so layer 2 starts at the same 0.36 mV/ms whatever layer 1 sends. Layer 1's own
  row sum (12) is fixed, so its windows get more drive from the extra ink.
- **Single layer:** no calibration — its row sum 78.4 was set for MNIST. With roughly twice the pixel spikes per ms, its
  neurons start with roughly twice MNIST's drive and decide on fewer input spikes relative to the image (the batch-3 lesson
  for layer 2 says that can cost accuracy). That is part of what this batch measures.

## What to watch
- Held-out accuracy of the two networks against each other (two seeds each — seeds differed by up to 2 points on MNIST).
- Layer 1 alone (teal in F1): how much of Fashion-MNIST layer 1's code carries.
- F7 (class bias): which classes get confused — shirt / T-shirt / pullover / coat are the usual ones.
- F13 (layer-2 templates in pixel space) and the single layer's templates: what the 400 prototypes look like for clothing.
- Shifted accuracy and the held-out − shifted gap (~25 for the single layer, ~29 for the two-layer network on MNIST).

## Run and summarise
```bash
git show edb3cd3:unity/submit_multi_layer.sh   # the script now holds batch 8; this is batch 7's version
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2       # summary.csv, S1, S2 in results_2/
```

## Results (Unity, 2026-11-25)

Mean of the snapshots of epochs 4–6, held-out / shifted ±2 px.

| network | seed | Fashion-MNIST | gap | best | MNIST (same seed) |
|---|---|---|---|---|---|
| single layer | 0 / 1 | **69.4 / 53.2 · 69.6 / 53.4** | 16.3 · 16.2 | 72.9 · 71.8 | 85.6 / 60.6 (seed 0) |
| two layers, stride 2 + 8 mV | 0 / 1 | 63.5 / 45.8 · 64.2 / 45.9 | 17.7 · 18.3 | 65.5 · 66.1 | 85.3 / 56.5 · 87.3 / 57.4 |

Layer 1 alone: 57.5 / 57.1 % (MNIST 67.2 / 68.3). Layer-1 output into layer 2: 4.97 / 5.02 spikes/ms (MNIST 3.3).
Both networks are flat from epoch 2–3 on.

Per-class recall (last 6 snapshots) and labels:

| class | ink | single layer: recall · neurons | two layers: recall · neurons | layer-1 spikes per image |
|---|---|---|---|---|
| T-shirt | 0.34 | 75 / 69 · 40 / 38 | 76 / 75 · 43 / 40 | 2,046 |
| trouser | 0.22 | 91 / 92 · 23 / 20 | 84 / 88 · 12 / 15 | 1,225 |
| pullover | 0.39 | 50 / 53 · 58 / 46 | 47 / 48 · 54 / 57 | 2,507 |
| dress | 0.26 | 69 / 75 · 29 / 27 | 72 / 70 · 28 / 26 | 1,579 |
| coat | 0.39 | 57 / 55 · 44 / 48 | 54 / 55 · 72 / 58 | 2,597 |
| **sandal** | **0.12** | **71 / 68 · 13 / 16** | **22 / 22 · 2 / 3** | **503** |
| shirt | 0.34 | 31 / 32 · 29 / 36 | 27 / 31 · 42 / 56 | 2,072 |
| sneaker | 0.16 | 81 / 82 · 16 / 16 | 76 / 77 · 7 / 9 | 912 |
| bag | 0.37 | 84 / 87 · 46 / 43 | 81 / 82 · 48 / 41 | 2,223 |
| boot | 0.29 | 91 / 86 · 41 / 41 | 91 / 93 · 34 / 34 | 1,670 |

("ink" = mean pixel intensity, "neurons" = layer-2 / output neurons carrying that label, seed 0 / seed 1; layer-1 spikes from
the seed-0 layer 1 replayed in numpy at 8 mV on 400 test images — 1,782 on average, 96 of 121 windows firing.)

- **The two-layer network loses 6 points to the single layer on Fashion-MNIST** (63.5 / 64.2 vs 69.4 / 69.6), where on
  MNIST they were level.
- **Most of the gap is one class: sandals.** Recall 22 % vs ~70 %; most sandals are labelled T-shirt (267 of the
  6-snapshot total in seed 0). Only 2–3 of the 400 output neurons carry the sandal label (single layer 13–16); sneakers get
  7–9 (single layer 16). The other classes are within a few points of the single layer.
- **Why: light images barely reach layer 2.** Sandals carry the least ink (0.12; average 0.29). Layer 1's fixed threshold
  turns that into 503 layer-1 spikes per image (average 1,782, coats 2,597; only 62 of 121 windows fire), a 3.5× deficit
  against the average where the ink deficit is 2.3×. Layer 2's weight scale is calibrated to the average image, so on a sandal
  its neurons get a fraction of the usual drive and few sandal-specific prototypes form. MNIST digits all carry similar ink
  (the issue never showed). The single layer sees the 2.3× pixel deficit directly and still forms 13–16 sandal neurons.
- **Shifts hurt less on Fashion-MNIST** (gap 16–18 vs 25–30): the objects are large and fill the frame.
- Likely fixes to test: normalise each image's total input (contrast normalisation before the Poisson spikes), or
  normalise layer 2's input by layer 1's total activity per image (divisive normalisation).
