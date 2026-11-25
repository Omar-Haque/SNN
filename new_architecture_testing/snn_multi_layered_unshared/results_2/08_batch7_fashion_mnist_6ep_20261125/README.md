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
bash unity/submit_multi_layer.sh smoke     # first, on your own computer: the four configurations on tiny settings
                                           # (downloads Fashion-MNIST to data/FashionMNIST the first time)
# on Unity, from the repo root, with the environment active (the script downloads Fashion-MNIST on the login node first):
module load conda/latest && conda activate snn
bash unity/submit_multi_layer.sh           # the four jobs
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2       # summary.csv, S1, S2 in results_2/
```

## Results
(to be filled in when the runs finish)
