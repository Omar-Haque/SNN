# 09 — batch 8: per-image input normalisation (submitted from `unity/submit_multi_layer.sh`, 2026-11-26)

That script now holds a later batch; this batch's version: `git show 1d7b3b8:unity/submit_multi_layer.sh`.

## Purpose
On Fashion-MNIST (batch 7, `../08_batch7_fashion_mnist_6ep_20261125/`) the best two-layer network lost 6 points to the single
layer — 63.5 / 64.2 % vs 69.4 / 69.6 % — and most of the gap was sandals: recall 22 % vs ~70 %, only 2–3 of 400 output
neurons labelled sandal (single layer 13–16). The explanation: sandals are the lightest class (mean intensity 0.12–0.14 vs
0.29), layer 1's fixed threshold turns that into ~500 layer-1 spikes per image vs 1,782 on average, and layer 2 is calibrated
to the average image, so on a sandal its neurons get a fraction of the usual drive and few sandal prototypes form.

`--input-norm` tests that explanation directly: every image is scaled so its mean intensity equals the training set's mean,
so light and heavy images send the same number of pixel spikes on average. If the explanation is right, the sandal and
sneaker gap should mostly close for the two-layer network. The same change on MNIST checks that it costs nothing there.

## The normalisation
Each image x → x × (training-set mean intensity / the image's own mean intensity), pixels capped at 10 (Poisson spike
probability 0.1 × 10 = 1 per ms). Applied in the dataset transform, so training, calibration, scoring and the shifted test set
all see normalised images.

| | training-set mean | per-image scale factor | images with a capped pixel |
|---|---|---|---|
| MNIST | 0.131 | 0.33 – 5.1 | 0 of 60,000 |
| Fashion-MNIST | 0.286 | 0.38 – 14.8 | 4 of 60,000 |

Fashion-MNIST per-class mean intensity, before → after: T-shirt 0.33 → 0.29, trouser 0.22 → 0.29, pullover 0.38 → 0.29,
dress 0.26 → 0.29, coat 0.39 → 0.29, **sandal 0.14 → 0.29**, shirt 0.33 → 0.29, **sneaker 0.17 → 0.29**, bag 0.35 → 0.29,
boot 0.30 → 0.29.

## References (not rerun; mean of the snapshots of epochs 4–6, held-out / shifted ±2 px)
| network | dataset | seed | accuracy | sandal recall |
|---|---|---|---|---|
| single layer | Fashion-MNIST | 0 / 1 | 69.4 / 53.2 · 69.6 / 53.4 | 71 · 68 |
| two layers, stride 2 + 8 mV | Fashion-MNIST | 0 / 1 | 63.5 / 45.8 · 64.2 / 45.9 | 22 · 22 |
| single layer | MNIST | 0 | 85.6 / 60.6 | |
| two layers, stride 2 + 8 mV | MNIST | 0 | 85.3 / 56.5 | |

## Configuration
The batch-7 settings plus `--input-norm`: 6 epochs, layer-2 θ decay τ = 30k examples, snapshots every 200 batches, shifted
test set ±2 px. 5-h limit.

| job | dataset | network |
|---|---|---|
| `ml8_fm_single_in` / `_s1` | Fashion-MNIST | single layer (`--no-l1 --l2-wta`) |
| `ml8_fm_best_in` / `_s1` | Fashion-MNIST | two layers (`--stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10`) |
| `ml8_mn_single_in` | MNIST | single layer |
| `ml8_mn_best_in` | MNIST | two layers |

## What to watch
- Fashion-MNIST: the two-layer network against the single layer, and sandal / sneaker recall and the number of output
  neurons labelled with them (F7; `assignments_snap` in `run.npz`).
- Whether the single layer gains as well — it saw the light images' deficit only linearly (2.3×), so it should change less.
- MNIST: any change against the unnormalised runs (light digits such as 1s get brighter, heavy ones such as 0s and 8s darker).
- Layer 1's output rate into layer 2 (`l2_input_rate_per_ms`) — the average should stay about the same.

## Run and summarise
```bash
bash unity/submit_multi_layer.sh smoke     # first, on your own computer: the six configurations on tiny settings
# on Unity, from the repo root:
module load conda/latest && conda activate snn
bash unity/submit_multi_layer.sh           # the six jobs
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2       # summary.csv, S1, S2 in results_2/
python new_architecture_testing/snn_multi_layered_unshared/describe_runs.py \
    new_architecture_testing/snn_multi_layered_unshared/results_2/09_batch8_input_norm_6ep_20261126   # a README.md per run
```

## Results (Unity, 2026-11-26/27)

Each run folder has its own README.md with every setting, layer by layer (`../../describe_runs.py`).

Mean of the snapshots of epochs 4–6, held-out / shifted ±2 px; "before" = the same network without `--input-norm`
(batch 7 for Fashion-MNIST, batches 2 / 6 for MNIST). All six jobs finished (1.1–2.4 h).

| network | dataset | seed | before | with `--input-norm` | change | best | gap |
|---|---|---|---|---|---|---|---|
| single layer | Fashion-MNIST | 0 / 1 | 69.4 / 53.2 · 69.6 / 53.4 | 71.1 / 55.3 · 71.6 / 54.8 | +1.7 · +2.0 | 74.4 · 73.9 | 15.8 · 16.8 |
| two layers, stride 2 + 8 mV | Fashion-MNIST | 0 / 1 | 63.5 / 45.8 · 64.2 / 45.9 | **72.9 / 53.3 · 73.3 / 53.6** | **+9.4 · +9.1** | 75.6 · 75.4 | 19.6 · 19.7 |
| single layer | MNIST | 0 | 85.6 / 60.6 | 87.1 / 61.1 | +1.5 | 89.1 | 26.0 |
| two layers, stride 2 + 8 mV | MNIST | 0 | 85.3 / 56.5 | 85.3 / 56.2 | 0.0 | 88.3 | 29.1 |

Layer 1 alone: Fashion-MNIST 60.8 / 60.4 % (before 57.5 / 57.1), MNIST 67.0 (67.2). Layer-1 output into layer 2:
Fashion-MNIST 4.92 / 4.96 spikes/ms (before 4.97 / 5.02), MNIST 3.46 (3.32) — the average stayed put, as intended.

Fashion-MNIST per-class recall (last 6 snapshots) and output neurons carrying each label, seed 0 / seed 1:

| class | ink before | single layer, before | single layer, normalised | two layers, before | two layers, normalised |
|---|---|---|---|---|---|
| T-shirt | 0.33 | 75 / 69 · 40 / 38 | 75 / 74 · 44 / 42 | 76 / 75 · 43 / 40 | 75 / 74 · 47 / 42 |
| trouser | 0.22 | 91 / 92 · 23 / 20 | 91 / 92 · 31 / 30 | 84 / 88 · 12 / 15 | 92 / 92 · 25 / 26 |
| pullover | 0.38 | 50 / 53 · 58 / 46 | 56 / 57 · 47 / 36 | 47 / 48 · 54 / 57 | 58 / 56 · 49 / 51 |
| dress | 0.26 | 69 / 75 · 29 / 27 | 75 / 76 · 30 / 32 | 72 / 70 · 28 / 26 | 78 / 78 · 38 / 37 |
| coat | 0.39 | 57 / 55 · 44 / 48 | 55 / 54 · 41 / 43 | 54 / 55 · 72 / 58 | 54 / 57 · 54 / 57 |
| **sandal** | **0.14** | 71 / 68 · 13 / 16 | 74 / 79 · 30 / 33 | **22 / 22 · 2 / 3** | **78 / 89 · 26 / 31** |
| shirt | 0.33 | 31 / 32 · 29 / 36 | 36 / 36 · 31 / 42 | 27 / 31 · 42 / 56 | 38 / 38 · 39 / 44 |
| **sneaker** | **0.17** | 81 / 82 · 16 / 16 | 83 / 85 · 33 / 31 | 76 / 77 · 7 / 9 | 80 / 77 · 24 / 22 |
| bag | 0.35 | 84 / 87 · 46 / 43 | 89 / 86 · 39 / 39 | 81 / 82 · 48 / 41 | 92 / 90 · 40 / 38 |
| boot | 0.30 | 91 / 86 · 41 / 41 | 85 / 86 · 40 / 40 | 91 / 93 · 34 / 34 | 88 / 87 · 34 / 33 |

- **The explanation holds, and the gap closes.** With every image carrying the same ink, the two-layer network's sandal
  recall goes from 22 % to 78 / 89 % and its sandal neurons from 2–3 to 26–31; sandals labelled T-shirt fall from 267 to
  3 / 0 (6-snapshot totals). Sneaker neurons go from 7–9 to 22–24.
- **The two-layer network now beats the single layer on Fashion-MNIST:** 72.9 / 73.3 % vs 71.1 / 71.6 % (+1.8 / +1.7 on the
  same seeds), where it trailed by 6 points. It gained +9.1–9.4; sandals alone account for 55–75 % of that, the rest is spread over
  trousers (also light, +4–8), pullovers, dresses, shirts and bags (+6–11 each). Output neurons are shared out much more
  evenly (22–57 per class, before 2–72: coats had taken 72 in seed 0).
- **The single layer gains less (+1.7 / +2.0),** as expected: it saw the light-image deficit only linearly. Its sandal and
  sneaker neuron counts double (13–16 → 30–33) too.
- **Shift tolerance does not change the ranking:** shifted, the single layer stays ahead (55.3 / 54.8 vs 53.3 / 53.6); the
  two-layer gap is ~19.7 points vs ~16.3.
- **MNIST: no cost.** Two layers 85.3 / 56.2 (unchanged); single layer 87.1 / 61.1 (+1.5, one seed — within the 2-point seed
  spread seen for two layers). Labels even out here too (digit 1: two layers 14 → 37 neurons, single layer 19 → 34), but in
  the two-layer network the per-digit recall changes cancel (2, 8, 9 down 5–8; 4, 5 up 3–5).
- Boots lose a few points in three of the four Fashion-MNIST runs (−3 to −6), probably because boots now share the footwear
  region with many more sandal and sneaker neurons.
