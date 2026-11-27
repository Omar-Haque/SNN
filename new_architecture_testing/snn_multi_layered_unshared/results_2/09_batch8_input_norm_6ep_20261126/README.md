# 09 — batch 8: per-image input normalisation (submitted from `unity/submit_multi_layer.sh`, 2026-11-26)

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
```

## Results
(to be filled in when the runs finish)
