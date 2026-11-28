# 02 — batch 2: three mechanisms for parts, and k winners followed up (2026-11-28)

Submitted from two scripts that write into this folder: `unity/submit_multi_layer.sh` (two-layer jobs, `c2_*`) and
`unity/submit_three_layer.sh` (three-layer jobs, `t2_*`, run folders starting `3L_`).

## Purpose
Batch 1 (`../01_batch1_several_winners_6ep_20261127/`) found two things:
- **k = 2–3 winners per image in layer 2 is the best two-layer network so far:** 89.2 / 88.7 % on seed 0, against 85.3 % for one
  winner. One seed.
- **But no parts:** every layer-2 neuron still learned a whole digit (class purity 0.87–1.00). The k winners are the k
  best-matching prototypes, usually of the same digit, and each learns from the whole image: they compete for the image, not
  for its parts.

This batch does two things at once.

**A. Follow the k-winner lead** (accuracy):
1. Replicate k = 2 and 3 on MNIST, seeds 1 and 2, plus a third seed of the one-winner baseline.
2. Try k = 2 and 3 on Fashion-MNIST with input normalisation (one winner: 72.9 / 73.3 %, batch 8).
3. Try k = 2 and 3 in the **three-layer network's layer 3** (new `--l3-k`), on its best configuration (batch 3's
   `tl3_a2r4_b3s1`: 86.0 %, one seed), and run that configuration's second seed.

**B. Three mechanisms that make neurons compete for the image's inputs** (parts):
1. **Explaining away** (new `--l2-explain`, with k winners, two-layer layer 2). When a neuron wins an image it "claims" the
   input lines its template covers (its weights scaled so the strongest is 1). Every other neuron then receives only the
   unclaimed part of the input, so the next winner is the neuron that best matches *the rest of the image*. Each winner learns
   from what the *other* winners leave. If one prototype explains the whole digit, the image gets one winner; if a template
   covers only part of it, another neuron is drawn to the remainder — the route by which parts can form.
2. **Learned lateral inhibition** (new `--l2-lat-lr`, two-layer layer 2). After every batch, the inhibition between two
   neurons grows in proportion to how often they fired on the same image, and shrinks slowly toward zero otherwise
   (anti-Hebbian, after Földiák 1990; target: ~5 of 400 neurons per image). It starts from soft inhibition (20 or 8 mV), where
   several neurons share each image. Neurons that keep firing together come to inhibit each other, which pushes them toward
   different images or different parts.
3. **Local receptive fields at the top** (three-layer network, large layer-2 blocks). Layer 2 blocks of 5×5, 7×7 or 9×9
   layer-1 windows (15–23 px) with 25–96 features each: every neuron sees one region of the image, so it can only learn a
   part, and the neurons of a block compete only with each other. **"Layer 2 alone"** — layer 2 scored as the output layer at
   the end of stage A2, every block's winner voting — is this local top layer's accuracy, with no whole-image layer at all.
   Layer 3 is then trained on top as usual, to see whether larger parts make a better input for it.

## References (mean of the snapshots of epochs 4–6, held-out / shifted ±2 px)
| network | seed | accuracy |
|---|---|---|
| two layers, one winner (MNIST) | 0 / 1 | 85.3 / 56.5 · 87.3 / 57.4 |
| two layers, k = 2 / k = 3 (batch 1) | 0 | 89.2 / 60.8 · 88.7 / 59.0 |
| two layers, one winner, Fashion-MNIST + input normalisation | 0 / 1 | 72.9 / 53.3 · 73.3 / 53.6 |
| three layers, best (`tl3_a2r4_b3s1`) | 0 | 86.0 / 56.6 |
| three layers, layer 2 alone with 5×5 blocks × 16 (batch 1) | 0 | 67.4 (end of stage A2) |
| single layer | 0 | 85.6 / 60.6 |

## Configuration
Two-layer jobs: MNIST (unless noted), 6 epochs, θ decay τ 30k, snapshots every 200 batches, and the best two-layer network
(`--stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10`). Three-layer jobs: `three_layer.py`
defaults plus the flags below. 6-h limit (two-layer), 10 h (explaining away, three-layer).

| job | change | seed |
|---|---|---|
| **A1** `c2_bs2` | one winner (baseline) | 2 |
| `c2_k2s1` / `c2_k2s2` | k = 2 | 1 / 2 |
| `c2_k3s1` / `c2_k3s2` | k = 3 | 1 / 2 |
| **A2** `c2_fk2` / `c2_fk2s1` | Fashion-MNIST + `--input-norm`, k = 2 | 0 / 1 |
| `c2_fk3` / `c2_fk3s1` | Fashion-MNIST + `--input-norm`, k = 3 | 0 / 1 |
| **A3** `t2_best_s1` | three layers, `--l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-stride 1` (best) | 1 |
| `t2_k2` / `t2_k2s1` | the same + `--l3-k 2` | 0 / 1 |
| `t2_k3` / `t2_k3s1` | the same + `--l3-k 3` | 0 / 1 |
| **B1** `c2_ea2` / `c2_ea3` / `c2_ea5` / `c2_ea10` | `--l2-explain` with k = 2 / 3 / 5 / 10 | 0 |
| `c2_ea5s1` | k = 5 + explaining away | 1 |
| **B2** `c2_lat5` / `c2_lat20` | `--inhibition 20 --l2-lat-lr 5` / `20` | 0 |
| `c2_l20i8` | `--inhibition 8 --l2-lat-lr 20` (start softer: more co-firing to learn from) | 0 |
| `c2_l20p5` | `--inhibition 20 --l2-lat-lr 20 --l2-lat-target 0.005` (sparser target: ~2 neurons per image) | 0 |
| **B3** `t2_lrf5` | layer-2 blocks 5×5 windows (15 px), stride 2 → 16 blocks × 25 features | 0 |
| `t2_lrf7` | 7×7 windows (19 px), stride 2 → 9 blocks × 48 | 0 |
| `t2_lrf7f96` | 7×7, 9 blocks × 96 | 0 |
| `t2_lrf9` | 9×9 windows (23 px), stride 1 → 9 blocks × 48 | 0 |
| `t2_lrf7k2` | 7×7 × 48, plus k = 2 in layer 3 | 0 |

All B3 jobs also use `--l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-examples 60000`.

New in the code: `--l2-explain`, `--l2-lat-lr` / `--l2-lat-target` / `--l2-lat-max` (`multi_layer_unshared.py`); `--l3-k`, and class
purity for layers 2 and 3 (`l2_class_purity_median`, `l3_class_purity_median` in run.json; `class_counts2/3` in run.npz) in
`three_layer.py`. The learned inhibition matrix is saved as `inh2` in run.npz.

## What to watch
- **Parts:** two-layer runs: F16 and `class_purity_median` (one winner: 1.00; parts: well below 1, F16 shows partial
  digits). Three-layer runs: F6 (what each layer-2 neuron fires for, per block) and `l2_class_purity_median`; F7 and
  `l3_class_purity_median` for layer 3.
- **Explaining away:** how many winners an image actually gets (`test_distinct_per_image`): if one prototype explains
  everything, the residual may be too weak for a second neuron to reach threshold, and the run behaves like one winner.
- **Learned inhibition:** whether it settles into a few neurons per image that cover different parts, or simply learns the
  baseline's "one winner per image" back (`distinct_snap` over training; the matrix `inh2`).
- **Local receptive fields:** "layer 2 alone" (the local top layer's accuracy) against batch 1's 67.4 % with 5×5 blocks × 16
  and against the two-layer 85–89 %; then layer 3's accuracy on top.
- **k follow-ups:** do 89.2 / 88.7 % hold on seeds 1–2; does k help on Fashion-MNIST and in layer 3?

## Run and summarise
The new mechanisms have not run under torch yet, so run both smoke sets first:
```bash
bash unity/submit_multi_layer.sh smoke     # on your own computer: the 18 two-layer configurations, tiny settings
bash unity/submit_three_layer.sh smoke     # the 10 three-layer configurations
bash unity/submit_multi_layer.sh           # on Unity: 18 jobs
bash unity/submit_three_layer.sh           # on Unity: 10 jobs
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication       # two-layer runs
python three_layer_architecture/summarize_three_layer.py --prefix three_layer_ \
    --results new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication       # three-layer runs
```

## Results
(to be filled in when the runs finish)
