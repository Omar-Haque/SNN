# 01 — batch 1: several winners per image in layer 2 (submitted from `unity/submit_multi_layer.sh`, 2026-11-27)

That script now holds a later batch; this batch's version: `git show 22255be:unity/submit_multi_layer.sh`.

## Purpose
See `../README.md` for the aim. In the best two-layer network, layer 2's 400 neurons inhibit each other at 200 mV and a
tie-break lets at most one fire per millisecond, so one neuron usually takes each image and learns a whole-digit prototype.
This batch lets several neurons share each image, in two ways, and measures whether they learn parts:

- **Softer lateral inhibition** (`--inhibition 50 / 20 / 8`, mV per spike; baseline 200). The tie-break stays on (one spike
  per millisecond), but the winner's spikes no longer silence everyone: a neuron whose drive beats the inhibition can still
  fire. How many share an image is not fixed — it depends on how well each neuron matches.
- **Exactly k winners per image** (new `--l2-k 2 / 3 / 5 / 10`). Nobody is inhibited until the image has k winners: the
  first k neurons to reach threshold — the k best-matching templates — become that image's winners. From then on only they
  may fire, they never inhibit each other, and every other neuron is silenced (200 mV inhibition, and its spikes are
  dropped). This replaces the one-spike-per-millisecond tie-break; if more neurons cross in the same millisecond than there
  are places left, the ones with the highest potential get them.

The readout and everything else are unchanged.

## What we expect, and what could go wrong
- With k winners, each neuron takes part in ~k× as many images. If those images share a part (the right-hand vertical stroke
  of 1, 4, 7, 9), STDP on many images reinforces that part and lets the rest fade under the row normalisation — the hoped-for
  route to parts. But every winner still potentiates on the *whole* image, so the winners may instead become blurrier
  prototypes that cover several similar digits. F16 tells the two apart.
- With soft inhibition, 8 mV may let dozens of neurons fire on every image; they would all drift toward the same average
  digit and accuracy would collapse. 50 mV may change almost nothing. 20 mV is the middle guess.
- More spikes per image make the adaptive thresholds rise faster (θ rises 0.5/16 mV per spike). `c1_k3tp1` doubles that rise
  so neurons that win too often give way to others sooner.

## Reference (batch 6, same settings; mean of the snapshots of epochs 4–6, held-out / shifted ±2 px)
| network | seed | accuracy |
|---|---|---|
| two layers, stride 2 + 8 mV (one winner per image) | 0 / 1 | 85.3 / 56.5 · 87.3 / 57.4 |
| single layer | 0 | 85.6 / 60.6 |

## Configuration
Every job: MNIST, 6 epochs, layer-2 θ decay τ = 30k images, snapshots every 200 batches, shifted test set ±2 px, and the
best two-layer network: `--stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10`.
6-h limit.

| job | layer 2 | seed |
|---|---|---|
| `c1_base` | unchanged: 200 mV, one winner per image (reference; also checks that the new code changes nothing here) | 0 |
| `c1_inh50` / `c1_inh20` / `c1_inh8` | lateral inhibition 50 / 20 / 8 mV | 0 |
| `c1_k2` / `c1_k3` / `c1_k5` / `c1_k10` | k = 2 / 3 / 5 / 10 winners per image | 0 |
| `c1_k3tp1` | k = 3, θ+ 1.0 instead of 0.5 | 0 |
| `c1_i20s1` / `c1_k5s1` | 20 mV / k = 5, second seed | 1 |

New in the code for this batch (`multi_layer_unshared.py`): `--l2-k`; at the end of every run, one pass over the 1,000
test digits records what each layer-2 neuron fires for (`sta2`, `class_counts2`, `distinct_test` in run.npz;
`test_distinct_per_image`, `class_purity_median` in run.json), drawn as **F16**; distinct neurons per image at every
snapshot (`distinct_snap`). Run folders get `_inh<mV>` / `_k<k>` in their names.

## What to watch
- **F16 and class purity:** prototypes (digit-shaped averages, purity near 1) or parts (strokes and loops, purity well below
  1)? `c1_base` gives the prototype reference values.
- **Distinct neurons per test image** (`test_distinct_per_image`): does softer inhibition actually spread each image over
  several neurons, and how many?
- **Accuracy** against `c1_base` / 85.3 %: a part code may cost accuracy under the label vote (see `../README.md`).
- **Shifted accuracy:** parts that several digits share might tolerate shifts better than whole-digit prototypes.
- Neurons that never fire (`test_neurons_fired`) and F6 (θ): with k winners, a few neurons may take most images.

## Run and summarise
The k-winner rule, the test-digit statistics and F16 are new and have not run under torch yet, so run the local smoke
tests first:
```bash
bash unity/submit_multi_layer.sh smoke     # on your own computer: all 11 configurations on tiny settings
bash unity/submit_multi_layer.sh           # the 11 jobs on Unity
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication
```
Each run folder gets its own README.md with every setting when the run finishes.

## Results (Unity, 2026-11-27/28)

Mean of the snapshots of epochs 4–6, held-out / shifted ±2 px. On the 1,000 test digits: "distinct" = layer-2 neurons that
fire per digit, "spikes" = layer-2 spikes per digit, "purity" = median over neurons (≥ 10 test spikes) of the share of a
neuron's spikes that come from its top digit, "fired" = neurons that fired at all. All 11 jobs finished (1.9–3.4 h).

| job | layer 2 | epochs 4–6 | gap | best | distinct | spikes | purity | fired |
|---|---|---|---|---|---|---|---|---|
| `c1_base` | one winner (200 mV) | 85.3 / 56.5 | 28.9 | 87.3 | 1.0 | 9.5 | 1.00 | 309 |
| `c1_inh50` | 50 mV | 85.6 / 57.5 | 28.1 | 87.1 | 1.2 | 9.6 | 1.00 | 312 |
| `c1_inh20` / `c1_i20s1` | 20 mV | 85.5 / 57.3 · 84.9 / 57.8 | 28.2 · 27.2 | 86.5 · 85.8 | 7.1 · 7.1 | 12.3 · 12.1 | 0.91 · 0.90 | 398 · 398 |
| `c1_inh8` | 8 mV | 76.9 / 50.3 | 26.6 | 80.0 | 12.2 | 17.7 | 0.87 | 400 |
| `c1_k2` | k = 2 | **89.2 / 60.8** | 28.3 | 90.4 | 2.0 | 13.6 | 1.00 | 372 |
| `c1_k3` | k = 3 | **88.7 / 59.0** | 29.7 | 89.6 | 2.9 | 16.3 | 1.00 | 381 |
| `c1_k5` / `c1_k5s1` | k = 5 | 84.4 / 54.1 · 83.6 / 53.4 | 30.3 · 30.1 | 85.3 · 84.5 | 4.7 · 4.7 | 19.8 · 19.5 | 0.95 · 0.94 | 398 · 396 |
| `c1_k10` | k = 10 | 71.9 / 43.7 | 28.2 | 80.9 | 8.4 | 24.5 | 0.90 | 400 |
| `c1_k3tp1` | k = 3, θ+ 1.0 | 84.4 / 54.5 | 29.9 | 85.6 | 2.8 | 10.9 | 1.00 | 382 |

- **`c1_base` reproduces batch 6 exactly** (85.3 / 56.5, same snapshots), so the new code changes nothing when it is off.
- **No parts, in any run.** In F16 every neuron's average is a whole digit (`C1_sta_comparison.png` in this folder puts six
  runs side by side: the 3 most active neurons per digit), and the median purity stays at 0.87–1.00. Sharing
  an image does not split it: the k winners are the k best-matching prototypes, almost always of the same digit, and each
  of them potentiates on the *whole* image, so they all move toward it. Nothing makes the winners divide the image between
  them — they compete for the image, not for its parts.
- **Too many winners merge similar digits.** With k = 10 or 8 mV, the 4 and 9 prototypes blend: neurons labelled 9 get 32–34 %
  of their spikes from 4s (baseline 15 %), and 241 / 312 of the 660 test 4s (6-snapshot totals) are called 9 (baseline 126).
  Accuracy falls (71.9 / 76.9 %) and keeps falling across epochs (k = 10: 73.7 → 71.6; 8 mV: 79.1 → 76.6). With 20 mV
  each digit reaches ~7 neurons, but most of the extra ones fire only once or twice; accuracy is unchanged (85.5 / 84.9) with
  a little 4/9 merging. 50 mV changes nothing (1.2 neurons per digit).
- **But k = 2–3 is the best two-layer result so far: 89.2 / 60.8 % and 88.7 / 59.0 %** on seed 0, against 85.3 / 56.5 for the
  baseline on the same seed (best snapshot 90.4). Shifted accuracy reaches the single layer's (60.8 vs 60.6), though the gap
  is unchanged (28.3), so it rises with the centred accuracy rather than through shift tolerance. Two reasons are visible:
  more neurons get trained (372–381 fire on the test digits instead of 309; 4 / 0 unlabelled instead of 17), and every digit
  is now decided by its 2–3 closest prototypes instead of one — a small nearest-prototypes vote. With k = 2, 4s called 9
  halve (65 vs 126), and recall rises by 5–11 points for 2, 3, 4 and 5 (9 loses 7). One seed: the baseline's two seeds differ by 2
  points, so k = 2 and 3 need seeds 1 and 2.
- **A stronger threshold rise hurts** (k = 3 with θ+ 1.0: 84.4 vs 88.7): it pushes images toward worse-matching neurons.
- **k = 5 is already past the optimum** (84.4 / 83.6 on two seeds; purity 0.94–0.95: the first signs of merging).

**For the aim of this folder:** several winners per image is not enough to get parts. The neurons have to compete for the
image's *inputs*, not just for the image. Three ways to make them do that: local receptive fields in the top layer (the
architectural route); a winner's spikes suppressing the input lines it already accounts for, so the next winner has to match
what is left ("explaining away"); or learned inhibition between neurons that tend to fire together (anti-Hebbian lateral
weights, which push co-active neurons toward different inputs).
