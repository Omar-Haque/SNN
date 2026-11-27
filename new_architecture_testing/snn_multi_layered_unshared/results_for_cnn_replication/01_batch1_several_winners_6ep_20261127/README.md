# 01 — batch 1: several winners per image in layer 2 (submitted from `unity/submit_multi_layer.sh`, 2026-11-27)

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

## Results
(to be filled in when the runs finish)
