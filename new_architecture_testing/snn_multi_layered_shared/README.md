# snn_multi_layered_shared: a two-layer STDP network

**Why.** A Diehl & Cook neuron that sees all 784 pixels learns a whole-digit template, so the single layer is a
prototype classifier whose ceiling is set by the number of prototypes. Layer-1 neurons with *local* receptive fields
can only learn strokes; layer 2 then learns prototypes over "which stroke is where". Two things to measure: does the
ceiling rise, and does the network become tolerant to translation (pixel templates are not).

## Architecture (defaults)

| | layer 1 (features) | layer 2 (prototypes) |
|---|---|---|
| input | 28×28 Poisson pixel spikes | the 1,024 layer-1 spike trains (256 with `--pool2`) |
| receptive field | 7×7 window, stride 3 → 8×8 = 64 positions | all layer-1 lines |
| neurons | 16 per position → 1,024 LIF cells | 400 (as before) |
| templates | **shared**: one 16×49 matrix; neuron (position p, feature f) uses template f at every p. `--unshared`: 1,024 private templates | 400 × 1,024 |
| competition | the 16 at a position inhibit each other (200 mV); positions independent | all-to-all 200 mV |
| learning | D&C STDP per neuron; with shared templates the 64 neurons of feature f write into the one template (mean over the batch, sum over positions) | D&C STDP, θ⁺ 0.5 |
| θ | per neuron, θ⁺ 0.1, decay τ 2k examples (layer-1 neurons win ~10× more often than layer-2 ones) | per neuron, θ⁺ 0.5, optional decay |
| normalisation | row sum 9.8 (mean weight 0.2 over 49), w_max 1 | row sum 0.1 × inputs, w_max 4 |

Training is sequential: **stage A** trains layer 1 alone (snapshots every 320 images: flat-template reservoir, θ,
usage per feature, fraction of ink windows that fired and their first-spike latency, weight change) and freezes it once
the reservoir has been empty for three snapshots and θ is stationary (cap `--l1-examples 9600`); **stage B** trains
layer 2 for `--epochs 2` with the full snapshot set of test4 plus, at every snapshot, accuracy on the same 1,000 test
images shifted by a fixed random offset of up to ±2 px (`--shift`). `--no-l1` feeds pixels to layer 2 directly: the
single-layer reference under identical evaluation.

The layer-2 LIF cell and D&C layer are copied verbatim from test4 (AST-checked; only the input count, row sum and
weight ceiling became parameters). The estimators are unchanged.

## What to look at

- **F15** (stage A): the layer-1 reservoir should hit 0 within a few hundred to a few thousand images, feature usage
  should spread across the 16 lines, and the fraction of ink windows that fire should approach 100 % with a latency
  well inside the 350 ms presentation. If windows don't fire or latency is >100 ms, raise `--l1-norm`.
- **F14**: the learned 7×7 templates. They should be strokes — bars, curves, ends, crossings — not blobs or noise.
- **F1**: layer-2 held-out accuracy (solid) and shifted accuracy (dashed). The single-layer reference should lose
  several points under shifts; a working feature front-end should lose less, pooling least.
- **F13**: layer-2 templates projected back to image space; they should look like digits assembled from strokes.
- **F2–F7** as before, for layer 2; `summary.csv`/S1/S2 across runs.

Known calibration risk: the drive into each layer is set by the row sums (`--l1-norm`, `--l2-norm`) and the first run
is the calibration. The console prints both layers' spike rates and layer 1's latency at every snapshot.

## How to run

```bash
cd new_architecture_testing/snn_multi_layered_shared
python multi_layer.py --smoke        # ~2-3 min: stage A, stage B, shifted eval and all figures on tiny settings
bash run_multi_layer.sh              # reference + default (2 seeds) + four one-knob comparisons, ~6 h
python summarize_multi_layer.py
```
