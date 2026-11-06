# snn_multi_layered_unshared: a locally connected two-layer STDP network

Every 7×7 window (stride 3 → 8×8 = 64 positions) has its own dedicated set of 16 neurons; each neuron has its own
49-weight template and competes (200 mV) only with the 15 others at its window. 1,024 layer-1 neurons, 1,024 private
templates, nothing shared or averaged across positions. Layer 2 is the 400-neuron D&C layer reading the 1,024 layer-1
spike trains (256 with `--pool2`). `--stride 7` gives the fully disjoint version: 4×4 non-overlapping tiles × 16 = 256
neurons.

## Why unshared, and what was wrong with the shared run

1. **Shared templates never imprinted.** With updates summed over positions, a template that wins 20 windows of one
   image is pulled toward the *average* of 20 unrelated patches; every template drifts to the same blob and the
   competition never breaks symmetry (top‑6 share stayed 0.20–0.28 for 9,600 images). Here each neuron imprints
   exactly the window it won — the single-layer mechanism replayed at 64 positions — so layer 1 learns the way the
   network we understand learns. (The shared design would need the convolutional-STDP rule "only a feature's
   best-matching window updates it per image"; that is the diagnosis to come back to.)
2. **Layer 2 never fired.** Layer 1 delivers ~0.7 input spikes/ms, not the ~3 the default weight scale assumed, so a
   flat layer-2 neuron's drive was 20× too small. This script measures layer 1's output rate after stage A and sets
   layer 2's mean weight so the initial drive equals the pixel network's (`--l2-drive 1.2` mV/ms), with
   w_max = 10 × mean weight as in the base file. `--l2-norm` / `--l2-wmax` override it.

## Training
- **Stage A**: layer 1 alone, up to `--l1-examples 30000` (each template sees one window per image, so it converges at
  the single-layer pace), snapshots every 640 images; frozen once the reservoir of flat templates has stopped
  shrinking for 5 snapshots and θ is stationary. Windows at the border that never see ink keep flat templates and
  simply never fire — the reservoir count will not reach 0, and that is expected.
- **Calibration**: 5 batches through the frozen layer 1 → layer-2 weight scale.
- **Stage B**: layer 2 for `--epochs 2` with the test4 snapshot set plus the shifted-test-set accuracy.

## What to look at
- Console during stage A: "flat templates" should fall from 1,024 to roughly the number of templates at border windows,
  "windows fired" should approach 100 %, latency should drop as templates sharpen.
- `[calibration]` line: the measured rate and the weight scale it chose.
- Stage A console, "per active window: N spikes from D features, T in its first spiking ms": D and T should be ≈ 1
  (one feature wins the window and keeps firing). D or T of several means the window's features fire together in a
  volley — see the findings below — and `--l1-wta` is needed.
- Stage B console: layer-2 spikes per presentation should be ~20–30 like the single layer, "from ~1 distinct neuron";
  if it is ~0 the drive is still too low (raise `--l2-drive`); if it is "59 from 59 distinct neurons" layer 2 is
  volleying and `--l2-wta` is needed.
- F15 (stage A curves), F14 (the 16 templates at four windows: strokes, not blobs), F1 (held-out and shifted
  accuracy), F13 (layer-2 templates projected to image space), `summary.csv` / S1 / S2.

## Diagnostics
- `replay_layer1.py results/<run> [--inhibition X --no-wta --pooling]`: torch-free replay of the frozen layer 1 on test digits —
  code sparsity, within- vs between-class similarity (against raw pixels), nearest-centroid accuracy, pooled variants.

## Run
```bash
cd new_architecture_testing/snn_multi_layered_unshared
python multi_layer_unshared.py --smoke          # ~3 min: stage A, calibration, stage B, figures, on tiny settings
bash run_multi_layer_unshared.sh                # reference + design (2 seeds) + stride 7 / pool2 / 32 features
```

## Findings so far

**Run 1 — no tie-break (`unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_20261031-135637`): 69.7 % held-out, 44 %
on the 2-px-shifted test set, 395/400 layer-2 templates below the 0.35 sharpness line at the end.** More training would
not have helped; the network was in a *volley* regime in both layers:

- Layer 2 fired 59 spikes per presentation at 3k examples, falling to 6 by 118k, and the spikes were spread over all
  400 neurons (the 20 most active neurons produced only 5–14 % of them; the most active neuron fired on 6 % of
  presentations). In the single-layer network a presentation is ~27 spikes from *one* neuron. So each presentation was
  one synchronous crossing by a few dozen neurons, one spike each, followed by silence (each tied neuron receives
  −200 mV per other tied neuron). No neuron ever took a presentation, so no template was ever imprinted by one image;
  each got single-spike nudges from whatever drove the volley, and the templates averaged everything (top-131 share
  0.23 → 0.30, recognisable digits in image space but low contrast). Every θ rose in lockstep (p10/median/p90 within a
  few mV of each other at every snapshot) to a median of +29 mV with no decay, 67 neurons above +55 mV, which a
  mean-drive template cannot reach (v∞ = −65 + 100·1.2 = +55 mV).
- Layer 1 shows the same signature: in the central windows all 16 features fire regularly (θ > −51 for all 16) and
  their templates are blurred (max top-k share 0.25–0.37), while border windows, which see ink rarely, have sharp
  templates (0.5+). The 16 features of a window see the identical input and inhibition arrives one step late, so once
  their templates are similar they cross threshold in the same millisecond and all fire; the identical STDP update
  keeps them similar. The 320 layer-1 spikes per presentation then arrive at layer 2 as a burst, which is what makes
  layer 2 volley.
- The same simultaneous-crossing effect exists in the single-layer baseline (STDP_testing/T1_ties: ~45 % of
  presentations begin with a tie), but there the random 784-pixel templates differ enough that a solo winner usually
  emerges; with 49-pixel windows, bursty input and identical templates it becomes the rule.

The fix is a within-step tie-break, which BindsNET's `DiehlAndCookNodes` applies by default (`one_spike=True`: one
crossing neuron per step, chosen at random); ours keeps the crossing neuron with the highest potential, as in the
WTA block of the original code: `--l1-wta` (per window) and `--l2-wta` (layer 2).
The 20-most-active templates in F13 of run 1 are not useless — they vote at 70 % — but they are averages, not imprints.

**Batch 1 on Unity (2026-11-02, tie-break in both layers, 2 epochs, no θ decay)** — `summary.csv`:

| run | held-out | shifted (2 px) |
|---|---|---|
| single-layer reference, `--no-l1 --l2-wta` | 86.8 | 60.6 |
| two-layer, `--l1-wta --l2-wta`, seeds 0 / 1 | 68.9 / 67.8 | 41.3 / 43.7 |
| `--l2-wta` only | 68.6 | 44.4 |
| `--features 32` | 68.7 | 45.6 |
| `--pool2` | 65.9 | 41.9 |
| `--stride 7` (disjoint windows) | 52.8 | 33.9 |

What the batch established, in order:

1. *Layer 1 works.* With the tie-break, exactly one feature fires per active window (D = T = 1.00 in the stage-A log), and
   every central window learned 16 distinct oriented stroke bands (F14). The old "flat templates" count (898/1024) was the
   top-6 share applied to bands 20 pixels wide; by spatial structure (1-px autocorrelation, now the criterion) 100 % of the
   central-window templates and 88 % of the ring are formed, and the remainder are windows that never see ink.
2. *The layer-1 code is better than pixels for a template matcher.* `replay_layer1.py` on 1,000 test digits: a
   nearest-class-centroid classifier gets 87.0 % on the code vs 82.0 % on raw pixels; within-class vs between-class
   contrast is 3.6x (Jaccard) / 4.0x (rate cosine) vs 1.5x for pixels. Softer readouts do not add information (17.5 mV:
   86.3 %; no inhibition: 80.0 %), nor does pooling (82–84 %).
3. *The D&C layer learns that code slowly* — 54 % at 29k examples where the pixel layer is at 82 %, then a logarithmic
   climb (64 % at 67k, 69 % at 118k, still rising 3–5 points per quarter-run). Its dynamics are matched to the single layer
   (spikes per presentation 26 → 7 in both, same θ climb, w_max/mean and norm scaled identically), so the input statistics
   are the only difference, and they differ in the one quantity the learning loop depends on: two instances of a class share
   ~15 % of their active channels (37 of 1,024), where in pixel space they share most of their ink. A neuron imprints the
   single instance it won and is barely favoured for the next one, so recruitment is a lottery and templates become diffuse
   unions (top-131 share ≤ 0.3 — also why the layer-2 "reservoir" never empties: with the calibrated w_max a 37-channel
   template cannot exceed ~0.42, so that criterion is replaced by the cosine to the initial row, `reservoir_init`).
4. pool2's falling reservoir count is pooled codes being more consistent, not accuracy: it is the variant that flattened.

**Batch 2 (queued 2026-11-06, `unity/submit_multi_layer.sh`)** tests that explanation: 6 epochs with layer-2 θ decay
τ 30k (control: time alone, with θ stationary instead of climbing past +30 mV); `--features 8` and `4` (coarser partition
per window, so the same stroke maps to the same channel more often); `--l1-readout-inhibition 17.5` (features learned at
200 mV, read out with a runner-up allowed: +50 % within-class overlap at 2.5x contrast in the replay); and the single-layer
reference in the same regime.
