# 07 — batch 6: combining and replicating batch 5's two winners (submitted from `unity/submit_multi_layer.sh`, 2026-11-22)

## Purpose
Batch 5 (`../06_batch5_layer1_code_stability_6ep_20261122/`) found two layer-1 changes that each beat the baseline by more
than run-to-run noise, one seed each:
- **stride 2** (121 overlapping windows instead of 64): 84.6 % held-out, 59.2 % shifted (baseline 82.1 / 57.4);
- **8 mV readout inhibition** (losing features keep firing after the winner, 4.4 per window): 84.2 % held-out, shifted
  unchanged (56.9).

The single layer is at 85.6 / 60.5. What tracked accuracy across batch 5 was how many layer-1 spikes per image reach layer
2 — and the two winners add them from different sources (more windows vs more features per window), so they may add up.
Batch 5 also showed that the ~25-point loss on ±2 px shifts did not follow the layer-1 code's shift stability: at 8 mV the
code is as shift-stable as raw pixels, yet shifted accuracy did not move. Pooling was tried once (batch 4), on the 200 mV
code, whose windows flip under a 1-px shift; on the 8 mV code it gets a fair test.

## Baselines (not rerun; all with `--l2-drive 0.36 --l2-lr-mult 10`, 6 epochs)
Accuracy = mean of every snapshot in epochs 4–6 (held-out / shifted ±2 px).

| run | seed | epochs 4–6 | folder |
|---|---|---|---|
| single layer | 0 | 85.6 / 60.5 | `../../results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |
| baseline layer 1 (stride 3, 200 mV) | 0 | 82.1 / 57.4 | `../05_batch4_best_followup_6ep_20261120/…_ep12_…` (first 6 epochs) |
| baseline layer 1 | 1 | 81.2 / 56.1 | `../05_batch4_best_followup_6ep_20261120/…_seed1_l1wta_l2wta_…` |
| 17.5 mV readout | 0 / 1 | 82.0 / 58.0 · 83.8 / 57.8 | `../05_batch4_best_followup_6ep_20261120/…_rd17.5_…` |
| pool2, 200 mV | 0 | 78.2 / 52.9 | `../05_batch4_best_followup_6ep_20261120/…_pool2_…` |
| stride 2 | 0 | 84.6 / 59.2 | `../06_batch5_layer1_code_stability_6ep_20261122/…_p7s2f16_…` |
| 8 mV readout | 0 | 84.2 / 56.9 | `../06_batch5_layer1_code_stability_6ep_20261122/…_rd8_…` |

Seed-0 runs at stride 3 share the same layer 1 (identical stage A), so the readout-inhibition runs differ only in the
readout. The baselines agree within ±0.5 on the epochs 4–6 average; count a change if it is ≥ 1.5–2 points.

## Configuration
Common to every job: `--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --l2-drive 0.36
--l2-lr-mult 10`; layer 1 otherwise default (7×7 windows, F = 16, row sum 12, w_max 1.5, θ⁺ 0.1, 200 mV training
inhibition, stage A up to 30,000 images).

| job | flags | question | compare with |
|---|---|---|---|
| `ml7_s2_rd8` | `--stride 2 --l1-readout-inhibition 8 --seed 0` | Do the two gains add up? | stride 2 (84.6), 8 mV (84.2), single layer (85.6) |
| `ml7_s2_rd8_s1` | same, `--seed 1` | … on a second seed | seed-1 baseline (81.2) |
| `ml7_s2_s1` | `--stride 2 --seed 1` | Is stride 2's gain real? | seed-1 baseline (81.2) |
| `ml7_rd8_s1` | `--l1-readout-inhibition 8 --seed 1` | Is 8 mV's gain real? | seed-1 baseline (81.2), 17.5 mV seed 1 (83.8) |
| `ml7_rd4` | `--l1-readout-inhibition 4 --seed 0` | Does an even softer readout keep helping? | 200 / 50 / 17.5 / 8 mV on the same layer 1: 82.1 / 81.7 / 82.0 / 84.2 |
| `ml7_pool2_rd8` | `--pool2 --l1-readout-inhibition 8 --seed 0` | Can 2×2 pooling buy shift tolerance once the code is stable? | pool2 at 200 mV (78.2 / 52.9), 8 mV without pooling (84.2 / 56.9); watch the held-out − shifted gap (~25–27 everywhere so far) |

`--pool2` needs an even window grid, so it runs at stride 3 (8×8 windows); at stride 2 (11×11) it would fail.
Stride-2 runs took ~2 h in batch 5 like the others; every job has a 5-h limit.

## What to watch
- Layer-1 output reaching layer 2 (`run.json` → `l2_input_rate_per_ms`, measured at calibration): 0.95 spikes/ms for the
  baseline, 1.55 at 8 mV, 2.03 at stride 2; the combination should give roughly 3–4. If accuracy keeps rising with it, the
  "more channels" reading holds; if it flattens, batch 5's curve was already near its ceiling.
- Layer-2 spikes per presentation (console, `summary.csv`): with many more input spikes per ms, calibration makes each
  one worth less; layer 2 should still fire ~10–15 times per presentation.
- Held-out − shifted gap for the pooling run.

## Run and summarise
```bash
git show ffdd549:unity/submit_multi_layer.sh   # the script now holds batch 7; this is batch 6's version
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2     # summary.csv, S1, S2 in results_2/
```

## Results (Unity, 2026-11-23; five of the six jobs first died on a faulty node, `gypsum-gpu083`, and were resubmitted)

Accuracy = mean of all snapshots in epoch 6 / in epochs 4–6; "gap" = held-out − shifted (epochs 4–6). Layer-1 columns:
each run's frozen layer 1 replayed in numpy on 300 test digits at its readout inhibition (spikes per image, distinct
window × feature channels that fired, code similarity for the same image twice and for a 1-px shift).

| run | seed | last 3 | epoch 6 | epochs 4–6 | shifted 4–6 | gap | best | layer-1 spikes / img | channels / img | same image | 1-px shift | layer-2 spikes / presentation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single layer (batch 2) | 0 | 85.9 | 85.9 | 85.6 | 60.6 | 25.0 | 87.7 | – | – | 0.97 (px) | 0.75 (px) | 20.7 |
| baseline (stride 3, 200 mV) | 0 | 84.2 | 82.7 | 82.1 | 57.4 | 24.7 | 84.7 | 308 | 36.6 | 0.75 | 0.39 | 14.4 |
| baseline | 1 | 82.4 | 82.0 | 81.2 | 56.1 | 25.1 | 83.2 | 305 | 36.5 | 0.74 | 0.40 | 14.3 |
| stride 2 (batch 5) | 0 | 84.8 | 84.5 | 84.6 | 59.2 | 25.4 | 86.2 | 655 | 76.7 | 0.74 | 0.40 | 13.8 |
| stride 2 | 1 | 84.9 | 85.1 | 84.4 | 58.0 | 26.4 | 86.3 | 652 | 76.6 | 0.74 | 0.40 | 13.7 |
| 8 mV (batch 5) | 0 | 85.6 | 84.9 | 84.2 | 56.9 | 27.3 | 87.2 | 500 | 159.5 | 0.98 | 0.74 | 11.0 |
| 8 mV | 1 | 83.7 | 84.5 | 84.1 | 55.7 | 28.4 | 85.9 | 496 | 159.9 | 0.98 | 0.74 | 11.0 |
| **stride 2 + 8 mV** | 0 | 86.8 | **86.3** | 85.3 | 56.5 | 28.9 | 87.3 | 1,069 | 343.2 | 0.98 | 0.74 | 10.3 |
| **stride 2 + 8 mV** | 1 | 87.3 | **87.5** | **87.3** | 57.4 | 29.9 | 88.9 | 1,064 | 343.7 | 0.98 | 0.74 | 10.3 |
| 4 mV | 0 | 84.8 | 84.6 | 84.8 | 56.0 | 28.8 | 86.8 | 771 | 240.4 | 0.98 | 0.80 | 9.3 |
| pool2 + 8 mV | 0 | 81.9 | 81.5 | 81.8 | 56.9 | 24.9 | 84.8 | 500 (before pooling) | 159.5 | 0.98 | 0.74 | 7.1 |
| pool2, 200 mV (batch 4) | 0 | 77.2 | 78.5 | 78.2 | 52.9 | 25.3 | 80.6 | 308 (before pooling) | 36.6 | 0.75 | 0.39 | 9.4 |

Per epoch (held-out / shifted):

| run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| single layer | 73.5 / 49.3 | 84.6 / 57.8 | 85.6 / 59.8 | 85.5 / 60.2 | 85.4 / 60.5 | 85.9 / 60.9 |
| stride 2 + 8 mV, seed 0 | 75.5 / 42.9 | 82.5 / 51.0 | 84.1 / 53.8 | 84.4 / 54.8 | 85.2 / 56.4 | 86.3 / 58.1 |
| stride 2 + 8 mV, seed 1 | 75.4 / 44.4 | 84.1 / 51.7 | 85.2 / 54.7 | 86.4 / 56.3 | 87.8 / 57.4 | 87.5 / 58.2 |

- **Both batch-5 winners replicate on seed 1**: stride 2 +3.2 points over the seed-1 baseline (seed 0: +2.5), 8 mV +2.9
  (seed 0: +2.1).
- **Together they reach the single layer on centred digits.** In epoch 6 both seeds are above it (86.3 and 87.5 vs 85.9),
  and seed 0 was still climbing; best snapshots 87.3 / 88.9 vs 87.7. The two seeds differ by 1–2 points although their
  layer 1s are statistically indistinguishable (1,069 vs 1,064 spikes per image, same stability), so the spread comes from
  layer 2's learning. One single-layer seed at 6 epochs is the reference: more seeds of both are needed before calling it
  better.
- **Shifted digits did not follow.** Shifted accuracy stays at 56–59 % in every configuration of batches 4–6 (single layer
  60.6), so every gain on centred digits widens the gap: ~25 for the 200 mV runs, 27–30 with the soft readout.
- **Pooling does not buy shift tolerance even on the stable code**: pool2 + 8 mV has the same shifted accuracy as 8 mV
  without pooling (56.9) and loses 2.4 points held-out. The shift loss sits in what layer 2 learns, not in layer 1's code.
- **4 mV** adds +0.6 over 8 mV on the same layer 1 — inside the noise; the readout curve flattens (200 / 50 / 17.5 / 8 / 4 mV
  → 82.1 / 81.7 / 82.0 / 84.2 / 84.8).
- **Accuracy keeps tracking layer-1 spikes per image**, still rising at the top: ~305 → 81–82 %, ~500 → 84.1–84.2,
  ~650 → 84.4–84.6, 771 → 84.8, ~1,065 → 85.3–87.3 (epochs 4–6). Layer 2 itself fires less as its input grows (14 → 10
  spikes per presentation).
