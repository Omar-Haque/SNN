# 06 — batch 5: layer-1 changes aimed at a more stable code (submitted from `unity/submit_multi_layer.sh`, 2026-11-22)

## Purpose
Batch 4 (`../05_batch4_best_followup_6ep_20261120/`) put every layer-2 variant at 82.4–83.8 % against the single layer's
85.9 %, and every two-layer run loses 23–24 points on digits shifted by ±2 px (single layer 25). More epochs, a lower
drive, the soft readout and pooling did not move either number, so this batch changes layer 1 only.

The defect it targets (`../../layer1_analysis/README.md`, sections 6–7): each active window outputs the one feature that
crosses threshold first, and a window's features are near-copies of each other (the same stroke a pixel over or slightly
tilted; median cosine to the closest sibling 0.72). The race is close — median gap 3.9 ms between the two fastest
features, under 2 ms in a third of the windows — so Poisson noise in the first ~40 ms, or a 1-px shift, decides the winner.
The same image shown twice keeps 75 % of its code (pixels: 97 %); shifted by one pixel, 40 % (pixels: 75 %).

## Baselines (not rerun)
All seed 0 with the same layer-2 setting. "Epoch 6" = mean of every snapshot in the 6th epoch, the least noisy number.

| run | held-out (last 3) | epoch 6 | shifted | layer 1 alone (shifted) | folder |
|---|---|---|---|---|---|
| best, batch 3 | 82.3 | 83.0 | 57.7 | not measured | `../../results/04_batch3_layer2_drive_lr_6ep_20261113/…_drv0.36_lr10_…` |
| best, batch 4 12-epoch run, at 6 epochs | 84.2 | 82.7 | 59.4 | 67.3 (48.1) | `../05_batch4_best_followup_6ep_20261120/…_ep12_…` |
| best + 17.5 mV readout, batch 4 | 82.6 | 82.7 | 59.6 | 67.3 (48.1) | `../05_batch4_best_followup_6ep_20261120/…_rd17.5_…seed0…` |
| single layer | 85.9 | 85.9 | 60.8 | – | `../../results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |

The first two rows are the same configuration and seed: 82.3 vs 84.2 on the last-3 measure, 83.0 vs 82.7 on the
epoch-6 average. That is the noise to read against — count a change only if it is ≥ 1.5–2 points better.
Runs 3–4 below keep the default layer 1 with seed 0, so their stage A is the same as batch 4's seed-0 runs (layer 1 alone
67.3 %).

## Configuration
Common to every job: `--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --seed 0 --l2-drive 0.36
--l2-lr-mult 10`. Layer 1 otherwise default: 7×7 windows, stride 3, F = 16, row sum 12, w_max 1.5, θ⁺ 0.1, 200 mV
inhibition, stage A up to 30,000 images.

| job | flags | question | why, and what to watch |
|---|---|---|---|
| `ml7_l1norm6` | `--l1-norm 6 --l1-wmax 0.75` (layer-1 drive ×0.5) | Does layer 1 choose more reliably when it waits for more input? | Batch 3's layer-2 lesson applied to layer 1. Halving the weights roughly doubles the pixel spikes integrated before the first crossing: ~1.4× less Poisson noise relative to the difference between features (ignoring leak). w_max is scaled with the row sum so templates learn the same shape and only the drive changes. Watch "windows fired" in the stage-A console (84 % of windows with ink now): faint windows may stop reaching threshold. |
| `ml7_l1norm8` | `--l1-norm 8 --l1-wmax 1.0` (×0.67) | Same, milder dose | |
| `ml7_rd8` | `--l1-readout-inhibition 8` | Is a softer readout than 17.5 mV better? | 17.5 mV was the only layer-1 change that improved every offline reader (same image 0.75 → 0.97, 1-px shift 0.39 → 0.55, k-means 80.7 → 84.1), but it gave only +0.3 / +1.2 points at this layer-2 setting (+3.5 at drive 0.6). Only one value has been tried. |
| `ml7_rd50` | `--l1-readout-inhibition 50` | … or a harder one? | Together with rd8, shows which side of 17.5 mV the best value is on. |
| `ml7_l1tp0.03` | `--l1-theta-plus 0.03` (conscience ÷3) | Do fewer, more distinct features give more stable races? | The conscience equalises use, which is what makes siblings near-copies with equal drive; 3× stronger made the code worse offline (k-means 80.7 → 77.5). The opposite direction is untested. Risk: fewer effective features per window lose information, as F = 8 / 4 did. |
| `ml7_stride2` | `--stride 2` (11×11 = 121 windows, 1,936 neurons) | Does more overlap average out the flips? | No overlap (stride 7) cost ~16 points in batch 1 (52 vs 68 %). With stride 2 every stroke is seen by more windows, so one window's flip is diluted by its neighbours. Layer 1 is ~2× larger: 8-h limit. |

## Not in this batch, and why
- F = 4 / 8 / 32, a stronger conscience, a 4× longer stage A, ±2 px shift augmentation of layer-1 training: the offline
  tests in `../../layer1_analysis/README.md` (section 9) found that none of them improves the code.
- New learning rules aimed at the near-copies (a temporal trace rule that learns shift invariance; lateral decorrelation
  between siblings): the next step if none of these knobs moves the numbers.
- A local mid-level layer: deferred.

## How to read the results
- Held-out and shifted accuracy against the baselines; prefer the epoch-6 average; ±1–2 points between runs is noise.
- Layer 1 alone (teal curve in F1; `run.json` → `l1.readout_accuracy_at_freeze`) reads the layer-1 code directly for
  jobs 1, 2, 5 and 6 (for 3 and 4 it is the baseline's, because the readout inhibition only applies after the freeze).
- Stage A (console, F15): windows fired, first-spike latency, features per active window, structured templates.
- Anything that clears the bar gets a second seed and combinations in batch 6.

## Run and summarise
```bash
bash unity/submit_multi_layer.sh smoke     # optional: one short job (stride 2) on tiny settings -> results/smoke_tests/
bash unity/submit_multi_layer.sh           # the six jobs
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2     # summary.csv, S1, S2 in results_2/
```

## Results
(to be filled in when the runs finish)
