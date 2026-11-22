# 05 — batch 4: following up batch 3's best layer-2 setting (submitted from `unity/submit_multi_layer.sh`, 2026-11-20)

Runs made with the code from 2026-11-15 on, which also scores layer 1 during stage A. In these runs F1 starts at image 0
(teal: layer 1 as the output during stage A; blue: layer 2 after the freeze), F1–F7 use total training images on the
x-axis, and a green dashed line marks the end of stage A. Earlier runs stay in `../../results/`.

## Purpose
Batch 3 found the two layer-2 settings that matter: a lower drive (layer 2 waits for more of the layer-1 code before it
fires) and a higher learning rate (it imprints enough per win). Its best run, `--l2-drive 0.36 --l2-lr-mult 10`, reached
82.3 % against the single layer's 85.9 % — one seed, still improving when it stopped. Batch 4 checks that result and
builds on it.

## Baselines (not rerun)
| run | held-out | shifted ±2 px | folder |
|---|---|---|---|
| single layer | 85.9 % | 60.8 % | `../../results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |
| two layers, drive 0.36, lr ×10 | 82.3 % | 57.7 % | `../../results/04_batch3_layer2_drive_lr_6ep_20261113/…_drv0.36_lr10_…` |
| two layers, drive 0.6, lr ×6, 17.5 mV readout | 80.8 % | 55.4 % | same batch-3 folder, `…_rd17.5_…_drv0.6_lr6_…` |

## Configuration
Common to every job: `--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200`, layer 1 with the
defaults (7×7 windows, stride 3, F = 16, stage A up to 30,000 images). "best" = `--l2-drive 0.36 --l2-lr-mult 10`.

| job | flags | question | how to read it |
|---|---|---|---|
| `ml7_best_s1` | best, `--seed 1` | Is 82.3 % real? | Each number is on 1,000 test images (±1.2 points); a second seed shows how much runs differ. |
| `ml7_best_rd17` | best, `--l1-readout-inhibition 17.5` | Do the two improvements add up? | The 17.5 mV readout added 3.5 points at drive 0.6 / lr ×6; compare with 82.3 %. |
| `ml7_best_rd17_s1` | same, `--seed 1` | … on a second seed | |
| `ml7_best_ep12` | best, `--epochs 12` | Is the remaining gap training time? | Reaching 85–86 % → the two-layer net is slower, not worse; stalling near 83 % → layer 1's code is the limit. |
| `ml7_drv0.24_lr15` | `--l2-drive 0.24 --l2-lr-mult 15` | Does the drive / learning-rate trend keep going? | 65 → 77 → 82 % so far (1.2/×1, 0.6/×6, 0.36/×10). Watch layer-2 spikes per presentation (~14 in the best run). |
| `ml7_best_pool2` | best, `--pool2` | Can two layers beat one on shifted digits? | Both lose ~25 points on ±2 px shifts today; 2×2 pooling should shrink that drop. |

Also new in every run: layer 1's own accuracy during stage A (F1, console, `run.json` → `l1.readout_accuracy_at_freeze`).
Near layer 2's final number → layer 2 mostly relabels layer-1 neurons; far below → layer 2 does real work.

## Run and summarise
```bash
git show 2ed80eb:unity/submit_multi_layer.sh   # the script now holds batch 5; this is batch 4's version
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2     # summary.csv, S1, S2 in results_2/
```

## Results (Unity, 2026-11-20/21)

"final" = mean of the last 3 snapshots (the summary.csv value); "epoch 6" = mean of all snapshots in the 6th epoch (less noisy).
"layer 1 alone" = layer 1's 1,024 neurons labelled and scored at the freeze (identical for runs with the same seed: same stage A).

| run | final | epoch 6 | best snapshot | shifted ±2 px | shift drop | layer 1 alone (shifted) |
|---|---|---|---|---|---|---|
| single layer (batch 2) | 85.9 | 85.9 | 87.7 | 60.8 | 25.0 | – |
| best, seed 0 (batch 3) | 82.3 | 83.0 | 84.8 | 57.7 | 24.6 | – |
| best, seed 1 | 82.4 | 82.0 | 83.2 | 58.4 | 24.0 | 65.1 (46.6) |
| best + 17.5 mV readout, seed 0 | 82.6 | 82.7 | 84.3 | 59.6 | 22.9 | 67.3 (48.1) |
| best + 17.5 mV readout, seed 1 | 83.6 | 84.3 | 85.6 | 59.8 | 23.8 | 65.1 (46.6) |
| best, 12 epochs | 83.0 | 82.7 (epochs 7–12: 83.4) | 86.2 | 59.3 | 23.7 | 67.3 (48.1) |
| drive 0.24, lr ×15 | 83.8 | 83.0 | 84.5 | 59.6 | 24.2 | 67.3 (48.1) |
| best + pool2 | 77.2 | 78.5 | 80.6 | 52.8 | 24.3 | 67.3 (48.1) |

- **The best setting replicates**: 82.3 / 82.4 on two seeds (epoch-6 averages 83.0 / 82.0 / 82.7 counting the 12-epoch run).
- **The 17.5 mV readout adds little at this drive**: +0.3 and +1.2 points on the two seeds (+3.5 at drive 0.6 / lr ×6 in
  batch 3), shifted +1–2 points. Consistent with it having helped mostly by giving layer 2 more input before it decides —
  which the lower drive already does.
- **More training is not the missing piece**: doubling to 12 epochs moves the epoch average from 82.7 to ~83.4, flat from
  epoch 9 on; the single layer is flat at ~85.5–86 from epoch 3.
- **The drive / learning-rate trend has saturated**: 0.24 / ×15 matches 0.36 / ×10 (epoch 6: 83.0 vs 83.0), with fewer
  layer-2 spikes per presentation (10.3 vs 14.3) and 57 rows that never left their initial weights.
- **Pooling hurts and buys no shift tolerance**: −5 points held-out and shifted, same ~24-point drop.
- **Layer 1 alone scores 65–67 %** (plateau by ~13k stage-A images); layer 2 adds ~16 points, so it does real work.
- Every layer-2 variant now lands at 82.4–83.8 %, 2–3.5 points under the single layer, and every two-layer run loses
  23–24 points on shifted digits (single layer 25). What is left points at layer 1's code, which was measured earlier to be
  less stable than raw pixels (same image twice: cosine 0.75 vs 0.97; 1-px shift: 0.40 vs 0.75; layer1_analysis/README.md).
