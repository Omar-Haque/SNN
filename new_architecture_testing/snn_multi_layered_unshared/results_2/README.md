# results_2 — batch 4 (submitted from `unity/submit_multi_layer.sh`, 2026-11-20)

Runs made with the code from 2026-11-15 on, which also scores layer 1 during stage A. In these runs F1 starts at image 0
(teal: layer 1 as the output during stage A; blue: layer 2 after the freeze), F1–F7 use total training images on the
x-axis, and a green dashed line marks the end of stage A. Earlier runs stay in `../results/`.

## Purpose
Batch 3 found the two layer-2 settings that matter: a lower drive (layer 2 waits for more of the layer-1 code before it
fires) and a higher learning rate (it imprints enough per win). Its best run, `--l2-drive 0.36 --l2-lr-mult 10`, reached
82.3 % against the single layer's 85.9 % — one seed, still improving when it stopped. Batch 4 checks that result and
builds on it.

## Baselines (not rerun)
| run | held-out | shifted ±2 px | folder |
|---|---|---|---|
| single layer | 85.9 % | 60.8 % | `../results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |
| two layers, drive 0.36, lr ×10 | 82.3 % | 57.7 % | `../results/04_batch3_layer2_drive_lr_6ep_20261113/…_drv0.36_lr10_…` |
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
bash unity/submit_multi_layer.sh smoke     # once: checks the new code on tiny settings (results/smoke_tests/)
bash unity/submit_multi_layer.sh           # the six jobs
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2     # summary.csv, S1, S2 in this folder
```

## Results
(to be filled in when the runs finish)
