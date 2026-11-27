# 03 — batch 3: layer 2 trained on the sparse layer-1 code (submitted from `unity/submit_three_layer.sh`, 2026-11-26)

## Purpose
Batch 2 (`../02_batch2_more_channels_6ep_20261125/`) found one change that stood out: layer 2 **learning on layer 1's
one-feature-per-window code** (layer 1 at 200 mV during stage A2, `--l1-a2-inhibition 200`) and then being **read out at
8 mV** reached 84.2 % with only 400 layer-2 neurons and 1.65 spikes/ms into layer 3 — while the many-channel runs needed 2–3×
the input and plateaued at ~83 %. One seed. The skip connection gave 85.1 / 84.7 %, but layer 3 ignored layer 2 there.

This batch: (1) replicates it, (2) asks why it works, (3) combines it with batch 2's other levers.

On (2): in the batch-2 run, layer 2 learned on layer 1 at 200 mV, but in stage B layer 1 was read at 8 mV, so layer 2 received
more input than it learned on (~1,065 instead of ~657 layer-1 spikes per image). The gain could come from what layer 2 learned
(features fitted to a sparse, sharp code) or from that switch (extra input at readout). `tl3_l1r200_r8` keeps layer 1 at
200 mV throughout — no switch; `tl3_a2inh17_r8` trains layer 2 on an intermediate code (17.5 mV, ~2 features per window).

## References (mean of the snapshots of epochs 4–6, held-out / shifted ±2 px)
| network | seed | accuracy |
|---|---|---|
| single layer | 0 | 85.6 / 60.6 |
| two layers, stride 2 + 8 mV | 0 / 1 | 85.3 / 56.5 · 87.3 / 57.4 |
| three layers, batch 2: layer 2 trained at 200 mV, read at 8 mV | 0 | 84.2 / 56.1 |
| three layers, batch 2: same training, read at 200 mV | 0 | 77.4 / 54.4 |
| three layers, batch 2: layer 2 trained and read at 8 mV | 0 | 80.3 / 53.3 |
| three layers, batch 2: skip connection | 0 / 1 | 85.1 / 56.1 · 84.7 / 56.0 |
| three layers, batch 2: blocks at stride 1, 4 mV | 0 | 83.2 / 54.6 |

## Configuration
Every job: `three_layer.py`'s defaults plus the flags below ("A2R8" = `--l1-a2-inhibition 200 --l2-readout-inhibition 8`).
10-h limit; the slow node `gypsum-gpu018` is now excluded in `unity/run.sbatch`.

| job | flags | question |
|---|---|---|
| `tl3_a2r8_s1` / `tl3_a2r8_s2` | A2R8, `--seed 1` / `--seed 2` | Is 84.2 % real? (three seeds with batch 2's) |
| `tl3_l1r200_r8` | `--l1-readout-inhibition 200 --l2-readout-inhibition 8` | Layer 1 at 200 mV throughout: is the gain the features, or the switch to 8 mV in stage B? |
| `tl3_a2inh17_r8` | `--l1-a2-inhibition 17.5 --l2-readout-inhibition 8` | How sparse does layer 2's training code need to be? (200 → 84.2, 8 → 80.3) |
| `tl3_a2r4` | `--l1-a2-inhibition 200 --l2-readout-inhibition 4` | softer layer-2 readout |
| `tl3_a2r8_b3s1` | A2R8, `--l2-stride 1` | + overlapping blocks (81 × 16 = 1,296 neurons) |
| `tl3_a2r4_b3s1` | `--l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-stride 1` | + both |
| `tl3_a2r8_b2s1` | A2R8, `--l2-block 2 --l2-stride 1` | + smaller blocks (100 × 16 = 1,600) |
| `tl3_a2r8_skip` / `_s1` | A2R8, `--l3-skip` | + skip connection: does layer 3 use layer 2 now? (`l3_weight_share_l2`, start 17 %) |
| `tl3_a2r8_b3s1_skip` | A2R8, `--l2-stride 1 --l3-skip` | (start 40 %) |
| `tl3_a2r8_f32` | A2R8, `--l2-features 32` | On a sparser code, do the extra features stay alive? (batch 2: 201 / 800 dead) |
| `tl3_a2r8_l2lr5` | A2R8, `--l2-lr-mult 5` | faster layer-2 learning |

## What to watch
- The three seeds of A2R8: does it hold at ~84 %?
- `tl3_l1r200_r8` against `tl3_a2r8_*`: if it is as good, the features are what matters; if it falls to ~77 %, the switch
  to the 8 mV layer-1 code in stage B is.
- Layer-3 input rate (`l3_input_rate_per_ms`) against accuracy — batch 2's A2R8 run was well above the channel curve.
- Skip runs: accuracy and `l3_weight_share_l2` — above the start value means layer 3 found the parts useful.
- F6 (what layer-2 neurons fire for) for the 200 mV-trained runs vs batch 2's 8 mV-trained ones.

## Run and summarise
```bash
bash unity/submit_three_layer.sh smoke     # first, on your own computer: all 13 configurations on tiny settings (~4 min)
bash unity/submit_three_layer.sh           # the 13 jobs on Unity
python three_layer_architecture/summarize_three_layer.py --results three_layer_architecture/results
```

## Results
(to be filled in when the runs finish)
