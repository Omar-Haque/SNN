# 03 — batch 3: layer 2 trained on the sparse layer-1 code (submitted from `unity/submit_three_layer.sh`, 2026-11-26)

`unity/submit_three_layer.sh` now holds a later batch; this batch's version: `git show 22255be:unity/submit_three_layer.sh`.

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
python three_layer_architecture/describe_runs.py three_layer_architecture/results/03_batch3_sparse_trained_layer2_6ep_20261126
```

## Results (Unity, 2026-11-26/27)

Each run folder has its own README.md with every setting, layer by layer (`../../describe_runs.py`).

Mean of the snapshots of epochs 4–6 (held-out / shifted ±2 px); "layer-3 input" = spikes per ms per presentation into
layer 3 (both pathways with `--l3-skip`); "dead" = layer-2 neurons that never fired on the 1,000 test digits;
"8 mV-trained" = the same configuration in batch 1 or 2, where layer 2 learned on the 8 mV layer-1 code. All 13 jobs ran in
2.7–3.7 h on different nodes; none failed.

| job | epochs 4–6 | gap | best | layer 2 alone | layer-3 input /ms | layer-3 lines | dead layer 2 | 8 mV-trained |
|---|---|---|---|---|---|---|---|---|
| *references:* single layer · two layers s0 / s1 | *85.6 / 60.6 · 85.3 / 56.5 · 87.3 / 57.4* | | | | | | | |
| `tl3_a2r8_s1` / `_s2` (batch 2, seed 0: 84.2 / 56.1) | 83.5 / 56.6 · 83.3 / 56.9 | 26.9 · 26.4 | 85.1 · 86.0 | 64.9 · 62.2 | 1.61 · 1.64 | 400 | 10 · 12 | 80.3 |
| `tl3_l1r200_r8` (layer 1 at 200 mV throughout) | 81.9 / **58.3** | 23.6 | 84.7 | 64.5 | 1.15 | 400 | 8 | – |
| `tl3_a2inh17_r8` | 84.2 / 56.8 | 27.4 | 86.9 | 64.7 | 1.63 | 400 | 15 | – |
| `tl3_a2r4` | 84.5 / 56.7 | 27.8 | 86.2 | 64.5 | 2.36 | 400 | 4 | 81.9 |
| `tl3_a2r8_b3s1` | 84.4 / 56.4 | 28.0 | 86.1 | 65.5 | 5.40 | 1,296 | 18 | 82.7 · 82.1 |
| `tl3_a2r4_b3s1` | **86.0 / 56.6** | 29.4 | 88.3 | 65.5 | 7.71 | 1,296 | 4 | 83.2 |
| `tl3_a2r8_b2s1` | 85.1 / 57.9 | 27.2 | 87.3 | 68.5 | 6.13 | 1,600 | 83 / 1,600 | 83.0 |
| `tl3_a2r8_skip` / `_s1` | **86.1 / 56.7 · 85.2 / 57.6** | 29.4 · 27.7 | 88.5 · 86.6 | 64.5 · 64.9 | 5.02 · 4.89 | 2,336 | 12 · 10 | 84.0 |
| `tl3_a2r8_b3s1_skip` | 85.3 / 56.5 | 28.9 | 87.2 | 65.5 | 8.76 | 3,232 | 18 | 83.8 |
| `tl3_a2r8_f32` | 81.2 / 55.9 | 25.3 | 83.6 | 67.9 | 1.82 | 800 | 97 / 800 | 79.9 |
| `tl3_a2r8_l2lr5` | 83.8 / 56.6 | 27.2 | 87.2 | 66.3 | 1.92 | 400 | 28 | – |

- **It replicates, a little lower: 83.7 % over three seeds** (84.2, 83.5, 83.3; shifted 56.1–56.9). That is still under the
  single layer (85.6) and the two-layer network (85.3 / 87.3).
- **Why it works: both halves contribute, about equally.** Same blocks and the same 8 mV layer-2 readout; only the layer-1
  code changes:

  | layer 2 learns on | layer 1 read at in stage B | accuracy | layer-3 input /ms |
  |---|---|---|---|
  | the 8 mV code (batch 1) | 8 mV | 80.3 | 0.99 |
  | the 200 mV code | 200 mV (`tl3_l1r200_r8`) | 81.9 | 1.15 |
  | the 17.5 mV code | 8 mV (`tl3_a2inh17_r8`) | 84.2 | 1.63 |
  | the 200 mV code | 8 mV (A2R8, three seeds) | 83.7 | 1.62 |

  Learning on the sparse code is worth ~1.6 points at almost the same input to layer 3. Reading that layer 2 with the richer
  8 mV code in stage B adds ~1.8 more: layer 2 now gets more input than it was trained on, so it sends layer 3 ~40 % more
  spikes. A 17.5 mV code (~2 features per window) is already sparse enough, so the cut-off lies between 8 and 17.5 mV.
- **The gain carries over to every configuration.** In all 8 matched pairs (last column) the sparse-trained layer 2 beats the
  8 mV-trained one by 1.3–3.4 points: 3×3/2 at 8 mV +3.4, at 4 mV +2.6; 3×3/1 at 8 mV +2.0, at 4 mV +2.8; 2×2/1 +2.1;
  F = 32 +1.3; skip +1.6 (mean of 2 seeds vs 1); skip with 3×3/1 +1.5. It also lifts the accuracy-vs-input curve by ~2 points:
  2.4 spikes/ms gives 84.5 here, where batch 2's best 8 mV-trained run needed 4.6 for 83.2.
- **Best without a skip connection: `tl3_a2r4_b3s1`, 86.0 %** (overlapping blocks + 4 mV layer-2 readout; best snapshot
  88.3; F1 is flat at ~86 from ~150k images on). It is the first three-layer run above the single layer (85.6) and the
  same-seed two-layer network (85.3), and level with the two-layer mean (86.3). One seed: the A2R8 seeds spread 0.9 points
  and the two-layer seeds 2.0, so it needs a replicate before it counts.
- **Skip connection: 86.1 / 85.2 %, still about the two-layer accuracy** (same seeds 85.3 / 87.3; batch-2 skip 85.1 / 84.7).
  Layer 3's weight share on layer-2 lines rose this time (27 % from 17 %; 56 % from 40 % with overlapping blocks), but in
  all seven skip runs so far that share simply follows layer 2's share of layer 3's input spikes, a few points below it:

  | run | layer 2's share of the input spikes | of the weight |
  |---|---|---|
  | batch 2 skip, seeds 0 / 1 | 15 % | 13 % |
  | batch 2 skip, 8 mV layer-2 readout | 23 % | 21 % |
  | batch 2 skip, 3×3/1, 8 mV | 48 % | 46 % |
  | batch 3 skip, seeds 0 / 1 | 33 % | 27 % |
  | batch 3 skip, 3×3/1 | 62 % | 56 % |

  STDP with row normalisation hands out weight roughly in proportion to how often each line fires, so the weight share cannot
  tell whether layer 3 relies on layer 2. (Batch 2's "layer 3 ignores layer 2" was the same pattern; a note is added there.)
  The direct test is an ablation: score a trained skip network with its layer-2 lines silenced, then with its layer-1 lines
  silenced.
- **More features per block still do not learn.** F = 32 on the sparse code leaves fewer dead neurons (97 / 800, batch 2
  143–201), but 273 of the 800 rows had not moved from their initial weights when layer 2 froze (F = 16: 32 / 400). The 8 mV
  readout still makes many of those unlearned neurons fire, which likely adds noise: 81.2 %.
- **Faster layer-2 learning (lr ×5) changes nothing** (83.8 %); layer 2 stops moving at ~15k images instead of 30k.
- **No shift gain anywhere** (shifted 55.9–58.3, single layer 60.6). Keeping layer 1 at 200 mV gives the smallest gap (23.6;
  batch 2's other 200 mV-read run 23.0), but at lower centred accuracy.
- **F6:** the sparse-trained layer-2 neurons learn the same kinds of stroke pieces, arcs and loops as batch 1's, just as
  sharp. The difference is that nearly all of them fire, and more often. In F6's edge block (2,0), 5 of batch 1's 16 neurons
  fired 6 times or fewer on the test set; in `tl3_a2r8_s1` the quietest fires 14 times and the next 33, and typical counts are
  ~1.6× higher.
