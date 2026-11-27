# three_layer_architecture — a three-layer STDP network

```
pixels ──► LAYER 1: local strokes ──► LAYER 2: local mid-level features ──► LAYER 3: 400-neuron D&C classifier
           7×7 windows, stride 2        3×3 blocks of layer-1 windows          all-to-all on layer 2's output
           121 windows × 16 = 1,936     25 blocks × 16 = 400 neurons           labelled and scored like the single layer
```

## Why
The two-layer study (`../new_architecture_testing/snn_multi_layered_unshared/`, `results_2/`) ended with:
- **Centred digits:** layer 1 at stride 2 read out at 8 mV, under the D&C layer (drive 0.36, lr ×10), reaches the single
  layer — 85.3 / 87.3 % on two seeds vs 85.6 % (mean of the epochs-4–6 snapshots).
- **Shifted digits (±2 px):** every two-layer variant stays at 56–59 % (single layer 60.6 %). The loss did not follow the
  layer-1 code's shift stability, and 2×2 pooling did not help even on a pixel-stable code. It sits in what the top layer
  learns: whole-digit templates, built from strokes, that a shift moves off.

A layer whose neurons see a *part* of the digit — a few neighbouring layer-1 windows — and compete only with each other
learns stroke combinations (corners, junctions, pieces of loops) for the same reason layer 1 learns strokes instead of
digits: a small receptive field and local competition. That is layer 2 here. Layer 3 builds digit templates from those
parts. The first question is whether the extra layer keeps the centred-digit accuracy; the second is whether parts make
the top layer more tolerant of shifts.

## The layers (defaults; every value is a flag of `three_layer.py`)
**Layer 1** — the two-layer study's `FeatureLayer`, imported unchanged: 7×7 windows at stride 2 (11×11 = 121 windows), 16
features each with its own template and threshold, row sum 12, w_max 1.5, θ⁺ 0.1 (decay 2,000 examples), 200 mV competition
plus the tie-break inside a window while learning. After stage A it is frozen and read out at **8 mV**
(`--l1-readout-inhibition`), so the features that lose a window's race keep firing (~1,065 spikes per image).

**Layer 2** (`BlockLayer`, new) — the same neuron model, conscience, STDP and row normalisation as layer 1, applied to layer
1's output instead of pixels. A block is `--l2-block` × `--l2-block` neighbouring layer-1 windows with all 16 feature channels
of each (3×3 → 144 inputs, an 11×11-pixel area); blocks step by `--l2-stride` windows (2 → 5×5 = 25 blocks); each block has
`--l2-features` neurons (16) that compete only with each other (200 mV + tie-break while learning; `--l2-readout-inhibition`
after its freeze, default unchanged). θ⁺ 0.1, decay 2,000 examples, STDP rates 1e-2 / 1e-4 (× `--l2-lr-mult`).
*Weight scale:* layer-1 spikes are much sparser than pixel spikes, so after stage A the frozen layer 1 is run on 160 images
and an active block's input rate is measured; layer 2's mean weight is set so that an active block starts with the drive an
active layer-1 window started with (pixel input × layer 1's initial mean weight 0.245, × `--l2-drive-mult`), and w_max keeps
layer 1's w_max / mean ratio (1.5 / 0.245 = 6.1).

**Layer 3** — the two-layer study's top layer: `DiehlAndCookNetwork`, 400 neurons, all-to-all on layer 2's output, θ⁺ 0.5,
θ decay 30k examples, 200 mV inhibition plus tie-break, weight scale calibrated from layer 2's measured output for
**0.36 mV/ms** (`--l3-drive`), learning rates **×10** (`--l3-lr-mult`).

## Training and scoring
| stage | what learns | length | scored with … as the output |
|---|---|---|---|
| A | layer 1 | up to 30,000 images; stops early when layer 1 is stationary | layer 1 (1,936 neurons) |
| calibration | — | 160 images | — |
| A2 | layer 2 (layer 1 frozen) | up to 30,000 images; stops early when no template leaves its initial weights for 5 snapshots and θ is stationary | layers 1 + 2 (layer 2's neurons) |
| calibration | — | 160 images | — |
| B | layer 3 (layers 1, 2 frozen) | `--epochs` (6) × 60,000 images | the whole network (layer 3) |

Scoring = the estimator of the whole project: label each output neuron with the digit it fires most for (train images
0–1999), then accuracy on test images 0–999 and on the same images shifted by up to ±2 px. Each stage is scored after its
first batch, every `--snapshot-interval` batches (200 = 6,400 images) and at its freeze.

## Files
| file | what it is |
|---|---|
| `three_layer.py` | the network and training; imports layer 1, layer 3, the estimators and stage A from `../new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py` (a change there changes these runs too) |
| `three_layer_plots.py` | the figures of one run (numpy + matplotlib); `python three_layer.py --replot <run dir>` redraws them |
| `summarize_three_layer.py` | every finished run under `results/` → `results/summary.csv`, `results/S1_accuracy_overlay.png` |
| `sta_replay.py` | spike-triggered averages (F6, F7) for runs made before `three_layer.py` saved them (batch 1), in numpy: `python sta_replay.py <run dir> ...` |
| `../unity/submit_three_layer.sh` | the current Unity batch (and `smoke`, and resubmitting jobs by name) |
| `results/<batch folder>/` | one folder per batch with a README.md (configuration, questions, results); one subfolder per run |

Each run folder: `run.json` (configuration, calibrations, summary numbers), `run.npz` (curves, every layer's weights and
thresholds, per-stage diagnostics, spike-triggered averages) and seven figures:
- **F1_accuracy** — accuracy across all three stages (teal: layer 1 as the output, orange: layer 2, blue: layer 3; dashed:
  shifted digits), green dashed lines at the two freezes, gray lines at the single-layer and two-layer references.
- **F2_layer1_templates** — the 16 layer-1 templates at four windows.
- **F3_layer2_features** — layer-2 neurons drawn in pixel space (each weight × the layer-1 template it reads, at that
  template's place). A linear back-projection through overlapping, slightly fuzzy layer-1 templates: it makes the features
  look much blurrier than they are — use F6 to see what they respond to.
- **F4_layer3_templates** — the 20 most active layer-3 neurons drawn in pixel space through layers 2 and 1, with labels.
- **F5_stages** — per stage: unformed layer-1 templates; layer-2 templates still at their initial weights, θ, blocks
  firing, latency, features per active block; layer-3 rows at initial weights, spikes per presentation, θ.
- **F6_layer2_sta** — the same layer-2 neurons as F3, as the average pixel patch each fired for on the 1,000 test digits
  (spike-triggered average): what actually drives the neuron. In batch 1: sharp arcs, corners, bars, loops, junctions.
- **F7_layer3_sta** — the same 20 layer-3 neurons as F4, as the average test digit each fired for (a neuron that fired on
  only a few digits gives a sharp but unrepresentative average; the spike count is in the title).

Figures can be redrawn without torch: `python three_layer_plots.py <run dir> ...`.

Options beyond the layer settings: `--l3-skip` (layer 3 also reads layer 1's output — a skip connection),
`--l1-a2-inhibition` (layer 1's inhibition while layer 2 learns; default = its readout inhibition), `--device auto|cpu|mps|cuda`.

## Running
```bash
# smoke tests on your own computer (no queue): every job of the current batch on tiny settings, pass / fail per job
bash unity/submit_three_layer.sh smoke             # from the repo root, with the project's python environment active
bash unity/submit_three_layer.sh smoke tl_b5s2     # only the jobs named
SMOKE_DEVICE=cpu bash unity/submit_three_layer.sh smoke     # force the CPU (default: MPS on a Mac)
PYTHON=venv/bin/python bash unity/submit_three_layer.sh smoke   # a specific interpreter
cd three_layer_architecture && python three_layer.py --smoke    # or one smoke run by hand (add any flags; --device cpu|mps|cuda)

bash unity/submit_three_layer.sh                   # on Unity: the current batch (smoke-unity = the smoke run as a Unity job)
python three_layer_architecture/summarize_three_layer.py    # after the runs: results/summary.csv and S1
```
A local smoke run trains every stage on 448–640 images at 25 ms per presentation (layers 2 and 3 barely fire that early)
and scores on 64–128, so its accuracies mean nothing; it checks that the whole code path — all three stages, both calibrations, scoring, saving and the five
figures — runs for that configuration. Logs and run folders go to `results/smoke_tests/`.

## Batches
| folder | question | outcome |
|---|---|---|
| `results/01_batch1_first_three_layer_6ep_20261125/` | Does a third layer keep the centred-digit accuracy, and what do the first layer-2 choices (readout, overlap, block size, learning rate) do? | 73–80 % vs two layers' 85–87 %; no shift gain. Layer 3 hears ~20 layer-2 channels per image (two layers: 343) and accuracy follows the channel count (8 mV readout 80.3, stride 1 79.4). Layer-2 features are sharp stroke combinations (F6); the pictures in F3/F4 are blurred by the drawing method. |
| `results/02_batch2_more_channels_6ep_20261125/` | Layer 2's features are good (F6) but layer 3 hears ~20 channels per image: what gives it more (softer readout, overlap, more features, smaller blocks, skip connection), and does training layer 2 on the sharper layer-1 code help? 19 jobs | Best 84.2 % without skip (layer 2 trained on the 200 mV layer-1 code, read at 8 mV) and 85.1 / 84.7 % with a skip connection (≈ two layers; layer 3 ignores layer 2). More channels plateaus at ~83 %; more features per block mostly die. No shift gain. |
| `results/03_batch3_sparse_trained_layer2_6ep_20261126/` | Does batch 2's best (layer 2 trained on the 200 mV layer-1 code, read at 8 mV) replicate, why does it work, and does it combine with more channels or the skip connection? 13 jobs | pending |
