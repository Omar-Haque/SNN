# layer2_analysis — batch 2 (2026-11-11) and why layer 2 reads its input badly

One figure: `figs/B2_summary.png`. Everything here is torch-free (numpy, scipy, matplotlib) and uses `../layer1_analysis/l1lib.py`.

## Batch 2 on Unity (6 epochs, layer-2 θ decay τ = 30k, tie-break in both layers, seed 0)

| run | held-out (mean of last 3 snapshots) | best snapshot | shifted ±2 px | predicted beforehand |
|---|---|---|---|---|
| single layer (`ml6_ref`) | 85.9 | 87.7 | 60.8 | 88–89 |
| 16 features, control (`ml6_wta_s0`) | 65.1 | 69.7 | 44.0 | low 70s — wrong |
| 16 features, 17.5 mV readout (`ml6_rd17`) | 70.0 | 73.8 | 45.7 | best two-layer run — right |
| 8 features (`ml6_f8`) | 61.3 | 64.1 | 39.6 | no better than 16 — right |
| 8 features, 17.5 mV readout (`ml6_f8_rd17`) | 65.0 | 67.5 | 42.2 | none |
| 4 features (`ml6_f4`) | 55.1 | 56.8 | 35.6 | worse — right |

- The order is the order of the layer-1 code scores from `layer1_analysis` (idealised learner: 4 features 76.0 < 8 features
  79.6 < 16 features 80.7 < 17.5 mV readout 84.1). Coarser features hurt; the soft readout is worth +4 to +5 points.
- Three times more training bought nothing: the control is flat from ~180k examples at 65–67 %, where the 2-epoch runs
  without θ decay ended at 67–69 %. The gap to the single layer is not training time.
- No two-layer run is more shift-tolerant than the single layer (36–46 % vs 61 %).

## The numpy replica of stage B (`l2lib.py`, `train_l2.py`)

Layer 1 frozen from a finished run, layer 2 trained with the same equations. Against Unity, same configuration:

| | examples | accuracy numpy / Unity | rows still at their initial weights numpy / Unity | median θ numpy / Unity |
|---|---|---|---|---|
| two-layer control | 12.8k | 39.7 / 39.0 | 400 / 400 | −36.4 / −36.1 |
| | 32k | 53.2 / 56.2 | 390 / 388 | −28.5 / −28.7 |
| | 60k | 60.6 / 58.5 | 302 / 314 | −23.4 / −23.8 |
| single layer | 6.4k | 68.7 / 68.7 | 167 / 167 | −41.8 / −42.0 |
| | 32k | 81.1 / 80.0 | 0 / 0 | −18.5 / −18.1 |

## Finding 1 — layer 2's templates are better than the network's answers (`static_l2.py`, panel b)

The trained Unity weights, replayed on 2,000 training images (labels) and 1,000 test images, with the winner chosen four ways:

| | ideal prototypes on the same input (k-means, from `layer1_analysis`) | the learned templates, best cosine match wins | + thresholds, no spike-timing noise ("race") | the spiking network (replay) |
|---|---|---|---|---|
| single layer | 90.9 | 87.8 | 85.6 | 84.5 |
| 16 features, control | 82.0 | 76.1 | 70.7 | 65.5 |
| 16 features, 17.5 mV readout | 84.1 | 80.6 | 76.0 | 67.7 |

The single layer loses ~3 points between "best-matching template" and "what the spiking network answers"; the two-layer
networks lose 10–13. The templates layer 2 learned are worth 76–81 %; the spiking race reads them out at 66–68 %.

## Finding 2 — layer 2 decides on its first ~12 input spikes (`probe_first.py`)

At the moment the winning layer-2 neuron fires its first spike (median over test images, trained networks):

| | time | input spikes received | channels that have reported / active over the whole presentation | a learned weight | distance rest → threshold |
|---|---|---|---|---|---|
| two-layer control | 30 ms | 12 | 12 of 38 windows (32 %) | ~10 mV | 70 mV |
| single layer | 17 ms | 163 | 89 of 137 pixels (65 %) | ~0.5 mV | 61 mV |

The calibration (`--l2-drive 1.2`) gives layer 2 the same *mean* drive as the pixel network. But layer 1 sends ~314 spikes per
image where the pixels send ~3,350, so each spike has to be ten times larger: a learned weight is ~10 mV and seven to
twelve of them reach threshold. The winner is chosen when a third of the windows have fired once — whichever third is
fastest on this presentation — and then holds the presentation.

## Finding 3 — making layer 2 wait helps without retraining (`readout_scale.py`, panel c)

The trained control network with layer-2's input weights multiplied by a factor at test time only (thresholds unchanged):

| factor | accuracy | winner's first spike after | windows reported by then | silent test images |
|---|---|---|---|---|
| 1 | 67.3 | 12 input spikes | 32 % | 0 |
| 0.5 | 71.3 | 26 | 66 % | 0 |
| 0.3 | 73.0 | 49 | 83 % | 14 |
| 0.2 | 64.1 | 87 | 91 % | 112 |
| 0.12 | 32.8 | 200 | 98 % | 530 |

+4 to +6 points from the same weights. Below ×0.3 the thresholds (learned at full scale) are out of reach and images go
silent; a network *trained* at a lower scale would set its own thresholds.

## Finding 4 — layer 2 learns slowly (less certain about the cause)

At 32k examples the single layer has no row left at its initial weights; the control has 388 of 400 (314 at 58k, 67 at
122k, 2 at 179k), so its 56 % at 32k comes from essentially random weights plus the threshold conscience (Unity logs). In the
numpy replica, the share of a neuron's wins that go to its commonest class is 0.43 at 3k examples, 0.52 at 13k, 0.62 at 32k
and 0.69 at 60k, against 0.64, 0.81 and 0.86 at 3k, 13k and 32k for the single layer. Likely cause (not tested): one won presentation changes a layer-2 row by
~3 % of the mean weight on the ~38 active inputs, against ~11 % on ~100 pixels in the single layer, because a layer-1
neuron fires at ~28 Hz and a pixel at up to 100 Hz (the presynaptic trace is 3–4 times smaller), and the next image of
the class shares few of those inputs. A partial numpy run at ×0.3 drive with the unchanged learning rate was slower
still (48.6 % vs 53.2 % at 32k: 6 instead of 15 winner spikes per presentation, so each imprint is smaller) — which is
why batch 3 pairs the lower drive with a higher layer-2 learning rate.

## Batch 3 (`unity/submit_multi_layer.sh`, same regime as batch 2, whose control and reference are the baselines)

| job | flags added to the control | question |
|---|---|---|
| `ml7_drv0.6` | `--l2-drive 0.6` | does deciding on ~2/3 of the code help when trained that way? |
| `ml7_drv0.36` | `--l2-drive 0.36` | ... on ~5/6 of the code? (risk: slower learning) |
| `ml7_lr3` | `--l2-lr-mult 3` | is the slow start a learning-rate problem? |
| `ml7_drv0.6_lr6` | `--l2-drive 0.6 --l2-lr-mult 6` | both, with the learning rate raised to keep the imprint per win |
| `ml7_drv0.36_lr10` | `--l2-drive 0.36 --l2-lr-mult 10` | same, further |
| `ml7_rd17_drv0.6_lr6` | `--l1-readout-inhibition 17.5 --l2-drive 0.6 --l2-lr-mult 6` | on the best code so far |

Ceiling to keep in mind: an ideal prototype learner on this layer 1 gets 82–84 %, below the single layer's 86–87 %. Fixing
layer 2 can close most of the gap; beating the single layer needs a better layer-1 code.

## Scripts

| script | what it does |
|---|---|
| `l2lib.py` | numpy two-layer network with layer-2 learning (sparse updates; ~70 images/s on one core) |
| `train_l2.py --run ../results/<run> --tag T [--drive-mult --lr-plus --lr-minus --theta-plus --init ...]` | stage B replica with checkpoints → `cache/l2_T.npz` |
| `static_l2.py ../results/<run>` | the four readers on a run's trained weights → `cache/static_*.json` |
| `probe_first.py ../results/<run>` | input received when layer 2 decides → `cache/probe_*.json` |
| `readout_scale.py ../results/<run> 1,0.5,0.3,0.2,0.12` | test-time weight scaling → `cache/readout_scale_*.json` |
| `figures_batch2.py` | `figs/B2_summary.png` |
