# results_for_cnn_replication — can STDP layers learn parts, as a CNN's upper layers do?

## The aim
In every network so far, the top layer learns **whole-digit templates**: the two-layer network's layer 2 (F13) and the
three-layer network's layer 3 (F4, F7) each look like a prototypical digit. The layers below them (layer 1's strokes,
layer 2's stroke combinations in the three-layer network) are local features, like the early layers of a CNN — but a CNN's
upper layers keep learning *parts* (loops, junctions, corners) that are shared across classes, and only its final linear
readout is class-specific.

Why our top layer learns prototypes:
1. **It sees the whole image.** Each top-layer neuron is connected to everything below, so nothing limits it to a part.
2. **One neuron wins each image.** All top-layer neurons inhibit each other at 200 mV, so usually only one fires per image,
   and STDP pulls the winner's weights toward the whole of that image. A neuron becomes the average of the images it won —
   a cluster centre, i.e. a prototype. That is how the Diehl & Cook network classifies, and the label readout depends on it.

The second point is the one we can change without supervision. **The question for this folder: if several top-layer
neurons share each image, do they learn parts instead of prototypes — and can the network still classify?**

## Why the two-layer network first
- Its layer 2 has the same problem (a whole-image layer with one winner per image).
- It has a replicated baseline: stride 2, layer 1 read at 8 mV, drive 0.36, lr ×10 gives 85.3 / 87.3 % (seeds 0 / 1,
  `../results_2/07_batch6_stride2_readout8_6ep_20261122/`). The three-layer network is not tuned yet, so changes there would
  be tangled up with other settings.
- It is cheaper (~2.4 h per run instead of ~3).
- Whatever works carries straight over to the three-layer network's layer 3.

## How we judge a run
- **Parts or prototypes?** F16 (new): the average test image each layer-2 neuron fired for. Prototypes look like digits;
  parts show a stroke, loop or corner. Backed by two numbers in run.json: `class_purity_median` (share of a neuron's spikes
  that come from its top digit: ~1 for a one-digit prototype, lower for a part several digits share) and
  `test_distinct_per_image` (how many different layer-2 neurons fire per test digit; ~1 in the baseline).
- **Can it still classify?** The readout is unchanged: each neuron is labelled with the digit it fires for most (training
  images 0–1,999), and a test digit gets the label whose neurons fire most, all firing neurons voting. Compared with
  85.3 / 87.3 % (and 56.5 / 57.4 % on digits shifted by ±2 px).

If parts form but accuracy falls, that may be the readout rather than the features: a part-neuron labelled with one digit
throws away what it says about the others. The next step would then be a **trained linear readout** — a 10 × 400 weight
table fitted with the labels on the frozen network's spike counts (supervised, unlike STDP and the label vote) — to measure
how much digit information the parts carry.

## Batches
| folder | question | result |
|---|---|---|
| `01_batch1_several_winners_6ep_20261127/` | Several layer-2 winners per image — softer lateral inhibition (50 / 20 / 8 mV) or exactly k winners (k = 2, 3, 5, 10; new `--l2-k`): do the neurons learn parts, and what happens to accuracy? 11 jobs | pending |

Each batch folder's README.md has the configuration, the questions and the results, and every run folder has its own
README.md with all its settings. `summary.csv` here covers every run in this folder:
```bash
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication
```
