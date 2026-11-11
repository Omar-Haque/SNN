# layer1_analysis — what layer 1 of the unshared two-layer network learns, and what it is worth

Tests on the finished Unity runs (2026-11-06) plus a numpy replica of stage A that logs what the torch code does not
(every neuron's lifetime win count). Question: **a third of the layer-1 synapses look like random noise — why, does it
matter, and what would make layer 1 more useful?**

Everything here is torch-free (numpy, matplotlib, scikit-learn for one reader) and runs on a laptop CPU. The numpy replica
of the network was validated against the runs: replaying the trained two-layer network gives 67.2 % vs the run's
66.3 % (single layer: 86.3 % vs 86.8 %), and the replica of stage A matches the torch layer 1 on every metric below
(274 vs 273 never-firing neurons, 9.8 vs 9.9 effective features per window, stability within 0.01, readers within ~1 point).

All numbers are for the newest seed-0 run, `unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133`
(7x7 windows, stride 3, 16 features, tie-break in both layers, stage A ran to its 30k cap), unless stated.

---

## 0. The 2026-11-06 runs were a repeat of batch 1, not batch 2

Unity ran the old `submit_multi_layer.sh` (job names `ml_wta_*`, 2 epochs, no θ decay) with the new code. That gives a
replication — and the run-to-run spread to read every comparison against:

| configuration | 2026-11-02 held-out / shifted | 2026-11-06 held-out / shifted |
|---|---|---|
| single-layer reference | 86.8 / 60.6 | 87.0 / 60.8 |
| two-layer, tie-break both layers, seed 0 | 68.9 / 41.3 | 67.6 / 42.9 |
| same, seed 1 | 67.8 / 43.7 | 66.6 / 42.8 |
| tie-break in layer 2 only | 68.6 / 44.4 | 67.5 / 43.7 |
| 32 features per window | 68.7 / 45.6 | 69.2 / 44.7 |
| pool2 | 65.9 / 41.9 | 64.1 / 40.6 |
| stride 7 | 52.8 / 33.9 | 52.0 / 32.6 |

Same configuration, same seed, different day: 1–2 points. F16, F32 and "no layer-1 tie-break" are indistinguishable.

---

## 1. Where the noise templates are — `figs/A1_noise_map.png`

347 of 1,024 templates are unstructured (1-px autocorrelation < 0.4). They are two different populations:

- **127 in starved windows** — corners and edges that fire on fewer than 20 % of images (hatched in panel d). Too
  little ink ever lands there.
- **220 dead units** — in windows that fire on 20–86 % of images (rings 0–1). Every window that fires on more than
  90 % of images has all 16 templates structured.

How to read A1: (b) is exposure, (c) is learned templates, (d) is noise. (c) and (d) are mirror images of (b) — the
noise follows exposure — but (d) is not only the hatched starved windows: the orange band one ring in is dead units.

## 2. A template forms if and only if its neuron wins — `figs/A2_wins_make_templates.png`

From the numpy replica (true lifetime win counts):

- (a) structure against wins in 30k images: nothing below ~30 wins is structured, essentially everything above ~300 is.
  The transition is sharp.
- (b) a template first becomes structured after a **median of 55 wins** (range ~25–300).
- (c) the four most-winning templates of a corner, a ring-1 and a centre window after 0 / 1.3k / 9.6k / 30k / 120k
  images: centre and ring-1 templates are formed by ~10k images; the corner's are still noise at 120k.

Intuition: one win = one 350-ms imprint at 1/32 strength (batch-mean STDP). It takes a few dozen of them to pull a
template out of its random initialisation. A neuron that never wins never moves.

## 3. More stage-A training does not fix it — `figs/A3_formation_timeline.png`

(a) the torch run by ring: the centre is fully formed by ~2k images, ring 2 by ~5k, ring 1 plateaus at ~88 %, the outer
ring at ~30 %. (b) the replica run 4x longer: 685 → 731 structured at 120k (outer ring 32 → 38 %, ring 1 89 → 95 %), and
the share of neurons that have never won flattens at ~4 %. The noise is not a stage-A length problem.

## 4. Why dead units never win — `figs/A4_dead_units.png`

(a) one ring-0 window that fires on 57 % of images: 6 templates share almost all the wins (26.6 %, 10.3 %, ...), the
other 10 have won 0–0.5 % and carry no threshold increase. (b) For a dead unit, on images where its window fires, its
expected threshold crossing (from its expected drive) comes a median **48 ms after the winner's**, and on 36 % of images
its drive cannot reach threshold at all — it is not narrowly beaten, it is out of the race. (c) The conscience (θ) lifts
the winners' thresholds by **+6 mV**; for a dead unit to cross first, the faster siblings would need to sit at
**~+23 mV**. The conscience is 3–4x too weak to give a random template a turn against a learned one.

The underlying asymmetry: a learned template puts ~0.45 of its weight on the stroke pixels; a random one puts its row
mean (~0.25) on them, so on any stroke the learned sibling gets roughly twice the drive. The conscience can only
equalise neurons that are already roughly equal.

## 5. The noise templates are silent and harmless — `figs/A5_ablation.png`

The trained network, replayed in numpy with layer-1 channels hidden from layer 2 (layer-1 dynamics untouched, labels
re-assigned):

| hidden from layer 2 | accuracy | layer-1 spikes per image layer 2 still sees |
|---|---|---|
| nothing | 67.2 | 314 |
| the 347 unstructured channels | 67.7 | 314 |
| 347 random structured channels | 49.0 | 150 |
| the whole outer ring (448 channels) | 66.9 | 252 |
| everything except the unstructured channels | 8.5 | 0.4 |

The noise channels emit 0.4 of the 314 layer-1 spikes per image and carry 0.6 % of the per-channel class information
(`info.py`). They do not corrupt anything — they are wasted capacity. The outer ring carries 20 % of the per-channel
information but can be removed at no cost: its strokes are also seen by the overlapping ring-1 windows.

**So "a third of layer 1 is noise" is not the reason layer 2 is at 68 %.** The reason is in what the other two thirds
produce.

## 6. The real problem: the code is fragile — `figs/A6_code_stability.png`

Cosine similarity between the layer-1 code of a test digit and the code of...

| | same image, new Poisson draw | same image shifted 1 px | another image, same class | different class | class contrast |
|---|---|---|---|---|---|
| raw pixels | 0.97 | 0.75 | 0.51 | 0.35 | 1.4x |
| layer 1 (as trained) | **0.75** | **0.40** | 0.19 | 0.05 | 3.8x |
| layer 1, 32 features | 0.67 | 0.32 | 0.14 | 0.03 | 4.1x |
| layer 1 read out at 17.5 mV | 0.97 | 0.55 | 0.29 | 0.10 | 3.0x |

The same image shown twice keeps only three quarters of its code; moved by one pixel it keeps 40 %. Most of the
"within-class variability" of the code is not variation between 3s — it is the code changing under perturbations a
human would not notice. Panel (b) shows the trade: layer 1 buys class contrast (3.8x vs 1.4x) with fragility.

## 7. Why it is fragile: every window is a photo-finish — `figs/A7_race.png`

Each active window emits the one feature that crosses threshold first, and that feature then fires alone for the rest of
the presentation (200 mV inhibition). From the expected drive of each feature:

- (a) the expected gap between the two fastest features is a median **3.9 ms**, against a first crossing at ~38 ms;
  33 % of windows are within 2 ms.
- (b) below ~5 ms the winner is decided by the Poisson noise of the first ~40 ms, not by the image: at < 2 ms gap the
  expected-fastest feature wins only 52–72 % of repeats; above 10 ms it wins 99 %.
- (c) the gaps are small because siblings are near-copies: a structured feature's most similar sibling has cosine 0.72
  (median; > 0.5 for 92 %) — the same stroke at a 1-px offset or a slight angle.

This is the conscience doing its job: it equalises usage, so siblings end up with equal drive on their shared strokes,
so races between them are close. It is also why a 1-px shift changes the code: the stroke moves into the neighbouring
sibling's territory. Pixels degrade gracefully under a shift; a one-of-N symbol flips.

## 8. What the code is good for — `figs/A8_readers.png`

Three readers trained on the code, standing in for "what layer 2 could do with it":

| reader | pixels | layer-1 code | layer-1 code at 17.5 mV |
|---|---|---|---|
| nearest class centroid (averages all examples of a class) | 74.6 | **78.8** | 80.5 |
| 1-NN from 1 / 4 / 10 / 40 stored examples per class (single-instance imprints) | 45 / 60 / 71 / 82 | 43 / 54 / 63 / 74 | 47 / 60 / 69 / 80 |
| 400-prototype k-means (an idealised competitive learner) | 90.9 | 82.0 | 84.1 |
| the actual D&C layer 2, 2 epochs | 86.8 | 67.6 | not run yet |

The code helps a reader that averages (flips cancel in a class mean) and hurts every reader that stores instances or
prototypes (a stored instance's symbols rarely recur in the next instance). About 9 of the 20-point gap between the
pixel and the two-layer network is in the code itself (91 → 82 for the idealised learner); the rest is how STDP/D&C
learns on it. k-means varies by ±0.9 points across clustering seeds; differences under ~2 points are noise.

## 9. Interventions on layer 1 — `figs/A9_interventions.png`

All trained with the numpy replica of stage A (30k images unless stated) and scored with the same harness
(`code_eval.py`).

| layer 1 | noise templates | never fire | effective features / window | same image | 1-px shift | same class | contrast | centroid | 1-NN (40/class) | k-means 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| replica of the torch stage A | 33 % | 27 % | 9.8 | 0.75 | 0.39 | 0.19 | 3.7x | 79.4 | 73.6 | 80.7 |
| 4x longer stage A (120k) | 29 % | 25 % | 10.8 | 0.75 | 0.39 | 0.19 | 3.8x | 79.6 | 73.8 | 81.5 |
| 3x stronger conscience (θ⁺ 0.3) | 22 % | 18 % | 12.6 | 0.76 | 0.36 | 0.17 | 4.2x | 78.3 | 71.6 | 77.5 |
| ±2 px shift augmentation | 25 % | 25 % | 9.0 | 0.76 | 0.39 | 0.19 | 3.3x | 80.6 | 73.1 | 77.6 |
| 8 features per window | 23 % | 17 % | 6.6 | 0.82 | 0.49 | 0.26 | 3.1x | 76.1 | 73.4 | 79.6 |
| 4 features per window | 13 % | 7 % | 3.6 | 0.89 | 0.60 | 0.35 | 2.4x | 70.8 | 71.2 | 76.0 |
| 32 features (torch run) | 45 % | 41 % | 11.8 | 0.67 | 0.32 | 0.14 | 4.1x | 80.7 | 72.6 | 79.8 |
| as trained, read out at 17.5 mV | 34 % | 27 % | 8.5 | 0.97 | 0.55 | 0.29 | 3.0x | 80.5 | 79.7 | 84.1 |

- **A stronger conscience does recruit the dead units** (798 structured templates instead of 685; ring 1: 314/320
  instead of ~285) — and the code gets worse: the recruits become more near-duplicate offsets, races get closer, and the
  idealised learner drops 3 points. *Structured is not the same as useful.*
- **Shift augmentation fills the starved windows** (noise 33 → 25 %) with templates of strokes that only reach the
  corners in shifted digits; on real, centred test digits they still never fire. No gain.
- **Fewer features per window** makes the code more stable (shift 0.39 → 0.49 → 0.60) by throwing information away
  (contrast 3.7 → 3.1 → 2.4x); the readers do not improve. *This is a prediction for the queued batch 2: F8/F4 should
  not help layer 2.*
- **32 features** only adds dead units: 41 % never fire and the effective features per window rise from ~10 to ~12. With
  this conscience a window supports about ten features however many it is given — which is why F32 ≈ F16 on Unity.
- **The soft readout (17.5 mV)** is the only change that improves every reader: it removes the Poisson flips (0.97) and
  softens the shift damage (0.55) by letting the runner-up fire too, adding information instead of removing it.
  *Prediction for batch 2: the `--l1-readout-inhibition 17.5` run should beat the 200 mV runs on layer 2.*

---

## How to think about layer 1 (the intuition)

Each active window outputs **one symbol out of ~10** learned stroke fragments. The fragments tile a continuous variable —
where, and at what angle, a stroke crosses the window — into narrow bins about a pixel wide, and the symbol is chosen by a
race that is decided in the first ~40 ms. Near a bin edge (which, with bins that narrow, is most of the time) Poisson
noise or a 1-px shift flips the symbol. Pixels degrade gracefully; symbols do not.

A layer-2 D&C neuron is a template over those symbols that it builds by imprinting the instances it wins. If the symbols
of the next instance of the same class are mostly different ones, the imprint does not generalise — which is what the
1-NN and k-means readers measure, and what layer 2 lives with. An averaging reader is fine because flips cancel in a mean.

So the noise templates are the visible problem but not the important one. **"Making layer-1 synapses useful" means making
the symbols stable under perturbations that do not change the digit, without merging the ones that do.** The conscience
pushes the other way (equal use → equal drive → close races among near-copies), and more features make it worse.

## Where to go from here

1. **Soft readout on layer 2** (cheap, already a flag): `--l1-readout-inhibition 17.5`. Best-supported prediction here.
2. **Learn invariance instead of tiling it.** The siblings are the same stroke at different offsets; a learning rule that
   binds views that follow each other in time — Földiák's trace rule (Neural Computation 3:194, 1991), which has STDP
   formulations — applied to short sequences of slightly shifted views of a digit would merge offset-siblings into
   shift-tolerant features. That is the direct attack on the measured defect, and a genuine research contribution for a
   spiking two-layer network. The numpy stage-A replica here is the cheap place to prototype it, scored with
   `code_eval.py`.
3. **Recruit for diversity, not for equal use.** If dead units are to be put to work, the pressure has to push them away
   from their siblings (e.g. anti-Hebbian lateral decorrelation), not just lower their threshold; θ⁺ 0.3 shows what
   equal-use recruitment does.
4. Housekeeping: the outer ring can be dropped (448 neurons, 69 % of them noise, no accuracy cost).

---

## Files

| script | what it does | output |
|---|---|---|
| `l1lib.py` | numpy replica of the inference path (layer 1, pool2, layer 2, label assignment, evaluation) | — |
| `validate.py RUN` | replays a finished run and compares with its own final accuracy | console |
| `census.py` | per-neuron structure, wins and window exposure | `cache/census.npz` |
| `ablate.py` | hides channel groups from layer 2 in the trained network | `cache/ablate.npy` |
| `info.py` | per-channel class information | `cache/info.npz` |
| `stability.py RUN INH wta/nowta TAG` | Jaccard-based stability of a code (repeats, 1-px shift, class) | `cache/stability_*.npz` |
| `mechanism.py`, `needed.py` | race margins, winner agreement, dead units' deficit | `cache/mechanism*.np*`, `cache/needed.npz` |
| `train_l1.py` | numpy replica of stage A with per-neuron win logging; options `--theta-plus --features --shift-aug --examples ...` | `cache/l1_*.npz` |
| `extract.py SRC OUT [EXAMPLES]` | a trained layer 1 (or a checkpoint of one) in the format `code_eval.py` reads | `cache/eval_*.npz` |
| `code_eval.py SPEC --tag T` | capacity, stability and the three readers for `pixels`, `run:<run>` or `npz:<file>` | `cache/code_eval/T.json` |
| `figures_layer1.py [A1 ... A9]` | all figures | `figs/` |

`bash run_all.sh` reproduces everything (~1.5 h on 2 CPU cores, mostly the five stage-A trainings). Paths: MNIST from
`<repo>/data/MNIST/raw`, runs from `../results/<run name>`.
