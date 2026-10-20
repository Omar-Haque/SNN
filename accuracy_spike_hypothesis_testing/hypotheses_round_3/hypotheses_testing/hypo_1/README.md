# Hypothesis 1 (round 3): reservoir exhaustion

**Hypothesis.** Training contains a reservoir of *untrained* neurons (template still ≈ its random init). A reservoir
neuron is recruited when it wins a presentation outright: it imprints that image in one go and, having the lowest θ
in the network plus a sharp single-image template, it wins everything resembling that image for ~10k examples — it
*hijacks* a class — until its θ rises to the trained population's level and it settles. While the reservoir lasts
there is always a current hijacker, so accuracy sits at ~55–65 % and the biased class rotates. When the last untrained
neuron is used up, the last hijacker settles within one lifetime, competition becomes fair, and accuracy jumps within
~5–10k examples and stays flat. Nobody needs to reach θ = 0; a hijacker only needs to reach the trained population's
level (which is why clamping θ at −10/−20 did nothing and −30 started to smear the spike).

Evidence so far (round-2 checkpoints, 10 runs × WTA on/off): ~75 untrained neurons at 100k, drained at a constant
12–15 per 5k examples, reaching 0 at ~130k — exactly when accuracy jumps; each recruit's template moves by ‖Δw‖ ≈ 3.5
in its recruitment interval (vs ~0.12 for a trained neuron) and keeps moving 5–8× more than the median for the next
~10k; the 20 biggest movers carry ~50 % of all weight change before the spike and ~10 % after.

## Files

| file | purpose |
|---|---|
| `hypo_1.py` | the experiment: base model (copied from `src/snn_normalized replicate_results.py`), interventions, measurements, per-run figures |
| `hypo_1_plots.py` | figure code (numpy/matplotlib only); `--replot` uses it |
| `summarize_hypo_1.py` | overlays all finished runs; writes `results/summary.csv` and `S1..S3` |
| `run_variations.sh` | the full set of variations, sequentially |

Base parameters (cross-checked against `hypotheses_round_2/master_test.py`): 400 neurons, batch 32, 3 epochs,
350 ms, θ+ = 0.05 (summed over the batch, /16), θ-decay 1e7, lr 1e-2 / 1e-4, 200 mV lateral inhibition, hard WTA off,
no θ clamp, weights normalised to 78.4.

## Variations and predictions

| variation | what happens at the intervention | hypothesis predicts | the alternative reading predicts |
|---|---|---|---|
| `baseline` | nothing | spike at ~125–135k, reservoir hits 0 there | — |
| `refill --n-refill n` (110k) | *n* neurons from the **lowest-θ half of the trained population** are reset to uniform random weights, θ = −52 | spike delayed by *n* / drain-rate (50 → ≈ +20k), **linear in n**; plateau continues | θ-clock on the dominants: no delay (dominants untouched) |
| `pseudo_refill --n-refill n` (110k) | same, but θ set to the trained **median** | no delay (untrained but cannot win) | — |
| `sabotage` (100k) | every untrained neuron's θ raised to the trained median; **weights untouched** | spike within ~10k, ceiling a point or two lower | "the jump is the recruits coming online": **no spike at all** |
| `imprint` (100k) | every untrained neuron given a random training image as template (normalised to 78.4), θ = trained median | spike within ~10k | — |

Spike time is defined quantitatively: the first snapshot after which held-out accuracy never drops below 78 % again.
Baseline seed spread is ~125–135k, so predicted shifts of ±20k are resolvable with 1–3 seeds per variation.

## How to run

```bash
cd <repo root>/hypotheses_round_3/hypotheses_testing/hypo_1
python hypo_1.py --smoke --variation refill        # 1–2 min: checks the whole code path, writes a *_SMOKE run (ignored by the summary)
python hypo_1.py --smoke --variation sabotage
bash run_variations.sh                             # or pick variations by hand, see below
python summarize_hypo_1.py                   # after any number of runs
```

Single variations: `python hypo_1.py --variation baseline --seed 0`, `--variation refill --n-refill 50`, `--variation pseudo_refill --n-refill 50`,
`--variation sabotage`, `--variation imprint`. Options: `--at` (intervention example count), `--seed`, `--epochs`, `--snapshot-interval`
(batches between held-out snapshots; 100 = every 3200 examples), `--tag`. Each full run is ~35–40 min on MPS with the
default snapshot interval. Every run writes `results/<variation>_..._<timestamp>/` with `run.npz`, `run.json` and the figures;
nothing is overwritten.

## How to read the figures

Per run (`results/<run>/`):

- **F1_accuracy** — held-out accuracy (blue), BindsNET-style estimate (orange, smoothed). Red dashed = intervention,
  green dotted = spike time by the 78 % criterion.
- **F2_reservoir** — number of untrained neurons (top-100 share < 0.35) vs. time, accuracy in the background.
  *Prediction:* drains at a constant rate and reaches 0 at the spike; jumps by exactly *n* at a refill; stays flat
  under sabotage (nothing can be recruited any more).
- **F3_hijacker_index** — share of the weight change since the last snapshot carried by the 20 biggest movers.
  *Prediction:* ~0.5 during the plateau, ~0.1 after the spike; collapses within ~10k after sabotage.
- **F4_winner_age** — top: age (examples since recruitment) of the 5 most active neurons in each window; bottom:
  % of window spikes from neurons recruited < 10k ago, and from untrained neurons. *Prediction:* the top winners are
  young throughout the plateau and old after the spike. This is the figure that separates "hijacking" from the duller
  "stray spikes from untrained neurons corrupt the vote".
- **F5_activity_by_recruitment** — the mechanism picture. Rows = neurons sorted by recruitment time (never trained at
  the bottom), columns = windows (each column spans the 3,200 examples ending at its snapshot), colour = sqrt of that
  neuron's share of the window's spikes, cyan dot = its recruitment time, which is the snapshot at the END of the window in
  which it imprinted — so the dot sits at the right edge of the neuron's first bright cell.
  *Prediction:* a bright band immediately to the right of each cyan dot (recruit → hijack → settle), forming a diagonal
  that ends at the spike; after the spike activity is spread evenly.
- **F6_theta** — θ of the trained population (median, 10–90 %, max) and of the untrained neurons. Shows what a
  hijacker has to climb to (the trained median), and that the max never needs to reach 0.
- **F7_class_bias** — per class and snapshot: predicted − true count on the held-out set (red = over-predicted).
  Black × = the class of the window's top winner when that winner is young. *Prediction:* the red cells sit on the ×'s.

Across runs (`results/`): **S1** accuracy overlays (refill variations left, drain variations right), **S2** spike time vs. number
of refilled neurons with the predicted line T₀ + n / drain-rate (drain-rate measured from the baseline runs), and spike
time per variation, **S3** reservoir overlays. `summary.csv` has one row per run: spike time, pre-spike floor, post-spike
ceiling, reservoir just before/after the intervention, drain rate.

## What would confirm / refute

- Confirmed if: refill delays linearly in *n* while pseudo refill does not; sabotage (and imprint) bring the spike forward to
  within ~10k; F2 hits 0 at the spike in every run; F4/F5 show young recruits dominating the plateau.
- Refuted if: sabotage produces no spike (then recruitment itself, not its ending, is what the jump needs — look at what
  the recruits fill in); or refill does not delay / pseudo refill delays as much as refill (then the reservoir is a bystander);
  or F4 shows the plateau's winners are old neurons (then the bias is not hijacking by recruits).
- Uninterpretable if: F2 keeps draining after sabotage (the median θ was not enough to stop recruitment — raise it).
