# Initial architecture testing: convergence-speed sweep

**Question.** Hijacking (round-3 conclusion) is what makes the baseline converge slowly: a freshly recruited neuron
keeps winning presentations of several classes until its θ has risen ~25 mV to the trained population's level, and the
reservoir of recruitable neurons feeds a new hijacker every ~350 examples until it is empty at ~130k. Variants with a
larger θ⁺ shorten each hijack and show no plateau, but reach a lower accuracy. This folder measures that trade-off and
looks for the variant with the **highest accuracy among the fast ones**.

**Why not a blind grid.** The round-0 grid (72 cells, batch 256, hard WTA on) scatters by ±5 points between neighbouring
cells with one seed each; at ~13 min per epoch a multi-knob grid mostly measures noise. The mechanism says one quantity
matters — how many spikes a recruit needs to close its θ gap — so the sweep is one-dimensional in θ⁺ with two seeds,
plus one structural reference (below), and only then widens around the winner.

## Files

- `sweep.py` — `hypo_1.py` with the interventions removed and three model constants turned into knobs
  (`--theta-plus`, `--inhibition`, `--init uniform|imprint`, optional `--theta-init`). Model dynamics, estimators and
  all per-snapshot measurements are verbatim (AST-identical to `hypo_1.py` except the two constructors, `main()` and
  four bookkeeping lines in `train()`). Default **1 epoch**.
- `sweep_plots.py` — copy of `hypo_1_plots.py`; the footer describes the knobs instead of an intervention.
- `summarize_sweep.py` — torch-free; all finished runs → `results/summary.csv` + `S1..S4`.
- `run_sweep.sh` — the planned runs.

## The variants

| variant | what changes | predicted effect on the plateau | predicted cost |
|---|---|---|---|
| θ⁺ ∈ {0.05, 0.1, 0.2, 0.5, 1.0} | θ += spikes.sum(batch) · θ⁺ / 16 per timestep | a hijack lasts ~(gap / θ⁺) spikes: ~1 window at 0.05, ~0.1 window at 0.5 | one won presentation (~30 spikes) costs ~30·θ⁺/16 mV: 0.09 mV at 0.05, ~1 mV at 0.5, ~2 mV at 1.0. Large θ⁺ hands the next image of a class to a worse-matching neuron (templates blend) and eventually θ outruns the input drive (silence; the 20–30 % corner of the round-0 grid) |
| `--init imprint` (imprint at zero) | every row starts as a random training image scaled to 78.4; θ⁺, inhibition, STDP unchanged | no flat template exists, so no reservoir and no hijack at all | none expected — this is the **reference**: the accuracy a no-plateau network reaches when nothing else changes. Data-dependent initialisation, not a learning rule; it bounds what the θ⁺ variants can hope for |
| `--theta-init` (optional) | all θ start above −52 | "sabotage at zero" | flat templates with high θ never fire, so only a few mV above base is meaningful |
| `--inhibition` (optional, later) | lateral inhibition < 200 mV | more than one winner per presentation | unknown; TODO item |

## Reading the results

`summarize_sweep.py` reports two numbers per run and refuses to collapse them: **final_acc** (mean of the last three
snapshots) and **t75** (first snapshot after which held-out accuracy never drops below 75 %; `t78` is the old spike
criterion). S4 plots one against the other. Also per run: `hijack_max` (≈0.5 if a plateau existed, ≈0.1–0.2 if not),
`reservoir_end`, `theta_med_end` (θ runaway), and the full F1–F7 set so every point on the curve comes with its
mechanism figures.

Things to look at first:
- Does imprint-at-zero reach the baseline ceiling (~85 %)? If yes, every θ⁺ variant that falls short is paying for
  θ⁺ itself, not for the missing plateau. If no, the plateau was doing useful work (the "every neuron converges" effect).
- The smallest θ⁺ whose `hijack_max` stays low: that is where the plateau disappears, and the first candidate.
- F6 of the large-θ⁺ runs: if the trained median keeps climbing without bound, that variant is on its way to silence.

## How to run

```bash
cd new_architecture_testing/initial_testing
python sweep.py --smoke                      # ~1-2 min code-path check (run folder tagged _SMOKE, ignored by the summary)
bash run_sweep.sh                            # the planned set, ~3.5 h on MPS
python summarize_sweep.py                    # any time; uses whatever has finished
python sweep.py --replot results/<run>       # regenerate F1-F7 without torch
```
