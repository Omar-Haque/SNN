# Hypothesis 2 (round 3): the sign of "number of weights changed" across the spike

**Question.** Cooper's report shows *number of weights changed per update* **rising** across the spike (his Fig. 6);
our round-1/2 measurement of *weights changed per timestep* **falls**. Same phenomenon, opposite signs. Hypothesis:
they are different quantities, and the reservoir/hijacker mechanism predicts both signs — so both are measured here,
on the same network, in the same run.

| metric | what is counted | when | expected to track | predicted at the spike |
|---|---|---|---|---|
| (A) STDP update | entries of the raw update Δw = Δw⁺ − Δw⁻ with \|Δw\| > thr, per timestep, summed over the batch | **before** clamp and row normalisation (our round-2 metric) | output spikes × active pixels | **falls** (hijackers re-fire every ~2 ms; trained winners have higher θ) |
| (B) net change | stored weights that differ before vs. after the timestep, \|W_after − W_before\| > thr, summed over the batch | **after** normalisation | 784 × rows active per timestep | **rises** |
| (C) net change per batch | stored weights that differ between the start and the end of the batch | after normalisation, once per batch | 784 × distinct neurons that fired in the batch | **rises** (Cooper's x-axis is "training batch number", so this is the most literal reading of his metric) |
| (D) exact inequality | stored weights with `W_after != W_before` (no threshold), per timestep and per batch | after normalisation | 784 × rows touched — IF float32 rounding leaves untouched rows bit-identical; otherwise it is noise (≈ all 313,600 weights) | rises, if meaningful |

Why normalisation matters: every STDP step is followed by `weight.mul_(78.4 / weight_sum)`, which rescales a *whole
row* whenever anything in that row moved. So after normalisation all 784 weights of an active neuron count as changed,
regardless of how many pixels the STDP update actually touched. (A) counts spikes; (B)/(C) count active neurons.

The run also records the two drivers directly — total output spikes per batch and distinct active neurons per batch —
so the interpretation can be checked: (A) should follow the spike count, (C) should equal 784 × active neurons.

**Result of the first full run (seed 0):** every thresholded count is flat or falls at the spike — (A) 0.97×, (B) 0.61–0.66×,
(C) 0.95–1.10×, Frobenius norms ~0.7×, spikes 0.82× — while distinct active neurons per batch rise 1.55×. A thresholded count is
(rows touched) × (weights per row clearing the threshold), and the second factor falls from 514 to 315 of 784 because each
active neuron's update gets smaller; the two cancel. Only a participation count (rows touched, or an exact-inequality diff if
float32 allows it) rises. Whether Cooper's metric was one is unknown without his code.

## Files

- `hypo_2.py` — base model and estimators copied verbatim from `src/snn_normalized replicate_results.py`
  (400 neurons, batch 32, θ+ 0.05 /16, decay 1e7, lr 1e-2/1e-4, 200 mV, WTA off), with **one** marked addition in
  `DiehlAndCookNetwork.forward()`: the raw STDP update is stored in `self.last_delta_w` before being applied. Baseline
  training only (no interventions). Same seeding as `hypo_1.py`, so seed 0 pairs with hypo_1's seed-0 baseline.
- `results/<run>/run.npz` — per-batch series: `batch_points`, `cnt_stdp` (A), `cnt_net` / `cnt_net_strict` (B at
  thr / strict thr), `rows_net` ((B) in rows), `fro_stdp`, `fro_net` (Frobenius norms), `cnt_batch` /
  `cnt_batch_strict` (C), `spikes`, `active`; plus `points`, `accuracy` (held-out, every 3,200 examples).
- `results/<run>/run.json` — config, spike time, pre/post-spike means of every series.

## How to run

```bash
cd <repo root>/hypotheses_round_3/hypotheses_testing/hypo_2
python hypo_2.py --smoke          # ~1 min code-path check
python hypo_2.py --seed 0         # ~40 min on MPS (one extra weight-matrix clone and a few reductions per timestep)
python hypo_2.py --replot results/<run>
```

`--thr` (default 1e-6, as in round 2) is the change threshold for all counts; `--thr-strict` (default 1e-5) is a
second threshold for (B) and (C) only. float32 rounding in the row normalisation moves *untouched* rows by ~1e-7 per
weight; if the 1e-6 and 1e-5 versions of (B) agree, that noise is irrelevant.

## How to read the figures

- **F1_per_timestep_counts** — (A) in purple and (B) in red on the same axes, light = raw per batch, bold = 25-batch
  moving average; dashed red = (B) at the looser threshold (noise check). Prediction: purple falls at the spike, red rises.
- **F2_per_batch_count** — (C), with 784 × (distinct active neurons per batch) dashed on top. If the two coincide, (C)
  *is* an active-neuron count.
- **F3_frobenius** — size (not count) of the change per batch, both ways.
- **F4_drivers** — output spikes per batch (top) and distinct active neurons per batch (bottom).
- **F5_exact_counts** — (D) per timestep (top) and per batch (bottom) against 784 × rows touched / 784 × active neurons, with the
  "every weight" noise level marked. Run `--smoke` first: if (D) sits at the noise level, exact counting is meaningless in float32.

The console prints pre/post-spike means and their ratio for (A), (B), (C), spikes and active neurons at the end.
