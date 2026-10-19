# test2: making the fast variant stationary

**Where initial_testing left off.** θ⁺ = 0.5 and 1.0 reach the baseline ceiling (85–86 %) by ~16k examples, 8× faster
than the baseline, and imprint-at-zero reaches the same ceiling — so the plateau costs time and buys nothing. But the
code has no effective θ decay (τ = 10⁷ batch-timesteps ≈ 0.9 M examples), so θ is a running count of lifetime spikes:
at 60k the median was +14 mV (θ⁺ 0.5) and +41 mV (θ⁺ 1.0), still climbing ~1 / ~2 mV per 1k examples, with spikes per
presentation halved (24→12, 20→8). Prediction: both keep drifting until presentations stop producing spikes and
accuracy falls — most likely what the round-0 grid saw at θ⁺ 0.5 (20–47 % after 3 epochs at batch 256).

**This round** keeps θ⁺ = 0.5 (the smallest value that gave the full speed-up) and asks how to stop the drift without
losing the ceiling. All runs are 3 epochs (180k examples).

| variant | knob | prediction |
|---|---|---|
| (a) no decay, θ⁺ 0.5 and 1.0 | — | the collapse: spikes/presentation keep falling, accuracy follows, 1.0 first |
| (b) θ decay, τ ∈ {15k, 30k, 60k} examples | `--theta-tau` | θ settles at τ·r·θ⁺/16 above base. Measured drift at θ⁺ 0.5 is 0.94 mV per 1k, so τ ≈ 30k puts the equilibrium at the baseline's trained level (25–35 mV above base), the configuration known to give 85 %. Expect: same speed, flat θ and flat spikes/presentation after ~2τ, ceiling 85 %; 15k lower θ (risk: compressed spread), 60k slower to settle |
| (c) clamp at 0 mV — *knob only, not run* | `--theta-max 0` | already known from rounds 1/2: 2–3 points lower, monotone in the cap. Mechanism: the cap pins every neuron that reaches it to the *same* θ, so the self-penalty for winning disappears among the most active neurons and templates merge. Kept as a knob (a 1-epoch run would show the compressed p90−p10 in S3) but not in the script |
| (d) anneal 0.5 → 0.05 at 15k | `--theta-plus-after 15000` | fast start, then the baseline's slow drift (~0.2 mV per 1k); no collapse within 3 epochs, ceiling 85 % |

The equilibrium argument, per neuron and per training example: Δθ = r·θ⁺/16 − (θ − θ_base)/τ, where r is the
neuron's spikes per example; decay grows with θ and the increment doesn't, so θ − θ_base → τ·r·θ⁺/16. Decay keeps θ
*proportional to each neuron's own rate* (ordering and spread preserved); a clamp does not. That is the mechanistic
reason to expect (b) to keep the ceiling where the clamp (c) loses a few points.

## Files

- `test2.py` — `sweep.py` plus three knobs: `--theta-tau` (examples; converted to batch-timesteps as τ·steps/32),
  `--theta-max` (enables the base file's commented-out clamp), `--theta-plus-after N` / `--theta-plus-late X`
  (annealing). Dynamics are verbatim except the decay constant and the optional clamp line in `excitatory_neurons`
  (AST-checked). Default **3 epochs**. All of hypo_1/sweep's measurements and F1–F7 are kept.
- `test2_plots.py` — copy of `sweep_plots.py`; footer shows the knobs, orange dashed line marks the θ⁺ switch.
- `summarize_test2.py` — torch-free; `results/summary.csv` + S1 (accuracy), S2 (spikes per presentation — the collapse
  readout), S3 (median θ and p90−p10 spread — stationarity and compression), S4 (ceiling per variant, with the 60k
  value as a hollow marker so a decline shows as a gap).
- `run_test2.sh` — first pass one seed of (a), (b), (d) (~4 h), summary, then second seeds (~2 h). The clamp is not run.

## How to read the results

- **Collapse (a):** S2 falling toward 0 and S1 turning down. Note where accuracy starts to fall relative to spikes
  per presentation — that is the margin the decay has to protect.
- **Stationarity (b):** S3 top panel flat after ~2τ, `drift_last_60k` ≈ 0 in the table, S2 flat. Then compare the
  three τ's ceilings and the θ level they settle at (does 15k, with a lower θ, lose accuracy the way the clamp does?).
- **Spread (b):** S3 bottom panel. Decay should keep a spread comparable to the baseline's ~15 mV; if τ = 15k
  compresses it and loses accuracy, that is the clamp's failure mode arriving by another route.
- **Annealing (d):** should look like the no-decay run up to 15k and like the baseline's post-spike regime afterwards.

Run folders: `results/tp0.5_tau30k_ep3_seed0_<timestamp>/`, `tp0.5_max0_…`, `tp0.5to0.05at15k_nodecay_…`,
`tp0.5_nodecay_…` (knobs at baseline values are left out of the name).

## How to run

```bash
cd new_architecture_testing/test2
python test2.py --smoke        # ~1-2 min; exercises decay, annealing and the figures
bash run_test2.sh              # ~6 h in two passes; summarize_test2.py can be run at any time
```
