# STDP_testing: two mechanism demonstrations

Both are recorded during one baseline training run (θ⁺ 0.05, uniform init, 200 mV inhibition — the hypo_1 network;
`dc_model.py` is the LIF cell and the D&C layer extracted verbatim from `test4.py`, AST-checked).

**Test 1 — the first neuron to fire takes the whole presentation.** At chosen points in training (default 3,200 and
32,000 examples) one batch's complete spike raster is logged: every spike of every neuron at every millisecond of all
32 presentations, plus the membrane potential of all 400 neurons for the first image.
- `T1_raster_<N>k.png`: four presentations, every neuron that fired, spikes in time order, winner in red.
- `T1_membrane_<N>k.png`: the winner and its two closest runners-up on one image — the winner crosses its threshold,
  is reset to E_rest = −65 mV (no refractory period) and re-fires every ~12 ms; each of its spikes knocks every other
  neuron down by 200 mV, so none recovers. The dashed lines are the thresholds θ(t), logged every millisecond of the
  raster batch; the inset zooms on the winner's θ, which rises by only θ⁺/16 = 0.003 mV per spike (≈0.08 mV over a
  presentation at θ⁺ = 0.05, invisible at the scale of the main axes). θ is one value per neuron shared by the batch's
  32 presentations, so the inset also collects the winner's spikes on the other 31 images.
- `T1_stats.png`: over all logged presentations — distinct neurons that fired, the winner's share of the spikes,
  first-spike latency, the winner's inter-spike interval, and how many neurons fired after the winner's first spike.
  ("Winner" in these panels = the neuron with the most spikes in the presentation, not the first spiker.)
- `T1_ties_<N>k.png`: every presentation of the batch as a row, sorted by how many neurons fired in the first spiking
  millisecond. All neurons see the same input spike train and thresholds are checked once per millisecond for all of
  them at once, while inhibition only arrives at the next step — so neurons whose drive differs by less than a few
  percent cross in the same step and all fire. Each tied neuron is then reset to −65 mV and receives −200 mV from
  every other tied neuron (everyone else receives −200 mV per tied neuron), and the presentation is silent until the
  neuron with the largest drive climbs back (≈ 60–100 ms after a 2-way tie, ≈ 180–230 ms after a 10–18-way tie; the
  curves in the middle panel are the LIF recovery time from −65 − 200(k−1) mV). The low "winner's share" values in
  `T1_stats` are exactly these presentations. Who ties: reservoir neurons (near-identical uniform weights, identical θ)
  on bold images, and same-digit template families (θ equalizes their latencies, which is what it is for).
- `T1_membrane_tie_<N>k.png` (runs made after 2026-11-01 only): membrane potentials on the biggest-tie presentation of
  that batch — the tied neurons dropping to −2,000…−3,600 mV and the strongest template climbing out first.

**Test 2 — the template imprints far faster than θ catches up.** From 20,000 examples on, a neuron whose template is
still flat (top‑100 share < 0.35) and that wins a presentation (≥ 15 spikes on one image) is a fresh hijacker. For the
first six, the run records its template every 50 ms during its first presentations, at every batch end for 300
batches (9,600 examples), its θ, the presentations it wins per batch, the trained population's θ, and the image it won.
- `T2_recruit<k>_n<id>.png`: the won image and the template filmstrip; template sharpness and similarity to the final
  template against examples since the first win; θ of the recruit against the trained population's median and 10th
  percentile, with the presentations it wins per batch (the hijack) as bars. Two dotted lines mark "template formed"
  (90 % of final sharpness) and "θ reached the trained level".
- `T2_timescales.png`: those two numbers for every recruit, log scale, with their ratio.

## Run
```bash
cd new_architecture_testing/STDP_testing
python stdp_demo.py --smoke                                     # ~1 min code-path check
python stdp_demo.py                                             # ~15 min on MPS: 60k examples, rasters at 3.2k and 32k
python stdp_demo.py --examples 160000 --raster-at 3200,32000,150000   # ~40 min: adds a post-spike raster
python stdp_demo.py --replot results/<run>
```
Knobs: `--raster-at`, `--watch-from`, `--n-recruits`, `--track-batches`, `--frame-every`, `--win-spikes`,
`--theta-plus` (0.5 shows the fast regime's short hijacks for contrast), `--inhibition` (e.g. 17.5 for the
BindsNET value: several neurons per presentation, the contrast case for test 1).

## What the 60k run showed (results/demo_tp0.05_inh200_ex60k_seed0_20261030-124211)

**Claim (a) holds with one word added: the first *solo* spiker takes the whole presentation.** In the 32k batch,
17/32 presentations begin with a single neuron crossing threshold; it then re-fires every ~12–13 ms (reset to −65 mV,
no refractory period, ~1 mV/ms climb back) and holds 100% of the spikes. The other 15/32 begin with a tie — 7 two-way,
4 three-to-four-way, and four of 10, 11, 12 and 18 neurons at once — and go dark for 58–234 ms before the strongest
neuron climbs out (presentation 5: an 11-way tie at 9 ms, a 3-way tie at 221 ms, a 2-way tie at 305 ms; 16 spikes from
13 neurons, top share 12%). Ties cost spikes (16–26 instead of 25–35) and split the evaluation vote. They are a
consequence of dt = 1 ms, inhibition delayed by one step and no within-step winner-take-all; the hard-WTA option is
exactly a within-step tie-break and stays off in the hypothesis runs.

**The same batch shows the sequential hijack directly.** One neuron, n332 (θ = −42.6 mV against a trained-population
median of −27.4), won 15 of the 17 solo presentations, across digits 0, 1, 2, 3, 6, 7, 8 — mostly the thin digits where
no specialist's drive is large enough to reach its higher threshold first. The specialists (n79 on 7s, n42 on 4s, n220
on 9s, n255/n46 on 0s) only got the presentations they could climb out of after a tie. At 3.2k the hijackers were two
reservoir neurons, n128 (11/32 wins over six digits) and n342 (6/32). The hijacker's θ rises ~0.08 mV per presentation
won, so a reign lasts a few hundred examples — the "window" of the hypothesis work.
