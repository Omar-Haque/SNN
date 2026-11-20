# 04 — batch 3: layer-2 drive and learning rate, 6 epochs (2026-11-13, Unity)

**Purpose:** fix the two layer-2 problems found in batch 2. `--l2-drive` (calibration target, default 1.2 mV/ms) scales
all layer-2 weights: lower drive means each layer-1 spike is worth less, so a neuron has to collect more of the code
before it fires. `--l2-lr-mult` multiplies layer 2's STDP learning rates, to counter slow learning (and the smaller
imprint per win that a lower drive causes).

**Configuration:** same regime as batch 2 (6 epochs, θ decay τ 30k, tie-break in both layers, seed 0); batch 2's control
(65.1 %) and single layer (85.9 %) are the baselines.

| run | held-out | best snapshot | shifted ±2 px | at 6k examples |
|---|---|---|---|---|
| drive 0.36, lr ×10 | **82.3** | 84.8 | 57.7 | 66.2 |
| drive 0.6, lr ×6, 17.5 mV readout | 80.8 | 83.3 | 55.4 | 60.8 |
| drive 0.6, lr ×6 | 77.3 | 80.3 | 55.2 | 58.2 |
| drive 0.6 | 73.7 | 75.5 | 47.6 | 33.8 |
| drive 0.36 | 72.9 | 75.3 | 46.4 | 30.2 |
| lr ×3 | 67.8 | 70.9 | 47.8 | 49.5 |

**What it showed:**
- Lower drive fixes the readout: at drive 0.36 the winner fires after ~22 input spikes from more than half the windows,
  and the spiking network now answers as well as its best-matching template (84 vs 85, against 67 vs 77 for the control).
- A higher learning rate fixes learning speed: 58–66 % after 6k examples instead of 32 %.
- Each alone falls short (lr ×3: 67.8; drive 0.36: 72.9); together 82.3 %, 3.6 points under the single layer on one seed.
  The 17.5 mV layer-1 readout adds ~3.5 points at the same drive and rate.
- Not yet run: drive 0.36 + lr ×10 + 17.5 mV readout, a second seed, more epochs (the best run was still rising).

Details: `../../layer2_analysis/README.md`.
