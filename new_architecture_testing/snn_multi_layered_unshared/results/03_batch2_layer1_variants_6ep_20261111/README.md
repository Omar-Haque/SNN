# 03 — batch 2: layer-1 code variants, 6 epochs (2026-11-11, Unity)

**Purpose:** test the predictions from `layer1_analysis/`: does a more stable layer-1 code help layer 2? Coarser features
(8 or 4 per window) make the code more stable by losing information; a softer layer-1 readout (17.5 mV inhibition after
freezing, so a runner-up feature can fire) makes it more stable while adding information. Also: is the gap to the single
layer just too little training?

**Configuration:** 6 epochs, layer-2 θ decay τ = 30k examples (thresholds stay stationary instead of climbing),
tie-break in both layers, seed 0, default layer-2 drive 1.2 and learning rate.

| run | held-out | shifted ±2 px |
|---|---|---|
| single layer | 85.9 | 60.8 |
| 16 features, 17.5 mV readout | 70.0 | 45.7 |
| 16 features (control) | 65.1 | 44.0 |
| 8 features, 17.5 mV readout | 65.0 | 42.2 |
| 8 features | 61.3 | 39.6 |
| 4 features | 55.1 | 35.6 |

**What it showed:**
- The variants rank exactly as the layer-1 analysis predicted: coarser features hurt, the soft readout helps (~+5).
- Three times more training bought nothing; the control is flat from ~180k examples.
- Replaying the trained weights (`layer2_analysis/`): layer 2's templates are worth 76–81 % if the best-matching one is
  chosen, but the spiking network reads them out at 66–68 %. The winning layer-2 neuron fires after only ~12 input spikes,
  when a third of the windows have reported, because the calibration makes each layer-1 spike worth ~10 mV.
  Layer 2 also learns slowly: 388 of 400 rows are still at their initial weights after 32k examples.
