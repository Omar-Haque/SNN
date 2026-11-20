# 00 — first unshared run, no tie-break (2026-10-31, Mac)

**Purpose:** first test of the locally connected two-layer network (64 windows of 7x7 at stride 3, 16 private features per
window, 400-neuron D&C layer 2 on the 1,024 layer-1 lines), after the shared-template version failed (templates averaged
into blobs, layer 2 silent).

**Configuration:** defaults of `multi_layer_unshared.py`, 2 epochs, layer-2 θ⁺ 0.5, no θ decay, no tie-break in either layer.

**Result:** 69.7 % held-out, 44.0 % on the ±2 px shifted test set (single layer: ~87 / 61).

**What it showed:** the architecture works (the shared version was at chance), but both layers ran in a *volley* regime.
Several features of a window crossed threshold in the same millisecond and all fired, and layer 2 received those
synchronous bursts and fired one spike each from dozens of neurons per presentation. No layer-2 neuron ever took a
presentation, so templates averaged everything and every θ rose in lockstep. This led to the `--l1-wta` / `--l2-wta`
tie-break flags used in every later batch.
