# 01 — batch 1: architecture variants with the tie-break (2026-11-02, Unity)

**Purpose:** with the volley regime fixed by the per-step tie-break, compare the main layer-1 architecture choices against
the single-layer network under the same evaluation.

**Configuration:** 2 epochs, layer-2 θ⁺ 0.5, no θ decay, tie-break in both layers unless stated.

| run | held-out | shifted ±2 px |
|---|---|---|
| single layer (`--no-l1`) | 86.8 | 60.6 |
| two-layer, seed 0 | 68.9 | 41.3 |
| two-layer, seed 1 | 67.8 | 43.7 |
| tie-break in layer 2 only | 68.6 | 44.4 |
| 32 features per window | 68.7 | 45.6 |
| 2x2 pooling between the layers | 65.9 | 41.9 |
| stride 7 (disjoint windows) | 52.8 | 33.9 |

**What it showed:**
- Every two-layer variant sits ~18 points under the single layer and is less shift-tolerant, not more.
- The layer-1 tie-break, 32 features and pooling make no real difference; disjoint windows are much worse.
- Layer 1 itself is fine: every central window learns 16 distinct oriented stroke features, and its code is more
  class-informative than pixels for a simple reader (87 % vs 82 % nearest-centroid). The problem is how layer 2 learns
  from that code, which `layer1_analysis/` and the later batches investigate.
