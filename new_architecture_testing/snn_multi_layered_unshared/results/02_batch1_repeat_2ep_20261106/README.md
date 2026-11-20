# 02 — repeat of batch 1 (2026-11-06, Unity)

**Purpose:** none intended. Batch 2 was meant to run, but the new submit script had not been committed, so Unity ran the
old batch-1 script again with the newer code (which logs extra layer-1 diagnostics). It serves as a replication.

**Configuration:** identical to `01_batch1_variants_2ep_20261102`.

| run | 2026-11-02 | this repeat |
|---|---|---|
| single layer | 86.8 / 60.6 | 87.0 / 60.8 |
| two-layer, seed 0 | 68.9 / 41.3 | 67.6 / 42.9 |
| two-layer, seed 1 | 67.8 / 43.7 | 66.6 / 42.8 |
| tie-break in layer 2 only | 68.6 / 44.4 | 67.5 / 43.7 |
| 32 features per window | 68.7 / 45.6 | 69.2 / 44.7 |
| 2x2 pooling | 65.9 / 41.9 | 64.1 / 40.6 |
| stride 7 | 52.8 / 33.9 | 52.0 / 32.6 |

(held-out / shifted ±2 px)

**What it showed:** the same configuration and seed differs by 1–2 points between runs, which is the noise level to read
every other comparison against. The seed-0 run here is the one analysed in `layer1_analysis/`: a third of the layer-1
templates stay random but are silent and harmless, and the real weakness of the layer-1 code is that it is fragile (a
fresh Poisson draw of the same image keeps 75 % of it, a 1-px shift 40 %).
