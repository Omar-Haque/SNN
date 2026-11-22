# 07 — batch 6: combining and replicating batch 5's two winners (submitted from `unity/submit_multi_layer.sh`, 2026-11-22)

## Purpose
Batch 5 (`../06_batch5_layer1_code_stability_6ep_20261122/`) found two layer-1 changes that each beat the baseline by more
than run-to-run noise, one seed each:
- **stride 2** (121 overlapping windows instead of 64): 84.6 % held-out, 59.2 % shifted (baseline 82.1 / 57.4);
- **8 mV readout inhibition** (losing features keep firing after the winner, 4.4 per window): 84.2 % held-out, shifted
  unchanged (56.9).

The single layer is at 85.6 / 60.5. What tracked accuracy across batch 5 was how many layer-1 spikes per image reach layer
2 — and the two winners add them from different sources (more windows vs more features per window), so they may add up.
Batch 5 also showed that the ~25-point loss on ±2 px shifts did not follow the layer-1 code's shift stability: at 8 mV the
code is as shift-stable as raw pixels, yet shifted accuracy did not move. Pooling was tried once (batch 4), on the 200 mV
code, whose windows flip under a 1-px shift; on the 8 mV code it gets a fair test.

## Baselines (not rerun; all with `--l2-drive 0.36 --l2-lr-mult 10`, 6 epochs)
Accuracy = mean of every snapshot in epochs 4–6 (held-out / shifted ±2 px).

| run | seed | epochs 4–6 | folder |
|---|---|---|---|
| single layer | 0 | 85.6 / 60.5 | `../../results/03_batch2_layer1_variants_6ep_20261111/singlelayer_…` |
| baseline layer 1 (stride 3, 200 mV) | 0 | 82.1 / 57.4 | `../05_batch4_best_followup_6ep_20261120/…_ep12_…` (first 6 epochs) |
| baseline layer 1 | 1 | 81.2 / 56.1 | `../05_batch4_best_followup_6ep_20261120/…_seed1_l1wta_l2wta_…` |
| 17.5 mV readout | 0 / 1 | 82.0 / 58.0 · 83.8 / 57.8 | `../05_batch4_best_followup_6ep_20261120/…_rd17.5_…` |
| pool2, 200 mV | 0 | 78.2 / 52.9 | `../05_batch4_best_followup_6ep_20261120/…_pool2_…` |
| stride 2 | 0 | 84.6 / 59.2 | `../06_batch5_layer1_code_stability_6ep_20261122/…_p7s2f16_…` |
| 8 mV readout | 0 | 84.2 / 56.9 | `../06_batch5_layer1_code_stability_6ep_20261122/…_rd8_…` |

Seed-0 runs at stride 3 share the same layer 1 (identical stage A), so the readout-inhibition runs differ only in the
readout. The baselines agree within ±0.5 on the epochs 4–6 average; count a change if it is ≥ 1.5–2 points.

## Configuration
Common to every job: `--l1-wta --l2-wta --theta-tau 30000 --epochs 6 --snapshot-interval 200 --l2-drive 0.36
--l2-lr-mult 10`; layer 1 otherwise default (7×7 windows, F = 16, row sum 12, w_max 1.5, θ⁺ 0.1, 200 mV training
inhibition, stage A up to 30,000 images).

| job | flags | question | compare with |
|---|---|---|---|
| `ml7_s2_rd8` | `--stride 2 --l1-readout-inhibition 8 --seed 0` | Do the two gains add up? | stride 2 (84.6), 8 mV (84.2), single layer (85.6) |
| `ml7_s2_rd8_s1` | same, `--seed 1` | … on a second seed | seed-1 baseline (81.2) |
| `ml7_s2_s1` | `--stride 2 --seed 1` | Is stride 2's gain real? | seed-1 baseline (81.2) |
| `ml7_rd8_s1` | `--l1-readout-inhibition 8 --seed 1` | Is 8 mV's gain real? | seed-1 baseline (81.2), 17.5 mV seed 1 (83.8) |
| `ml7_rd4` | `--l1-readout-inhibition 4 --seed 0` | Does an even softer readout keep helping? | 200 / 50 / 17.5 / 8 mV on the same layer 1: 82.1 / 81.7 / 82.0 / 84.2 |
| `ml7_pool2_rd8` | `--pool2 --l1-readout-inhibition 8 --seed 0` | Can 2×2 pooling buy shift tolerance once the code is stable? | pool2 at 200 mV (78.2 / 52.9), 8 mV without pooling (84.2 / 56.9); watch the held-out − shifted gap (~25–27 everywhere so far) |

`--pool2` needs an even window grid, so it runs at stride 3 (8×8 windows); at stride 2 (11×11) it would fail.
Stride-2 runs took ~2 h in batch 5 like the others; every job has a 5-h limit.

## What to watch
- Layer-1 output reaching layer 2 (`run.json` → `l2_input_rate_per_ms`, measured at calibration): 0.95 spikes/ms for the
  baseline, 1.55 at 8 mV, 2.03 at stride 2; the combination should give roughly 3–4. If accuracy keeps rising with it, the
  "more channels" reading holds; if it flattens, batch 5's curve was already near its ceiling.
- Layer-2 spikes per presentation (console, `summary.csv`): with many more input spikes per ms, calibration makes each
  one worth less; layer 2 should still fire ~10–15 times per presentation.
- Held-out − shifted gap for the pooling run.

## Run and summarise
```bash
bash unity/submit_multi_layer.sh smoke     # optional: one short job (stride 2 + 8 mV) on tiny settings -> results/smoke_tests/
bash unity/submit_multi_layer.sh           # the six jobs
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2     # summary.csv, S1, S2 in results_2/
```

## Results
(to be filled in when the runs finish)
