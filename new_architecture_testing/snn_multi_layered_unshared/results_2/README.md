# results_2 — runs made with the stage-A accuracy code (2026-11-15 onward)

From batch 4 on, runs are written here (`--results-dir`), one folder per batch. In these runs F1 starts at image 0:
during stage A layer 1 is scored as the output layer (teal), after the freeze layer 2 is (blue). F1–F7 use total training
images on the x-axis, and a green dashed line marks the end of stage A. Earlier runs (batches 0–3) are in `../results/`;
the folder numbering continues from there.

| folder | batch | question | outcome |
|---|---|---|---|
| `05_batch4_best_followup_6ep_20261120/` | 4 | Does batch 3's best layer-2 setting (drive 0.36, lr ×10) replicate, and do the 17.5 mV readout, 12 epochs, a lower drive or pooling improve it? | Replicates (82.3 / 82.4 %); every variant lands at 82–84 % (single layer 85.9 %); pooling −5 points; the ~24-point loss on shifted digits is unchanged. Layer 1 alone: 65–67 %. |
| `06_batch5_layer1_code_stability_6ep_20261122/` | 5 | Do layer-1 changes aimed at the close races between near-copy features (lower drive, readout inhibition 8 / 50 mV, weaker conscience, stride 2) improve the code? | Stride 2 (+2.5 held-out, +2 shifted) and 8 mV readout inhibition (+2.1 held-out) help; lower layer-1 drive costs 8–14 points; 50 mV = 200 mV. Accuracy tracked layer-1 spikes per image, not code stability. Best: 84.6 % (epochs 4–6) vs single layer 85.6. |
| `07_batch6_stride2_readout8_6ep_20261122/` | 6 | Do stride 2 and the 8 mV readout add up, do they replicate on a second seed, does 4 mV help further, and can pooling buy shift tolerance on the 8 mV code? | Both replicate on seed 1; together (stride 2 + 8 mV) they reach the single layer on centred digits: epoch 6 86.3 / 87.5 % vs 85.9. Shifted digits stay at 56–59 % (single layer 60.6); pooling does not help even on the 8 mV code; 4 mV ≈ 8 mV. |
| `08_batch7_fashion_mnist_6ep_20261125/` | 7 | Do the best two-layer network and the single layer carry over to Fashion-MNIST unchanged, and how do they compare there? | Single layer 69.4 / 69.6 %, two layers 63.5 / 64.2 %. The gap is mostly sandals (recall 22 vs 70 %): light images produce few layer-1 spikes, so layer 2, calibrated to the average image, forms almost no sandal neurons. |
| `09_batch8_input_norm_6ep_20261126/` | 8 | Does per-image input normalisation close the Fashion-MNIST gap (sandals), and does it cost anything on MNIST? | pending |

Each folder's README.md has the configuration, the questions and the results.
`summary.csv`, `S1_accuracy_overlay.png` and `S2_shift_tolerance.png` here cover every run in results_2 (the `group`
column is the batch folder):
```bash
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2
```
