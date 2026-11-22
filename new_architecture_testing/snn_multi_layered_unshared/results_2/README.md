# results_2 — runs made with the stage-A accuracy code (2026-11-15 onward)

From batch 4 on, runs are written here (`--results-dir`), one folder per batch. In these runs F1 starts at image 0:
during stage A layer 1 is scored as the output layer (teal), after the freeze layer 2 is (blue). F1–F7 use total training
images on the x-axis, and a green dashed line marks the end of stage A. Earlier runs (batches 0–3) are in `../results/`;
the folder numbering continues from there.

| folder | batch | question | outcome |
|---|---|---|---|
| `05_batch4_best_followup_6ep_20261120/` | 4 | Does batch 3's best layer-2 setting (drive 0.36, lr ×10) replicate, and do the 17.5 mV readout, 12 epochs, a lower drive or pooling improve it? | Replicates (82.3 / 82.4 %); every variant lands at 82–84 % (single layer 85.9 %); pooling −5 points; the ~24-point loss on shifted digits is unchanged. Layer 1 alone: 65–67 %. |
| `06_batch5_layer1_code_stability_6ep_20261122/` | 5 | Do layer-1 changes aimed at the close races between near-copy features (lower drive, readout inhibition 8 / 50 mV, weaker conscience, stride 2) improve the code? | pending |

Each folder's README.md has the configuration, the questions and the results.
`summary.csv`, `S1_accuracy_overlay.png` and `S2_shift_tolerance.png` here cover every run in results_2 (the `group`
column is the batch folder):
```bash
python new_architecture_testing/snn_multi_layered_unshared/summarize_multi_layer.py \
    --results new_architecture_testing/snn_multi_layered_unshared/results_2
```
