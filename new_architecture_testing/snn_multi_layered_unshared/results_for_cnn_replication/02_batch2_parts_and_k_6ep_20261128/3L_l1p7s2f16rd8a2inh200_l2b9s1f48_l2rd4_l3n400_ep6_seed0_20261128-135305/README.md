<!-- written by three_layer_architecture/describe_runs.py; rerun it to refresh -->
# 3L_l1p7s2f16rd8a2inh200_l2b9s1f48_l2rd4_l3n400_ep6_seed0_20261128-135305

three-layer network · MNIST · seed 0 · 6 epochs × 60,000 images in stage B · Unity job `t2_lrf9` on gypsum-gpu013 (Tesla M40 24GB), 179 min

**In short:** layer 1 learned with 200 mV within-window inhibition; it then ran at 200 mV while layer 2 learned and at 8 mV in stage B and for scoring. Layer 2 = 9×9-window blocks (23×23 px) at a stride of 1 window, 48 features per block (432 neurons); it learned at 200 mV and was read at 4 mV. Layer 3 = 400 neurons reading layer 2 only.

**Started with** (every setting not on this line is the default):
```
python three_layer.py --results-dir ../new_architecture_testing/snn_multi_layered_unshared/results_for_cnn_replication/02_batch2_parts_and_k_6ep_20261128 --l1-a2-inhibition 200 --l2-readout-inhibition 4 --l2-examples 60000 --l2-block 9 --l2-stride 1 --l2-features 48 --seed 0
```
Note in the submit script: *3x3 blocks of 23x23 px x 48 = 432*

## Settings

| | layer 1 | layer 2 | layer 3 |
|---|---|---|---|
| reads | 7×7-pixel windows of the image | 9×9 neighbouring layer-1 windows (23×23 px), all 16 features of each | every layer-2 neuron |
| layout | 11×11 windows, stride 2 px, × 16 features = 1,936 neurons | 3×3 blocks, stride 1 window, × 48 features = 432 neurons | 400 neurons |
| inputs per neuron | 49 pixels | 1296 | 432 |
| weights | unshared: each window has its own 16 templates | unshared: each block has its own | fully connected |
| competition (lateral inhibition) | among the 16 features of a window | among the 48 features of a block | among all 400 |
| inhibition while it learns | 200 mV (stage A) | 200 mV (stage A2) | 200 mV (stage B) |
| inhibition afterwards | 200 mV while layer 2 learns (stage A2); 8 mV in stage B and scoring | 4 mV in stage B and scoring | – |
| tie-break (`wta`: same-step winners → only the strongest fires) | on | on | on |
| winners per image | one per window at 200 mV, runner-ups too at lower inhibition | one per block at 200 mV, runner-ups too at lower inhibition | the usual Diehl & Cook competition: the winner's spikes inhibit all others |
| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | 0.1, τ 2,000 images | 0.1, τ 2,000 images | 0.5, τ 30,000 images |
| weight scale | row sum 12 (mean 0.245), w_max 1.5 | calibrated (drive × 1): mean 0.192, row sum 249.5, w_max 1.18 | calibrated for 0.36 mV/ms: mean 0.4153, row sum 179.4, w_max 4.153 |
| STDP rates lr+ / lr− | 1e-2 / 1e-4 | × 1 = 0.01 / 0.0001 | × 10 = 0.1 / 0.001 |
| training | stage A: up to 30,000 images, stops early once the templates stop changing; froze at 30,016 | stage A2: up to 60,000 images, same early stop; froze at 5,760 | stage B: 6 epochs × 60,000 images |
| pooling | none | none | – |

Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold −52 mV + θ); pixels → Poisson spikes at 0.1 × intensity per ms, 350 ms per image, no input normalisation; STDP traces τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32; layers frozen once their stage ends. Calibration: layer 2's and layer 3's weight scales are set from the measured input rate (mean weight = drive ÷ rate, w_max = 10 × mean for layer 3). Scoring: neurons labelled from training images 0–1,999; accuracy on test images 0–999 and on the same images shifted by up to ±2 px, every 200 batches (6,400 images).

## Results

- **Held-out / shifted ±2 px, mean of the snapshots of the last 3 epochs: 74.3 / 47.1 %** (best snapshot 76.4 %; last 3 snapshots 75.7 / 47.6 %)
- Per epoch: 58.4 · 70.8 · 72.5 · 74.1 · 73.7 · 75.0
- Layer 1 alone (end of stage A, as the output layer): 67.2 / 49.3 %
- Layer 2 alone (end of stage A2, at its learning inhibition): 62.7 / 40.8 %
- Input into layer 3: 0.87 spikes per ms per image
- Never fired on the 1,000 test digits: 0 of 432 layer-2 neurons, 80 of 400 layer-3 neurons
- Median class purity on the test digits (share of a neuron's spikes from its top digit: ~1 = one-digit prototype): layer 2 0.22, layer 3 1.00
- Run time: 179 min

The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.
