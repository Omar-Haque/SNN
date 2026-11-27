<!-- written by three_layer_architecture/describe_runs.py; rerun it to refresh -->
# 3L_l1p7s2f16rd8a2inh200_l2b3s2f16_l2rd8_l3n400_ep6_seed1_20261127-024702

three-layer network · MNIST · seed 1 · 6 epochs × 60,000 images in stage B · Unity job `tl3_a2r8_s1` on gypsum-gpu088 (NVIDIA GeForce GTX TITAN X), 182 min

**In short:** layer 1 learned with 200 mV within-window inhibition; it then ran at 200 mV while layer 2 learned and at 8 mV in stage B and for scoring. Layer 2 = 3×3-window blocks (11×11 px) at a stride of 2 windows, 16 features per block (400 neurons); it learned at 200 mV and was read at 8 mV. Layer 3 = 400 neurons reading layer 2 only.

**Submitted with** (`unity/submit_three_layer.sh`, job `tl3_a2r8_s1`; every other setting is the default):
```
python three_layer.py --results-dir results/03_batch3_sparse_trained_layer2_6ep_20261126 --l1-a2-inhibition 200 --l2-readout-inhibition 8 --seed 1
```

## Settings

| | layer 1 | layer 2 | layer 3 |
|---|---|---|---|
| reads | 7×7-pixel windows of the image | 3×3 neighbouring layer-1 windows (11×11 px), all 16 features of each | every layer-2 neuron |
| layout | 11×11 windows, stride 2 px, × 16 features = 1,936 neurons | 5×5 blocks, stride 2 windows, × 16 features = 400 neurons | 400 neurons |
| inputs per neuron | 49 pixels | 144 | 400 |
| weights | unshared: each window has its own 16 templates | unshared: each block has its own | fully connected |
| competition (lateral inhibition) | among the 16 features of a window | among the 16 features of a block | among all 400 |
| inhibition while it learns | 200 mV (stage A) | 200 mV (stage A2) | 200 mV (stage B) |
| inhibition afterwards | 200 mV while layer 2 learns (stage A2); 8 mV in stage B and scoring | 8 mV in stage B and scoring | – |
| tie-break (`wta`: same-step winners → only the strongest fires) | on | on | on |
| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | 0.1, τ 2,000 images | 0.1, τ 2,000 images | 0.5, τ 30,000 images |
| weight scale | row sum 12 (mean 0.245), w_max 1.5 | calibrated (drive × 1): mean 1.722, row sum 248.0, w_max 10.55 | calibrated for 0.36 mV/ms: mean 0.2231, row sum 89.2, w_max 2.231 |
| STDP rates lr+ / lr− | 1e-2 / 1e-4 | × 1 = 0.01 / 0.0001 | × 10 = 0.1 / 0.001 |
| training | stage A: up to 30,000 images, stops early once the templates stop changing; froze at 30,016 | stage A2: up to 30,000 images, same early stop; froze at 30,016 | stage B: 6 epochs × 60,000 images |
| pooling | none | none | – |

Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold −52 mV + θ); pixels → Poisson spikes at 0.1 × intensity per ms, 350 ms per image, no input normalisation; STDP traces τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32; layers frozen once their stage ends. Calibration: layer 2's and layer 3's weight scales are set from the measured input rate (mean weight = drive ÷ rate, w_max = 10 × mean for layer 3). Scoring: neurons labelled from training images 0–1,999; accuracy on test images 0–999 and on the same images shifted by up to ±2 px, every 200 batches (6,400 images).

## Results

- **Held-out / shifted ±2 px, mean of the snapshots of the last 3 epochs: 83.5 / 56.6 %** (best snapshot 85.1 %; last 3 snapshots 84.4 / 57.4 %)
- Per epoch: 73.7 · 81.4 · 82.8 · 82.6 · 83.9 · 83.8
- Layer 1 alone (end of stage A, as the output layer): 68.3 / 49.1 %
- Layer 2 alone (end of stage A2, at its learning inhibition): 64.9 / 47.1 %
- Input into layer 3: 1.61 spikes per ms per image
- Never fired on the 1,000 test digits: 10 of 400 layer-2 neurons, 85 of 400 layer-3 neurons
- Run time: 182 min

The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.
