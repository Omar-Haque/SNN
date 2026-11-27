<!-- written by new_architecture_testing/snn_multi_layered_unshared/describe_runs.py; rerun it to refresh -->
# unshared_p7s2f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_rd8_l2wta_drv0.36_lr10_inorm_20261127-024713

two-layer network · MNIST with per-image input normalisation · seed 0 · 6 epochs × 60,000 images · Unity job `ml8_mn_best_in` on gypsum-gpu019 (Tesla M40 24GB), 142 min

**In short:** layer 1 = unshared 7×7 windows at stride 2 px (11×11 windows × 16 features = 1,936 neurons), learned at 200 mV and read at 8 mV; tie-break on. Layer 2 = 400 neurons over all of layer 1, calibrated for 0.36 mV/ms, STDP × 10, tie-break on.

**Submitted with** (`unity/submit_multi_layer.sh`, job `ml8_mn_best_in`; every other setting is the default):
```
python multi_layer_unshared.py --theta-tau 30000 --epochs 6 --snapshot-interval 200 --input-norm --results-dir results_2/09_batch8_input_norm_6ep_20261126 --dataset mnist --stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10 --seed 0
```

## Settings

| | layer 1 | layer 2 (output) |
|---|---|---|
| reads | 7×7-pixel windows of the image | every layer-1 neuron (1,936 lines) |
| layout | 11×11 windows, stride 2 px, × 16 features = 1,936 neurons | 400 neurons |
| inputs per neuron | 49 pixels | 1,936 |
| weights | unshared: each window has its own templates | fully connected |
| competition (lateral inhibition) | among the 16 features of a window | among all 400 |
| inhibition while it learns | 200 mV (stage A) | 200 mV (stage B) |
| inhibition afterwards | 8 mV in stage B and scoring (`--l1-readout-inhibition`) | – |
| tie-break (`wta`: same-step winners → only the strongest fires) | on | on |
| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | 0.1, τ 2,000 images, one θ per neuron | 0.5, τ 30,000 images |
| weight scale | row sum 12 (mean 0.245), w_max 1.5 | calibrated for 0.36 mV/ms from layer 1's measured rate: mean 0.1041, row sum 201.6, w_max 1.041 |
| STDP rates lr+ / lr− | 1e-2 / 1e-4 | × 10 = 0.1 / 0.001 |
| training | stage A: up to 30,000 images, stops early once the templates stop changing; froze at 30,016 | stage B: 6 epochs × 60,000 images |
| pooling | none | – |

Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold −52 mV + θ); pixels → Poisson spikes at 0.1 × intensity per ms, 350 ms per image; each image first scaled to the training set's mean intensity (`--input-norm`, pixels capped at 10); STDP traces τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32; layer 1 frozen after stage A. Scoring: neurons labelled from training images 0–1,999; accuracy on test images 0–999 and on the same images shifted by up to ±2 px, every 200 batches (6,400 images). Other keys in run.json (arm, sweep, init, theta_init, anneal_at, top_k, ...) are fixed placeholders for the shared plotting code.

## Results

- **Held-out / shifted ±2 px, mean of the snapshots of the last 3 epochs: 85.3 / 56.2 %** (best snapshot 88.3 %; last 3 snapshots 85.4 / 55.8 %)
- Per epoch: 74.6 · 83.0 · 84.7 · 85.9 · 84.8 · 85.0
- Layer 1 alone (end of stage A, as the output layer): 67.0 / 47.7 %
- Layer 1's output into layer 2: 3.46 spikes per ms per image
- Run time: 141 min

The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.
