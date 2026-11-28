<!-- written by new_architecture_testing/snn_multi_layered_unshared/describe_runs.py; rerun it to refresh -->
# unshared_p7s2f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_rd8_l2wta_k3_drv0.36_lr10_fmnist_inorm_20261128-112346

two-layer network · Fashion-MNIST with per-image input normalisation · seed 0 · 6 epochs × 60,000 images · Unity job `c2_fk3` on gypsum-gpu002 (Tesla M40 24GB), 145 min

**In short:** layer 1 = unshared 7×7 windows at stride 2 px (11×11 windows × 16 features = 1,936 neurons), learned at 200 mV and read at 8 mV; tie-break on. Layer 2 = 400 neurons over all of layer 1, calibrated for 0.36 mV/ms, STDP × 10, lateral inhibition 200 mV, **3 winners per image**.

**Started with** (every setting not on this line is the default):
```
python multi_layer_unshared.py --theta-tau 30000 --epochs 6 --snapshot-interval 200 --results-dir results_for_cnn_replication/02_batch2_parts_and_k_6ep_20261128 --stride 2 --l1-readout-inhibition 8 --l1-wta --l2-wta --l2-drive 0.36 --l2-lr-mult 10 --dataset fmnist --input-norm --l2-k 3 --seed 0
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
| tie-break (`wta`: same-step winners → only the strongest fires) | on | on (replaced by the k-winner rule) |
| winners per image | – (competition is within each window, above) | k = 3: the first 3 neurons to reach threshold share each image (no inhibition among them, none at all until there are 3); every other neuron is silenced |
| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | 0.1, τ 2,000 images, one θ per neuron | 0.5, τ 30,000 images |
| weight scale | row sum 12 (mean 0.245), w_max 1.5 | calibrated for 0.36 mV/ms from layer 1's measured rate: mean 0.0732, row sum 141.8, w_max 0.732 |
| STDP rates lr+ / lr− | 1e-2 / 1e-4 | × 10 = 0.1 / 0.001 |
| training | stage A: up to 30,000 images, stops early once the templates stop changing; froze at 14,080 | stage B: 6 epochs × 60,000 images |
| pooling | none | – |

Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold −52 mV + θ); pixels → Poisson spikes at 0.1 × intensity per ms, 350 ms per image; each image first scaled to the training set's mean intensity (`--input-norm`, pixels capped at 10); STDP traces τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32; layer 1 frozen after stage A. Scoring: neurons labelled from training images 0–1,999; accuracy on test images 0–999 and on the same images shifted by up to ±2 px, every 200 batches (6,400 images). Other keys in run.json (arm, sweep, init, theta_init, anneal_at, top_k, ...) are fixed placeholders for the shared plotting code.

## Results

- **Held-out / shifted ±2 px, mean of the snapshots of the last 3 epochs: 76.0 / 57.4 %** (best snapshot 77.3 %; last 3 snapshots 75.5 / 58.1 %)
- Per epoch: 70.8 · 75.5 · 76.4 · 76.1 · 75.8 · 76.2
- Layer 1 alone (end of stage A, as the output layer): 60.8 / 43.8 %
- Layer 1's output into layer 2: 4.92 spikes per ms per image
- On the 1,000 test digits: 3.0 distinct layer-2 neurons and 13.5 layer-2 spikes per image; 399 of 400 neurons fired; median class purity 0.84 (share of a neuron's spikes from its top digit: ~1 = one-digit prototype) — F16 shows what each neuron fired for
- Run time: 145 min

The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.
