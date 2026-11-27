<!-- written by new_architecture_testing/snn_multi_layered_unshared/describe_runs.py; rerun it to refresh -->
# singlelayer_n400_tp0.5_tau30k_ep6_seed0_l2wta_fmnist_inorm_20261127-024717

single-layer reference (pixels straight into one Diehl & Cook layer) · Fashion-MNIST with per-image input normalisation · seed 0 · 6 epochs × 60,000 images · Unity job `ml8_fm_single_in` on gypsum-gpu015 (Tesla M40 24GB), 71 min

**In short:** 400 neurons reading the 784 pixels directly (the base Diehl & Cook network, row sum 78.4, w_max 1), tie-break on.

**Submitted with** (`unity/submit_multi_layer.sh`, job `ml8_fm_single_in`; every other setting is the default):
```
python multi_layer_unshared.py --theta-tau 30000 --epochs 6 --snapshot-interval 200 --input-norm --results-dir results_2/09_batch8_input_norm_6ep_20261126 --dataset fmnist --no-l1 --l2-wta --seed 0
```
Note in the submit script: *Fashion-MNIST, normalised*

## Settings

| | the one layer |
|---|---|
| reads | all 784 pixels |
| layout | 400 neurons |
| competition (lateral inhibition) | among all 400, 200 mV |
| tie-break (`wta`: same-step winners → only the strongest fires) | on |
| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | 0.5, τ 30,000 images |
| weight scale | row sum 78.4, w_max 1 (not calibrated) |
| STDP rates lr+ / lr− | × 1 = 0.01 / 0.0001 |
| training | 6 epochs × 60,000 images |

Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold −52 mV + θ); pixels → Poisson spikes at 0.1 × intensity per ms, 350 ms per image; each image first scaled to the training set's mean intensity (`--input-norm`, pixels capped at 10); STDP traces τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32. Scoring: neurons labelled from training images 0–1,999; accuracy on test images 0–999 and on the same images shifted by up to ±2 px, every 200 batches (6,400 images). Other keys in run.json (arm, sweep, init, theta_init, anneal_at, top_k, ...) are fixed placeholders for the shared plotting code.

## Results

- **Held-out / shifted ±2 px, mean of the snapshots of the last 3 epochs: 71.1 / 55.3 %** (best snapshot 74.4 %; last 3 snapshots 71.4 / 54.4 %)
- Per epoch: 63.0 · 71.2 · 71.2 · 71.1 · 70.8 · 71.4
- Run time: 71 min

The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.
