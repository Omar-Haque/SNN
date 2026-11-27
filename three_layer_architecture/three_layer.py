"""
three_layer.py -- cooper/three_layer_architecture: a three-layer STDP network on MNIST.

    pixels --> LAYER 1 (local strokes) --> LAYER 2 (local mid-level features) --> LAYER 3 (400-neuron D&C classifier)

LAYER 1  The two-layer study's best layer 1 (new_architecture_testing/snn_multi_layered_unshared, batches 5-6), imported
         unchanged: 7x7 windows at stride 2 (11x11 = 121 windows) x 16 features, each neuron with its own template and
         theta, 200 mV within-window competition plus the tie-break while it learns. After stage A it is frozen and read out
         at 8 mV (--l1-readout-inhibition), so the features that lose the race keep firing.
LAYER 2  NEW (BlockLayer). Each layer-2 neuron reads a block of --l2-block x --l2-block neighbouring layer-1 windows -- all 16
         feature channels of each, so block^2 x 16 inputs -- and competes only with the other --l2-features neurons of its
         block. Same neuron model, conscience, STDP and row normalisation as layer 1: layer 1's mechanism replayed one level
         up, on layer 1's output instead of pixels. Default: 3x3 blocks (an 11x11-pixel area) at stride 2 on the 11x11 window
         grid -> 5x5 = 25 blocks x 16 = 400 neurons, 144 inputs each. Layer-1 spikes are much sparser than pixel spikes, so
         layer 2's weight scale is calibrated after stage A: an active block starts with the same mean drive (mV/ms) an
         active layer-1 window started with (x --l2-drive-mult); w_max keeps layer 1's w_max / mean-weight ratio (6.1).
LAYER 3  The two-layer study's top layer: the D&C layer (400 neurons, all-to-all on layer 2's output, theta_plus 0.5, theta
         decay tau 30k examples, tie-break), its weight scale calibrated from layer 2's measured output (--l3-drive 0.36
         mV/ms), learning rates x10 (--l3-lr-mult).

TRAINING  stage A: layer 1 alone (the two-layer study's train_layer1), then freeze; stage A2: layer 2 on the frozen layer 1,
          then freeze; stage B: layer 3 for --epochs epochs. Every stage is scored with its own top layer as the output
          (labels from train images 0-1999, accuracy on test images 0-999 and on the same images shifted by up to +-2 px),
          so F1's accuracy curve runs across all three stages; green dashed lines mark the two freezes. At the end, the
          spike-triggered averages (the average pixel patch / image each layer-2 / layer-3 neuron fired for, on the 1,000
          test digits) are saved for F6 / F7.
OPTIONS   --l3-skip: layer 3 also reads layer 1's output (a skip connection). --l1-a2-inhibition: layer 1's inhibition while
          layer 2 learns (default = its readout inhibition), e.g. 200 to train layer 2 on the one-feature-per-window code.

USAGE  python three_layer.py                      # the default configuration
       python three_layer.py --smoke              # tiny settings: checks the whole code path in a few minutes
       python three_layer.py --replot results/<batch>/<run>
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from snntorch import spikegen

HERE: Path = Path(__file__).resolve().parent                         # <repo>/three_layer_architecture
REPO: Path = HERE.parent
TWO_LAYER: Path = REPO / "new_architecture_testing" / "snn_multi_layered_unshared"
sys.path.insert(0, str(TWO_LAYER))
sys.path.insert(0, str(HERE))
# The two-layer study's building blocks, reused unchanged (importing that module loads MNIST from <repo>/data).
from multi_layer_unshared import (FeatureLayer, DiehlAndCookNetwork, assign_neuron_labels, evaluate_network,   # noqa: E402
                                  train_layer1, template_sharpness, cosine_to_init, shifted_test_set, train_dataset, test_dataset)
from three_layer_plots import make_figures                                                                       # noqa: E402


# =====================================================================================================================
# Layer 2: blocks of layer-1 windows
# =====================================================================================================================
class BlockLayer(FeatureLayer):
    """FeatureLayer on layer 1's output instead of pixels. The input is layer 1's spike vector (B, side_in^2 * F_in), index
    p1 * F_in + f1 for window p1 = wy * side_in + wx. Block q = qy * side + qx covers windows (qy*stride + dy, qx*stride + dx)
    for dy, dx in 0..block-1; its K = block^2 * F_in inputs are ordered (dy, dx, f1): k = (dy * block + dx) * F_in + f1.
    Everything else -- per-neuron templates and thetas, within-block competition and tie-break, STDP, row normalisation --
    is FeatureLayer's code, because its forward() only needs the index table idx (P, K) into whatever vector it is given."""
    def __init__(self, side_in: int, F_in: int, block: int = 3, stride: int = 2, n_features: int = 16,
                 theta_plus: float = 0.1, theta_decay_tau_steps: float = 1e7, w_norm: float = 10.0, w_max: float = 1.0,
                 lr_plus: float = 1e-2, lr_minus: float = 1e-4, inhibition_mV: float = 200.0, wta: bool = True):
        # FeatureLayer first builds a placeholder 1x1 'image' geometry; everything geometric is replaced below.
        super().__init__(patch=1, stride=1, n_features=n_features, shared=False, theta_plus=theta_plus,
                         theta_decay_tau_steps=theta_decay_tau_steps, theta_shared=False, w_norm=w_norm, w_max=w_max,
                         lr_plus=lr_plus, lr_minus=lr_minus, inhibition_mV=inhibition_mV, image_side=1, wta=wta)
        self.side_in, self.F_in, self.block = side_in, F_in, block
        self.patch, self.stride = block, stride                          # patch = block size here (in layer-1 windows)
        self.K = block * block * F_in
        self.side = (side_in - block) // stride + 1                      # blocks per row / column
        self.P = self.side * self.side
        self.N = self.P * self.F
        idx = []
        for qy in range(self.side):
            for qx in range(self.side):
                row = []
                for dy in range(block):
                    for dx in range(block):
                        p1 = (qy * stride + dy) * side_in + (qx * stride + dx)       # the layer-1 window at this offset
                        row.extend(p1 * F_in + f for f in range(F_in))              # all its feature channels
                idx.append(row)
        self.idx = torch.tensor(idx, dtype=torch.long)                   # (P, K); replaces FeatureLayer's buffer
        W = torch.empty(self.N, self.K).uniform_(0.01, 0.3)              # same initialisation as layer 1, rescaled by calibration
        W = W * (w_norm / W.sum(dim=1, keepdim=True))
        self.W = nn.Parameter(W, requires_grad=False)                    # row p*F + f = neuron (block p, feature f)
        self.theta = torch.ones(self.N) * self.theta_base


class ThreeLayerNetwork(nn.Module):
    """Layer 1 -> layer 2 -> layer 3, all in the same millisecond (no delays). Exposes the interface the estimators use
    (reset, forward -> layer-3 spikes, num_neurons) and, for the two-layer study's train_layer1, .l1 / .learn_l1 / .learn_l2.
    skip = True (--l3-skip): layer 3 reads layer 1's output as well as layer 2's -- input [layer-1 spikes, layer-2 spikes],
    lines 0..N1-1 from layer 1 and N1.. from layer 2."""
    def __init__(self, l1: FeatureLayer, l2: BlockLayer, l3: DiehlAndCookNetwork, skip: bool = False):
        super().__init__()
        self.l1, self.l2, self.l3, self.skip = l1, l2, l3, skip
        self.learn_l1, self.learn_l2, self.learn_l3 = True, False, False   # each stage switches on its own layer only

    @property
    def num_neurons(self) -> int:
        return self.l3.num_neurons

    def l3_input(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return torch.cat([s1, s2], dim=1) if self.skip else s2

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.l1.reset(batch_size, device); self.l2.reset(batch_size, device); self.l3.reset(batch_size, device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        s1 = self.l1(input_spikes, learning=learning and self.learn_l1)
        s2 = self.l2(s1, learning=learning and self.learn_l2)
        return self.l3(self.l3_input(s1, s2), learning=learning and self.learn_l3)


class StackReadout(nn.Module):
    """The lower layers alone, shaped like an output layer for the estimators: the output is the last layer's spikes.
    Never learns. Used to score layer 2 during stage A2 (layers 1 + 2)."""
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_neurons: int = layers[-1].N

    def reset(self, batch_size: int, device: torch.device) -> None:
        for layer in self.layers: layer.reset(batch_size, device)

    def forward(self, x: torch.Tensor, learning: bool = False) -> torch.Tensor:
        for layer in self.layers: x = layer(x, learning=False)
        return x


def score(readout: nn.Module, loaders: tuple, device: torch.device, num_steps: int):
    """Held-out accuracy with `readout` as the output layer (the two-layer study's estimator): labels from train images
    0-1999, accuracy on test images 0-999 and on their shifted copies. Returns (acc %, shifted acc %, labelled neurons, labels)."""
    assign_loader, eval_loader, shifted_loader = loaders
    labels = assign_neuron_labels(readout, assign_loader, device, num_steps)
    acc = float(evaluate_network(readout, eval_loader, labels, device, num_steps))
    acc_s = float(evaluate_network(readout, shifted_loader, labels, device, num_steps)) if shifted_loader is not None else float("nan")
    return acc, acc_s, int((labels >= 0).sum()), labels


# =====================================================================================================================
# Calibrations (after each freeze)
# =====================================================================================================================
@torch.no_grad()
def calibrate_layer2(model: ThreeLayerNetwork, loader: DataLoader, device: torch.device, num_steps: int, drive_mult: float,
                     n_batches: int = 5) -> dict:
    """Give an active layer-2 block the same initial drive an active layer-1 window started with.
    Layer-1 reference: for every window with any ink, its expected pixel input (0.1 x intensity summed over its 49 pixels,
    spikes/ms) x layer 1's initial mean weight (row sum / 49 = 0.245), averaged over windows with ink.
    Layer 2: the frozen layer 1 (at its readout inhibition) is run on n_batches batches and the input spikes each block
    receives are counted; rate = mean over blocks that received any input, per ms. Mean weight m = reference x drive_mult /
    rate; row sum = m x K; w_max = m x layer 1's w_max / mean-weight ratio (1.5 / 0.245 = 6.1). Rescales the random weights."""
    l1, l2 = model.l1, model.l2
    ref_in = 0.0; ref_n = 0; tot = 0.0; n_active = 0
    for bi, (data, _) in enumerate(loader):
        if bi >= n_batches: break
        B = data.shape[0]; data = data.to(device)
        x = data.view(B, -1) * 0.1                                        # expected pixel spikes per ms
        r = x[:, l1.idx].sum(2)                                           # (B, P1) expected input per window
        ref_in += float(r[r > 0].sum()); ref_n += int((r > 0).sum())
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        l1.reset(B, device)
        cnt = torch.zeros(B, l2.P, device=device)
        for t in range(num_steps):
            s1 = l1(spike_data[t], learning=False)
            cnt += s1[:, l2.idx].sum(2)                                   # input spikes into each block this ms
        tot += float(cnt[cnt > 0].sum()); n_active += int((cnt > 0).sum())
    l1_rate = ref_in / max(ref_n, 1); l1_mean_w = l1.w_norm / l1.K
    target = l1_rate * l1_mean_w * drive_mult
    rate = tot / max(n_active, 1) / num_steps
    m = target / max(rate, 1e-9); ratio = l1.w_max / l1_mean_w
    old = l2.w_norm
    l2.w_norm = m * l2.K; l2.w_max = ratio * m
    l2.W.mul_(l2.w_norm / old)
    info = {"l1_ref_input_rate_per_ms": l1_rate, "l1_ref_mean_weight": l1_mean_w, "l2_target_drive": target,
            "l2_input_rate_per_ms_active_block": rate, "l2_active_blocks_per_image": n_active / max(1, min(n_batches, len(loader)) * loader.batch_size),
            "l2_mean_weight": m, "l2_norm": l2.w_norm, "l2_wmax": l2.w_max}
    print(f"[calibration L2] layer-1 windows with ink started at {l1_rate:.2f} pixel spikes/ms x mean weight {l1_mean_w:.3f} = "
          f"{l1_rate * l1_mean_w:.2f} mV/ms; an active layer-2 block receives {rate:.3f} layer-1 spikes/ms -> mean weight {m:.3f} "
          f"(row sum {l2.w_norm:.1f}, w_max {l2.w_max:.2f}) for {target:.2f} mV/ms")
    return info


@torch.no_grad()
def calibrate_layer3(model: ThreeLayerNetwork, loader: DataLoader, device: torch.device, num_steps: int, target_drive: float,
                     n_batches: int = 5) -> dict:
    """The two-layer study's calibration, one layer up: measure the rate at which the frozen layers 1 + 2 feed layer 3 (input
    spikes per ms per presentation, summed over lines -- layer 1's lines included with --l3-skip) and set mean weight
    m = target_drive / rate; row sum = m x inputs; w_max = 10 m. Rescales the random weights."""
    l1, l2, l3 = model.l1, model.l2, model.l3
    tot = 0.0; pres = 0
    for bi, (data, _) in enumerate(loader):
        if bi >= n_batches: break
        B = data.shape[0]; data = data.to(device)
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        l1.reset(B, device); l2.reset(B, device)
        for t in range(num_steps):
            s1 = l1(spike_data[t], learning=False)
            tot += float(model.l3_input(s1, l2(s1, learning=False)).sum())
        pres += B
    rate = tot / max(1, pres) / num_steps
    m = target_drive / max(rate, 1e-9)
    old = l3.w_norm
    l3.w_norm = m * l3.n_inputs; l3.w_max = 10.0 * m
    l3.synapses.weight.mul_(l3.w_norm / old)
    info = {"l3_input_rate_per_ms": rate, "l3_mean_weight": m, "l3_norm": l3.w_norm, "l3_wmax": l3.w_max, "l3_target_drive": target_drive}
    print(f"[calibration L3] layer 2 -> layer 3: {rate:.2f} spikes/ms per presentation over {l3.n_inputs} lines; mean weight {m:.3f} "
          f"(row sum {l3.w_norm:.1f}, w_max {l3.w_max:.2f}) for {target_drive} mV/ms")
    return info


@torch.no_grad()
def spike_triggered_averages(model: ThreeLayerNetwork, loader: DataLoader, device: torch.device, num_steps: int) -> dict:
    """What actually makes the neurons fire: run the trained network (learning off, readout settings of stage B) on the
    images of `loader` and average, for every layer-2 neuron, the pixel patch of its block area (c x c, c = l1 patch +
    (block - 1) x l1 stride) weighted by how many times it fired on each image; for every layer-3 neuron, the whole 28x28
    image weighted the same way. Unlike F3/F4 (weights drawn back through the layer-1 templates) this involves no
    templates: it is the average input the neuron responded to. Counts = total spikes (0 -> the neuron never fired)."""
    l1, l2, l3 = model.l1, model.l2, model.l3
    c = l1.patch + (l2.block - 1) * l1.stride; o = l2.stride * l1.stride      # block area in pixels, and its step
    s2sum = torch.zeros(l2.P, l2.F, c, c, device=device); n2 = torch.zeros(l2.P, l2.F, device=device)
    s3sum = torch.zeros(l3.num_neurons, 28, 28, device=device); n3 = torch.zeros(l3.num_neurons, device=device)
    for data, _ in loader:
        B = data.shape[0]; data = data.to(device); img = data.view(B, 28, 28)
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        model.reset(B, device)
        c2 = torch.zeros(B, l2.N, device=device); c3 = torch.zeros(B, l3.num_neurons, device=device)
        for t in range(num_steps):
            s1 = l1(spike_data[t], learning=False); s2 = l2(s1, learning=False)
            c2 += s2; c3 += l3(model.l3_input(s1, s2), learning=False)
        patches = torch.stack([img[:, (q // l2.side) * o:(q // l2.side) * o + c, (q % l2.side) * o:(q % l2.side) * o + c]
                               for q in range(l2.P)], dim=1)                       # (B, P2, c, c)
        c2 = c2.view(B, l2.P, l2.F)
        s2sum += torch.einsum("bqf,bqij->qfij", c2, patches); n2 += c2.sum(0)
        s3sum += torch.einsum("bn,bij->nij", c3, img); n3 += c3.sum(0)
    sta2 = s2sum / n2.clamp(min=1.0)[..., None, None]; sta3 = s3sum / n3.clamp(min=1.0)[:, None, None]
    return {"sta2": sta2.view(l2.N, c, c).cpu().numpy().astype(np.float32), "sta2_count": n2.view(-1).cpu().numpy(),
            "sta3": sta3.cpu().numpy().astype(np.float32), "sta3_count": n3.cpu().numpy()}


# =====================================================================================================================
# Stage A2: train layer 2 on the frozen layer 1
# =====================================================================================================================
def train_layer2(model: ThreeLayerNetwork, loader: DataLoader, device: torch.device, cfg: dict, loaders: tuple) -> dict:
    """Layer 2 learns on the frozen layer 1 for at most cfg["l2_examples"] images. Every cfg["l2_snapshot"] batches: how many
    templates have not moved from their initial weights (cosine > 0.95), theta, and per active block (a block that received
    layer-1 input): did it fire, first-spike latency, distinct features that fired, features in its first spiking ms, spikes.
    Stops early when the unmoved count has not shrunk for 5 snapshots and the median theta moved < 0.3 mV. Layer 2 is scored
    as the output layer after the first batch, every cfg["snapshot_interval"] batches and at the freeze."""
    l1, l2 = model.l1, model.l2
    num_steps, snap_every, max_examples = cfg["num_steps"], cfg["l2_snapshot"], cfg["l2_examples"]
    hist: dict = {k: [] for k in ("points", "unmoved", "cos_init_med", "theta_med", "theta_p90", "usage", "active_frac", "latency",
                                  "distinct", "tie", "spikes_per_block", "acc_points", "acc", "acc_shifted")}
    W_init = l2.templates().clone()
    usage = torch.zeros(l2.F, device=device)
    n_in = 0; n_fired = 0; lat_sum = 0.0; distinct_sum = 0.0; tie_sum = 0.0; spk_sum = 0.0
    seen = 0; batch_i = 0; t0 = time.time()
    model.learn_l1, model.learn_l2, model.learn_l3 = False, True, False

    def score_layer2() -> None:
        acc, acc_s, n_lab, _ = score(StackReadout(l1, l2), loaders, device, num_steps)
        hist["acc_points"].append(seen); hist["acc"].append(acc); hist["acc_shifted"].append(acc_s)
        print(f"[layer 2] {seen:>6d} ex | layer 2 as the output layer: held-out accuracy {acc:5.1f}% (shifted {acc_s:5.1f}%), {n_lab}/{l2.N} neurons labelled")

    for data, _ in loader:
        B = data.shape[0]; data = data.to(device)
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        l1.reset(B, device); l2.reset(B, device)
        got_input = torch.zeros(B, l2.P, dtype=torch.bool, device=device)
        fired = torch.zeros(B, l2.P, dtype=torch.bool, device=device)
        first_t = torch.full((B, l2.P), float("nan"), device=device)
        first_group = torch.zeros(B, l2.P, device=device)
        cnt = torch.zeros(B, l2.P, l2.F, device=device)
        for t in range(num_steps):
            s1 = l1(spike_data[t], learning=False)
            got_input |= s1[:, l2.idx].sum(2) > 0
            s = l2(s1, learning=True).view(B, l2.P, l2.F)
            any_f = s.sum(2) > 0
            newly = any_f & ~fired
            first_t = torch.where(newly, torch.full_like(first_t, float(t)), first_t)
            first_group = torch.where(newly, s.sum(2), first_group)
            fired |= any_f
            usage += s.sum(dim=(0, 1)); cnt += s
        act = fired & got_input
        n_in += int(got_input.sum()); n_fired += int(act.sum())
        lat_sum += float(first_t[act].sum()); distinct_sum += float((cnt[act] > 0).sum(1).float().sum())
        tie_sum += float(first_group[act].sum()); spk_sum += float(cnt[act].sum())
        seen += B; batch_i += 1
        if batch_i % snap_every == 0:
            W = l2.templates(); ci = cosine_to_init(W, W_init).numpy(); unmoved = int((ci > 0.95).sum())
            th = l2.theta.detach().float().cpu().numpy()
            nf = max(1, n_fired)
            hist["points"].append(seen); hist["unmoved"].append(unmoved); hist["cos_init_med"].append(float(np.median(ci)))
            hist["theta_med"].append(float(np.median(th))); hist["theta_p90"].append(float(np.percentile(th, 90)))
            hist["usage"].append((usage / max(1, snap_every * B)).cpu().numpy().astype(np.float32))
            hist["active_frac"].append(n_fired / max(1, n_in)); hist["latency"].append(lat_sum / nf)
            hist["distinct"].append(distinct_sum / nf); hist["tie"].append(tie_sum / nf); hist["spikes_per_block"].append(spk_sum / nf)
            print(f"[layer 2] {seen:>6d} ex | templates still at their initial weights {unmoved:4d}/{l2.N} | θ med {np.median(th):6.1f} p90 "
                  f"{np.percentile(th, 90):6.1f} | blocks with input that fired {100 * hist['active_frac'][-1]:4.0f}% latency {hist['latency'][-1]:5.1f} ms | "
                  f"per active block: {hist['spikes_per_block'][-1]:4.1f} spikes from {hist['distinct'][-1]:4.1f} features, {hist['tie'][-1]:4.1f} in its "
                  f"first spiking ms | {(time.time() - t0) / 60:4.1f} min")
            usage.zero_(); n_in = n_fired = 0; lat_sum = distinct_sum = tie_sum = spk_sum = 0.0
            if (cfg["l2_stop_stationary"] and len(hist["unmoved"]) >= 6 and hist["unmoved"][-1] >= hist["unmoved"][-6]
                    and abs(hist["theta_med"][-1] - hist["theta_med"][-6]) < 0.3):
                print(f"[layer 2] stationary: no template left its initial weights for 5 snapshots ({unmoved} unmoved, mostly blocks that "
                      f"never get input) and the median theta moved < 0.3 mV -> freezing at {seen} examples"); break
        if batch_i == 1 or batch_i % cfg["snapshot_interval"] == 0: score_layer2()
        if seen >= max_examples: break
    if not hist["acc_points"] or hist["acc_points"][-1] != seen: score_layer2()    # the curve ends where stage B begins
    model.learn_l2 = False   # FREEZE
    hist["frozen_at"] = seen; hist["minutes"] = (time.time() - t0) / 60
    return hist


# =====================================================================================================================
# Stage B: train layer 3 on the frozen layers 1 + 2
# =====================================================================================================================
def train_layer3(model: ThreeLayerNetwork, train_loader: DataLoader, loaders: tuple, device: torch.device, cfg: dict,
                 run_dir: Path, h1: dict, h2: dict) -> None:
    """Layer 3 learns for cfg["num_epochs"] passes over the training images. Every cfg["snapshot_interval"] batches: held-out and
    shifted accuracy, layer-3 rows still at their initial weights, layer-3 spikes per presentation, median theta. Then writes
    run.npz, run.json and the figures."""
    l1, l2, l3 = model.l1, model.l2, model.l3
    n, E, T, si = l3.num_neurons, cfg["num_epochs"], cfg["num_steps"], cfg["snapshot_interval"]
    k3 = max(3, int(round(l3.n_inputs / 7.84)))
    P, A, S, UM, SPP, TH, WS, SH = [], [], [], [], [], [], [], []
    W_init = l3.synapses.weight.detach().clone()
    spikes_since = torch.zeros(n, device=device); pres_since = 0
    total = 0; gbi = 0; t0 = time.time(); labels = None
    model.learn_l1, model.learn_l2, model.learn_l3 = False, False, True
    for epoch in range(E):
        for data, _ in train_loader:
            B = data.shape[0]; data = data.to(device)
            spike_data = spikegen.rate(data * 0.1, num_steps=T).view(T, B, 784)   # type: ignore
            model.reset(B, device)
            bts = torch.zeros(B, n, device=device)
            for t in range(T):
                bts += model(spike_data[t], learning=True)
            spikes_since += bts.sum(0); pres_since += B; total += B
            if gbi % si == 0:
                acc, acc_s, n_lab, labels = score(model, loaders, device, T)
                W = l3.synapses.weight.detach()
                ci = cosine_to_init(W, W_init).numpy(); sharp = template_sharpness(W, k3).numpy()
                th = l3.neurons.theta.detach().float().cpu().numpy()   # type: ignore
                ws = spikes_since.cpu().numpy().copy()
                P.append(total); A.append(acc); S.append(acc_s); UM.append(int((ci > 0.95).sum()))
                SPP.append(float(ws.sum()) / max(1, pres_since)); TH.append(float(np.median(th))); WS.append(ws.astype(np.float32)); SH.append(float(np.median(sharp)))
                spikes_since.zero_(); pres_since = 0
                print(f"[layer 3] ep {epoch + 1} | {total:>7d} ex | held-out {acc:5.1f}% shifted {acc_s:5.1f}% | rows at initial weights {UM[-1]:3d}/{n} | "
                      f"{SPP[-1]:5.1f} layer-3 spikes per presentation | θ med {TH[-1]:6.1f} | {n_lab}/{n} labelled | {(time.time() - t0) / 60:5.1f} min", flush=True)
            gbi += 1
    print("Training complete")
    sta = spike_triggered_averages(model, loaders[1], device, T)          # on the 1,000 held-out test digits
    W3f = l3.synapses.weight.detach().float()
    share_l2 = float(W3f[:, l1.N:].sum() / W3f.sum()) if model.skip else None    # --l3-skip: layer 3's weight on layer-2 lines
    Pa, Aa, Sa = np.array(P), np.array(A), np.array(S)
    n_train = len(train_loader.dataset)                                  # type: ignore
    ep_acc = [float(Aa[(Pa > e * n_train) & (Pa <= (e + 1) * n_train)].mean()) if ((Pa > e * n_train) & (Pa <= (e + 1) * n_train)).any() else float("nan") for e in range(E)]
    ep_sh = [float(Sa[(Pa > e * n_train) & (Pa <= (e + 1) * n_train)].mean()) if ((Pa > e * n_train) & (Pa <= (e + 1) * n_train)).any() else float("nan") for e in range(E)]
    last3 = (Pa > max(0, E - 3) * n_train)                               # the last three epochs (epochs 4-6 of 6)
    out = dict(cfg); out.update({
        "minutes": (time.time() - t0) / 60 + h1["minutes"] + h2["minutes"],
        "final_accuracy": float(np.mean(Aa[-3:])), "final_accuracy_shifted": float(np.nanmean(Sa[-3:])), "max_accuracy": float(Aa.max()),
        "epoch_accuracy": ep_acc, "epoch_accuracy_shifted": ep_sh,
        "last3_epochs_accuracy": float(Aa[last3].mean()), "last3_epochs_accuracy_shifted": float(np.nanmean(Sa[last3])),
        "final_l3_spikes_per_presentation": SPP[-1], "final_l3_unmoved_rows": UM[-1],
        "l3_weight_share_l2": share_l2, "l3_weight_share_l2_at_start": (l2.N / (l1.N + l2.N)) if model.skip else None,
        "l2_never_fired_on_test": int((sta["sta2_count"] == 0).sum()), "l3_never_fired_on_test": int((sta["sta3_count"] == 0).sum()),
        "l1": {"frozen_at": h1["frozen_at"], "converged_at": h1["converged_at"], "minutes": h1["minutes"], "P": l1.P, "F": l1.F, "K": l1.K, "side": l1.side,
               "readout_accuracy_at_freeze": h1["acc"][-1] if h1["acc"] else None, "readout_accuracy_shifted_at_freeze": h1["acc_shifted"][-1] if h1["acc"] else None},
        "l2": {"frozen_at": h2["frozen_at"], "minutes": h2["minutes"], "P": l2.P, "F": l2.F, "K": l2.K, "side": l2.side, "block": l2.block, "stride": l2.stride,
               "readout_accuracy_at_freeze": h2["acc"][-1] if h2["acc"] else None, "readout_accuracy_shifted_at_freeze": h2["acc_shifted"][-1] if h2["acc"] else None,
               "unmoved_at_freeze": h2["unmoved"][-1] if h2["unmoved"] else None},
    })
    npz = dict(points=Pa, accuracy=Aa, accuracy_shifted=Sa, l3_unmoved=np.array(UM), l3_spikes_per_presentation=np.array(SPP),
               l3_theta_med=np.array(TH), l3_sharp_med=np.array(SH), window_spikes=np.array(WS),
               labels=(labels.cpu().numpy().astype(np.int16) if labels is not None else np.zeros(0, np.int16)),
               W1=l1.templates().float().cpu().numpy(), th1=l1.theta.detach().float().cpu().numpy(),
               W2=l2.templates().float().cpu().numpy(), th2=l2.theta.detach().float().cpu().numpy(),
               W3=l3.synapses.weight.detach().float().cpu().numpy().astype(np.float16), th3=l3.neurons.theta.detach().float().cpu().numpy(),   # type: ignore
               l1_points=np.array(h1["points"]), l1_reservoir=np.array(h1["reservoir"]), l1_theta_med=np.array(h1["theta_med"]),
               l1_active_frac=np.array(h1["active_frac"]), l1_latency=np.array(h1["latency"]), l1_spikes_per_window=np.array(h1["spikes_per_window"]),
               l1_struct_final=(np.array(h1["struct"][-1]) if h1["struct"] else np.zeros(0, np.float32)),
               l1_acc_points=np.array(h1["acc_points"]), l1_accuracy=np.array(h1["acc"]), l1_accuracy_shifted=np.array(h1["acc_shifted"]),
               l2_points=np.array(h2["points"]), l2_unmoved=np.array(h2["unmoved"]), l2_cos_init_med=np.array(h2["cos_init_med"]),
               l2_theta_med=np.array(h2["theta_med"]), l2_usage=np.array(h2["usage"]), l2_active_frac=np.array(h2["active_frac"]),
               l2_latency=np.array(h2["latency"]), l2_distinct=np.array(h2["distinct"]), l2_tie=np.array(h2["tie"]),
               l2_spikes_per_block=np.array(h2["spikes_per_block"]),
               l2_acc_points=np.array(h2["acc_points"]), l2_accuracy=np.array(h2["acc"]), l2_accuracy_shifted=np.array(h2["acc_shifted"]),
               **sta)
    np.savez(run_dir / "run.npz", **npz)
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    make_figures(run_dir)
    try:   # README.md in the run folder: every setting, layer by layer, and the command line (describe_runs.py); never fails the run
        import importlib.util
        _spec = importlib.util.spec_from_file_location("three_layer_describe_runs", HERE / "describe_runs.py")
        _describe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_describe)      # type: ignore[union-attr]
        _describe.write_run_readme(run_dir, sys.argv[1:], torch.cuda.get_device_name(device) if torch.device(device).type == "cuda"
                                   else torch.device(device).type)
    except Exception as e:                                                                                # noqa: BLE001
        print(f"(run README not written: {e})")
    print(f"\nrun finished in {out['minutes']:.1f} min | final acc {out['final_accuracy']:.1f}% shifted {out['final_accuracy_shifted']:.1f}% | "
          f"last-3-epochs average {out['last3_epochs_accuracy']:.1f}% / {out['last3_epochs_accuracy_shifted']:.1f}% | {run_dir}")


# =====================================================================================================================
# main
# =====================================================================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # layer 1 (defaults = the two-layer study's best: stride 2, 8 mV readout)
    p.add_argument("--l1-patch", type=int, default=7); p.add_argument("--l1-stride", type=int, default=2); p.add_argument("--l1-features", type=int, default=16)
    p.add_argument("--l1-theta-plus", type=float, default=0.1); p.add_argument("--l1-theta-tau", type=float, default=2000.0, help="layer-1 theta decay in examples (0 = off)")
    p.add_argument("--l1-norm", type=float, default=12.0); p.add_argument("--l1-wmax", type=float, default=1.5)
    p.add_argument("--l1-inhibition", type=float, default=200.0, help="within-window inhibition while layer 1 learns")
    p.add_argument("--l1-readout-inhibition", type=float, default=8.0, help="within-window inhibition after stage A (8 = the two-layer study's best)")
    p.add_argument("--l1-a2-inhibition", type=float, default=None, help="layer 1's within-window inhibition while layer 2 learns (stage A2 and "
                   "its calibration); default = --l1-readout-inhibition. E.g. 200: layer 2 learns on the sparser one-feature-per-window code, "
                   "and layer 1 switches to the readout inhibition for stage B")
    p.add_argument("--l1-examples", type=int, default=30000); p.add_argument("--l1-snapshot", type=int, default=20)
    p.add_argument("--l1-no-early-stop", dest="l1_stop_stationary", action="store_false"); p.add_argument("--l1-no-wta", dest="l1_wta", action="store_false")
    # layer 2 (new)
    p.add_argument("--l2-block", type=int, default=3, help="block size, in layer-1 windows (3 = an 11x11-pixel area at layer-1 stride 2)")
    p.add_argument("--l2-stride", type=int, default=2, help="block step on the layer-1 window grid")
    p.add_argument("--l2-features", type=int, default=16, help="neurons per block")
    p.add_argument("--l2-theta-plus", type=float, default=0.1); p.add_argument("--l2-theta-tau", type=float, default=2000.0, help="layer-2 theta decay in examples (0 = off)")
    p.add_argument("--l2-drive-mult", type=float, default=1.0, help="layer-2 initial drive relative to layer 1's (1 = an active block starts like an active layer-1 window)")
    p.add_argument("--l2-lr-mult", type=float, default=1.0, help="multiply layer 2's STDP rates (lr+ 1e-2, lr- 1e-4)")
    p.add_argument("--l2-inhibition", type=float, default=200.0, help="within-block inhibition while layer 2 learns")
    p.add_argument("--l2-readout-inhibition", type=float, default=None, help="within-block inhibition after stage A2 (default: keep --l2-inhibition)")
    p.add_argument("--l2-examples", type=int, default=30000); p.add_argument("--l2-snapshot", type=int, default=20)
    p.add_argument("--l2-no-early-stop", dest="l2_stop_stationary", action="store_false"); p.add_argument("--l2-no-wta", dest="l2_wta", action="store_false")
    # layer 3 (defaults = the two-layer study's best top layer)
    p.add_argument("--l3-neurons", type=int, default=400); p.add_argument("--l3-theta-plus", type=float, default=0.5)
    p.add_argument("--l3-theta-tau", type=float, default=30000.0, help="layer-3 theta decay in examples (0 = off)")
    p.add_argument("--l3-inhibition", type=float, default=200.0); p.add_argument("--l3-drive", type=float, default=0.36)
    p.add_argument("--l3-lr-mult", type=float, default=10.0); p.add_argument("--l3-no-wta", dest="l3_wta", action="store_false")
    p.add_argument("--l3-skip", action="store_true", help="layer 3 reads layer 1's output as well as layer 2's (a skip connection)")
    # run
    p.add_argument("--epochs", type=int, default=6); p.add_argument("--shift", type=int, default=2); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=350); p.add_argument("--snapshot-interval", type=int, default=200)
    p.add_argument("--tag", default=""); p.add_argument("--smoke", action="store_true"); p.add_argument("--replot", default=None, metavar="RUN_DIR")
    p.add_argument("--results-dir", default="results", help="folder that receives the run folder, relative to this script's folder or absolute")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="auto = mps if available, then cuda, then cpu")
    args = p.parse_args()
    if args.replot:
        make_figures(args.replot); print("figures regenerated:", args.replot); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device != "auto": device = torch.device(args.device); print(f"Using {args.device} (--device).")
    elif torch.backends.mps.is_available(): device = torch.device("mps"); print("Using MPS.")
    elif torch.cuda.is_available(): device = torch.device("cuda"); print("Using CUDA.")
    else: device = torch.device("cpu"); print("Using CPU.")

    if args.smoke:      # tiny settings: every stage, calibration, scoring, saving and the figures in a few minutes
        train_subset = Subset(train_dataset, range(0, 32 * 20)); assign_subset = Subset(train_dataset, range(0, 128)); eval_subset = Subset(test_dataset, range(0, 64))
        bs_eval = 128; args.epochs, args.steps, args.snapshot_interval = 1, 25, 5
        args.l1_examples, args.l1_snapshot, args.l2_examples, args.l2_snapshot = 32 * 14, 2, 32 * 14, 2; n_shift = 64
    else:
        train_subset = train_dataset; assign_subset = Subset(train_dataset, range(0, 2000)); eval_subset = Subset(test_dataset, range(0, 1000))
        bs_eval = 512; n_shift = 1000
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed))           # stage B
    l1_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1000))     # stage A + calibrations
    l2_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 2000))     # stage A2
    loaders = (DataLoader(assign_subset, batch_size=bs_eval, shuffle=False), DataLoader(eval_subset, batch_size=bs_eval, shuffle=False),
               DataLoader(shifted_test_set(n_shift, args.shift), batch_size=bs_eval, shuffle=False) if args.shift > 0 else None)

    def tau_steps(tau_examples: float) -> float:          # theta decays once per timestep; a batch of 32 images is `steps` timesteps
        return 1e7 if tau_examples <= 0 else tau_examples * args.steps / 32
    l1 = FeatureLayer(args.l1_patch, args.l1_stride, args.l1_features, False, args.l1_theta_plus, tau_steps(args.l1_theta_tau), False,
                      args.l1_norm, args.l1_wmax, 1e-2, 1e-4, args.l1_inhibition, "sum", wta=bool(args.l1_wta))
    if args.l2_block > l1.side or (l1.side - args.l2_block) // args.l2_stride + 1 < 1:
        raise SystemExit(f"--l2-block {args.l2_block} does not fit the {l1.side}x{l1.side} layer-1 window grid")
    l2 = BlockLayer(l1.side, l1.F, args.l2_block, args.l2_stride, args.l2_features, args.l2_theta_plus, tau_steps(args.l2_theta_tau),
                    w_norm=10.0, w_max=1.0, lr_plus=1e-2 * args.l2_lr_mult, lr_minus=1e-4 * args.l2_lr_mult,
                    inhibition_mV=args.l2_inhibition, wta=bool(args.l2_wta))           # weight scale provisional: set by calibrate_layer2
    n_in3 = l1.N + l2.N if args.l3_skip else l2.N
    l3 = DiehlAndCookNetwork(n_in3, args.l3_neurons, args.l3_theta_plus, args.l3_inhibition, tau_steps(args.l3_theta_tau), None,
                             0.1 * n_in3, 1.0, wta=bool(args.l3_wta))                  # weight scale provisional: set by calibrate_layer3
    l3.lr_plus *= args.l3_lr_mult; l3.lr_minus *= args.l3_lr_mult
    model = ThreeLayerNetwork(l1, l2, l3, skip=bool(args.l3_skip)).to(device)

    run_name = (f"3L_l1p{args.l1_patch}s{args.l1_stride}f{args.l1_features}rd{args.l1_readout_inhibition:g}"
                + (f"a2inh{args.l1_a2_inhibition:g}" if args.l1_a2_inhibition is not None else "")
                + f"_l2b{args.l2_block}s{args.l2_stride}f{args.l2_features}")
    if args.l2_readout_inhibition is not None: run_name += f"_l2rd{args.l2_readout_inhibition:g}"
    if args.l2_drive_mult != 1.0: run_name += f"_l2drv{args.l2_drive_mult:g}"
    if args.l2_lr_mult != 1.0: run_name += f"_l2lr{args.l2_lr_mult:g}"
    if args.l2_theta_plus != 0.1: run_name += f"_l2tp{args.l2_theta_plus:g}"
    run_name += f"_l3n{args.l3_neurons}" + ("skip" if args.l3_skip else "")
    if args.l3_drive != 0.36: run_name += f"_l3drv{args.l3_drive:g}"
    if args.l3_lr_mult != 10.0: run_name += f"_l3lr{args.l3_lr_mult:g}"
    run_name += f"_ep{args.epochs}_seed{args.seed}"
    if args.tag: run_name += f"_{args.tag}"
    if args.smoke: run_name += "_SMOKE"
    run_name += "_" + time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.results_dir) if Path(args.results_dir).is_absolute() else HERE / args.results_dir
    run_dir = out_root / run_name; run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {k: v for k, v in vars(args).items() if k != "replot"}
    cfg.update({"run_name": run_name, "num_epochs": args.epochs, "num_steps": args.steps, "batch_size": 32,
                "l1_side": l1.side, "l1_neurons": l1.N, "l2_side": l2.side, "l2_neurons": l2.N, "l2_inputs": l2.K, "l3_inputs": l3.n_inputs})
    print("config:", json.dumps(cfg))
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))

    # ---- stage A: layer 1 (the two-layer study's code), then freeze and switch to the readout inhibition -------------
    model.learn_l1, model.learn_l2, model.learn_l3 = True, False, False
    h1 = train_layer1(model, l1_loader, device, cfg, loaders)              # type: ignore[arg-type]
    l1.inhibition = float(args.l1_a2_inhibition if args.l1_a2_inhibition is not None else args.l1_readout_inhibition)
    print(f"[layer 1] frozen at {h1['frozen_at']} examples; inhibition {l1.inhibition:g} mV while layer 2 learns (was {args.l1_inhibition:g} while layer 1 learned)")
    cfg.update(calibrate_layer2(model, l1_loader, device, args.steps, args.l2_drive_mult))
    # ---- stage A2: layer 2, then freeze and (optionally) switch its readout inhibition ------------------------------
    h2 = train_layer2(model, l2_loader, device, cfg, loaders)
    if l1.inhibition != float(args.l1_readout_inhibition):
        l1.inhibition = float(args.l1_readout_inhibition)
        print(f"[layer 1] readout inhibition {l1.inhibition:g} mV from here on (stage B and evaluation)")
    if args.l2_readout_inhibition is not None:
        l2.inhibition = float(args.l2_readout_inhibition)
        print(f"[layer 2] readout inhibition {l2.inhibition:g} mV (was {args.l2_inhibition:g} while learning)")
    print(f"[layer 2] frozen at {h2['frozen_at']} examples")
    cfg.update(calibrate_layer3(model, l1_loader, device, args.steps, args.l3_drive))
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))       # the config and calibrations survive a crash in stage B
    # ---- stage B: layer 3 --------------------------------------------------------------------------------------------
    train_layer3(model, train_loader, loaders, device, cfg, run_dir, h1, h2)


if __name__ == "__main__":
    main()
