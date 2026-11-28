"""
multi_layer_unshared.py -- new_architecture_testing / snn_multi_layered_unshared: a two-layer STDP network whose
first layer is LOCALLY CONNECTED: every 7x7 window has its own dedicated set of F = 16 neurons, each with its own
template, competing only with the 15 others at its window. 64 windows (stride 3) x 16 = 1,024 layer-1 neurons and
1,024 private templates; nothing is shared or averaged across positions. Layer 2 is the 400-neuron D&C layer reading
the 1,024 layer-1 spike trains.

WHY THIS AND NOT THE SHARED VERSION (what the shared run showed)
  1. With shared templates and updates summed over positions, a template that wins 20 windows of one image is pulled
     toward the AVERAGE of 20 unrelated patches; all 16 templates drift toward the same blob and competition never breaks
     symmetry: after 9,600 images every template was still flat (top-6 share 0.20-0.28). Here each neuron imprints
     exactly the window it won, so layer-1 learning is the single-layer mechanism replayed at 64 positions.
  2. Layer 2 never fired (0.1 spikes per presentation): layer 1 delivers ~0.7 input spikes/ms, not the ~3 the default
     weight scale assumed. This script MEASURES layer 1's output rate after stage A and sets layer 2's weight scale so
     the initial drive equals the pixel network's (--l2-drive 1.2 mV/ms), with w_max = 10 x the mean weight as in the
     base file (0.1 / 1.0).

LAYER 1 (defaults; all flags)
  windows 7x7, stride 3 -> 8x8 = 64 positions, K = 49 inputs; F = 16 neurons per position, each with its own 49
  weights (row sum --l1-norm 12, w_max --l1-wmax 1.5), its own theta (theta_plus 0.1, decay tau 2,000 examples), 200 mV
  inhibition among the 16 at a position only; D&C STDP per neuron (batch-mean). Stage A trains layer 1 alone for up to
  --l1-examples 30,000 (each template sees ONE window per image, so it converges at the single-layer pace, not the
  shared layer's 64x pace) and freezes it once the reservoir of flat templates has stopped shrinking for 5 snapshots
  and theta is stationary. Border windows that never see ink keep flat templates forever; they are reported and
  excluded from nothing -- they simply never fire. While it trains, layer 1 is also scored as the output layer (its 1,024
  neurons labelled and evaluated exactly like layer 2), so F1's accuracy curve covers stage A; a green dashed line in the
  figures marks the freeze, where layer 2 becomes the output.
LAYER 2  the test4 D&C layer on the 1,024 (or pooled 256) lines, theta_plus 0.5, optional decay; evaluation as before
         (held-out 1k test images, plus the same images shifted by up to +-2 px).
USAGE  python multi_layer_unshared.py --seed 0          python multi_layer_unshared.py --pool2       python multi_layer_unshared.py --no-l1
       python multi_layer_unshared.py --smoke           python multi_layer_unshared.py --replot results/<run>
       python multi_layer_unshared.py --dataset fmnist  (Fashion-MNIST instead of MNIST)   --input-norm  (equal mean intensity per image)
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from snntorch import spikegen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # the local multi_layer_plots.py, importable from any CWD
from multi_layer_plots import make_run_figures, spike_time_from_curve

# WHERE THE CODE COMES FROM
#   * excitatory_neurons and DiehlAndCookNetwork: the base file's classes (src/snn_normalized replicate_results.py, via
#     test4.py), with its hard-coded constants turned into constructor arguments; comments marked ADDED / NEW below say
#     exactly what differs.
#   * assign_neuron_labels and evaluate_network: the base file's estimators, unchanged.
#   * Everything else (FeatureLayer, pool2x2, TwoLayerNetwork, the helpers, stage A, the calibration, stage B and main)
#     is new to this script.
#
# Paths. This file lives in <repo>/new_architecture_testing/snn_multi_layered_unshared/, so parents[0] is
# new_architecture_testing and parents[1] is the repo root. MNIST is read from (downloaded to) <repo>/data, the folder
# every script in the repo shares; each run writes its own folder under ./results/ (or under --results-dir, e.g. results_2).
script_dir: Path = Path(__file__).resolve().parent
repo_root: Path = script_dir.parents[1]            # .../cooper
DATA_ROOT: Path = repo_root / "data"
results_root: Path = script_dir / "results"

# Both MNIST splits, loaded once at import time. ToTensor() gives float intensities in [0, 1], shape (1, 28, 28) per image.
train_dataset: datasets.MNIST = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transforms.ToTensor())
test_dataset: datasets.MNIST = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=transforms.ToTensor())


# =====================================================================================================================
# The LIF cell: copied verbatim from test4.py (excitatory_neurons). Used by layer 2; layer 1 re-implements the SAME
# equations inline because its theta may be per feature rather than per neuron (see FeatureLayer.forward).
# =====================================================================================================================
class excitatory_neurons(nn.Module):
    # ADDED vs the base file: the constants the base file hard-codes are constructor arguments here -- theta_plus (base 0.05),
    # theta_decay_tau_steps (base 1e7 timesteps, i.e. no decay in practice), num_neurons (base 400) -- plus two options the base
    # file has switched off: theta_max (a ceiling on theta; base: a commented-out clamp; this script always passes None) and
    # wta (the per-timestep tie-break; base: the commented-out "WTA" block). The rest of the class is the base file's code.
    def __init__(self, theta_plus: float = 0.05, theta_decay_tau_steps: float = 1e7, theta_max: float | None = None,
                 num_neurons: int = 400, wta: bool = False):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = num_neurons
        self.wta: bool = wta   # hard winner-take-all per timestep (the base file's commented-out block); off = baseline behaviour
        # NEW (--l2-k, results_for_cnn_replication): k winners per image. None = off (baseline behaviour). When set, the first k
        # neurons to cross threshold during an image become that image's winners; from then on only they may fire. See _k_winners.
        self.k_winners: int | None = None
        self.winners: torch.Tensor | None = None    # (B, N) bool: the current image's winners so far
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = theta_plus
        self.theta_decay: float = math.exp(-1.0 / theta_decay_tau_steps)
        self.theta_max: float | None = theta_max
        self.v:     torch.Tensor | None = None
        # g_e, g_i, tau_m, decay_ge, decay_gi, E_exc and E_inh are left over from the base file's conductance-based version:
        # forward() never uses them. Inputs are added straight to v as millivolts (current-based).
        self.g_e:   torch.Tensor | None = None
        self.g_i:   torch.Tensor | None = None
        self.theta: torch.Tensor | None = (torch.ones(self.num_neurons) * self.theta_base)
        self.tau_m :   float = 100.0
        self.decay_ge: float = math.exp(-1.0 / 1.0)
        self.decay_gi: float = math.exp(-1.0 / 2.0)
        self.E_rest: float = -65.0  # Baseline voltage for each neuron
        self.E_exc:  float = 0.0
        self.E_inh:  float = -100.0
        self.decay = math.exp(-1.0 / 100)

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.v     = torch.ones(batch_size, self.num_neurons, device=device) * self.E_rest
        self.g_e   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.g_i   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.theta = self.theta.to(device) # type: ignore
        self.winners = torch.zeros(batch_size, self.num_neurons, dtype=torch.bool, device=device)   # NEW (--l2-k): no winners yet

    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor, learning: bool = True) -> torch.Tensor:
        if self.v is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")
        self.v = self.decay * (self.v - self.E_rest) + self.E_rest
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh
        self.raw_v: torch.Tensor = self.v.clone()
        spikes: torch.Tensor = (self.v >= self.theta).float()
        # NEW (--l2-k): with k winners per image, the k-winner rule replaces the one-spike-per-millisecond tie-break below.
        if self.k_winners:
            spikes = self._k_winners(spikes)
        # NEW (the base file's commented-out WTA block, switched on by --l2-wta): keep at most one spike per image per millisecond.
        # A copy of v gets -inf wherever the neuron did not cross threshold, argmax picks the crossing neuron with the highest v,
        # scatter_ builds a one-hot mask, and the product keeps only that spike. If no neuron crossed, argmax returns 0 but that
        # neuron's spike is 0, so the mask changes nothing.
        elif self.wta:
            # Resolve simultaneous threshold crossings: per image, only the spiking neuron with the highest potential fires.
            competing_v = self.v.clone()
            competing_v[spikes == 0.0] = -float("inf")
            max_v_indices: torch.Tensor = torch.argmax(competing_v, dim=1)
            wta_mask = torch.zeros_like(spikes)
            wta_mask.scatter_(1, max_v_indices.unsqueeze(1), 1.0)
            spikes = spikes * wta_mask
        self.v[spikes == 1.0] = self.E_rest
        if learning:
            self.theta += (spikes.sum(dim=0) * self.theta_plus) / 16
            # NEW: optional ceiling on theta (unused here: main() passes theta_max=None).
            if self.theta_max is not None:
                self.theta.clamp_(max=self.theta_max)
        return spikes

    def _k_winners(self, spikes: torch.Tensor) -> torch.Tensor:
        # NEW (--l2-k). An image's winners may always fire. A neuron that crosses threshold and is not yet a winner joins the
        # winners while the image has fewer than k of them; if more cross in the same millisecond than there are free places,
        # the ones with the highest potential take them (the --l2-wta tie-break, generalised to k). Every other crossing is
        # dropped. Together with DiehlAndCookNetwork.forward (no inhibition until the image has its k winners, and winners never
        # inhibit each other) this makes the first k neurons to reach threshold -- the k best-matching templates -- share the image.
        assert self.winners is not None and self.v is not None and self.k_winners is not None
        kmax = min(int(self.k_winners), self.num_neurons)
        free = kmax - self.winners.sum(dim=1)                                      # (B,) places left for this image
        new = (spikes > 0) & ~self.winners                                          # crossed threshold, not yet a winner
        top_v, top_i = torch.topk(self.v.masked_fill(~new, -float("inf")), kmax, dim=1)   # best newcomers first, (B, kmax)
        rank = torch.arange(kmax, device=spikes.device).unsqueeze(0)
        take = (rank < free.unsqueeze(1)) & torch.isfinite(top_v)                   # within the free places, and a real crossing
        admit = torch.zeros_like(spikes).scatter_(1, top_i, take.float())
        self.winners = self.winners | (admit > 0)
        return spikes * self.winners.float()


# =====================================================================================================================
# Layer 2: the Diehl & Cook layer of test4.py, verbatim except that the number of inputs (784), the row-normalisation
# target (78.4) and the weight ceiling (1.0) are constructor arguments.
# =====================================================================================================================
class DiehlAndCookNetwork(nn.Module):
    # ADDED vs the base file: n_inputs (base: 784 pixels), w_norm (base: 78.4, the sum every weight row is rescaled to), w_max
    # (base: 1.0), inhibition_mV (base: 200) and the theta/wta options passed on to excitatory_neurons. In this script n_inputs
    # is 1,024 (one input line per layer-1 neuron), 256 with --pool2, or 784 with --no-l1; w_norm and w_max are set
    # provisionally in main() and then replaced by calibrate_layer2(). Weight initialisation, the STDP rule, the traces and the
    # lateral inhibition are the base file's.
    def __init__(self, n_inputs: int = 784, num_neurons: int = 400, theta_plus: float = 0.05, inhibition_mV: float = 200.0,
                 theta_decay_tau_steps: float = 1e7, theta_max: float | None = None, w_norm: float = 78.4, w_max: float = 1.0, wta: bool = False):
        super(DiehlAndCookNetwork, self).__init__()
        self.n_inputs: int = n_inputs
        self.num_neurons: int = num_neurons
        # synapses.weight has shape (num_neurons, n_inputs) = (400, 1024): row j holds layer-2 neuron j's weight from every layer-1
        # neuron (all-to-all, no spatial structure). synapses(input_spikes) computes input_spikes @ weight.T: for every layer-2
        # neuron, the sum of its weights on the lines that spiked this millisecond, added to v as mV.
        self.synapses: nn.Linear = nn.Linear(self.n_inputs, self.num_neurons, bias=False)
        self.synapses.weight.requires_grad = False # Using STDP instead
        self.w_max:    float = w_max      # weight upper limit (1.0 in the base file)
        self.w_norm:   float = w_norm     # row sum after normalisation (78.4 in the base file)
        self.lr_plus:  float = 1e-2 # Learning rate for potentiation 0.01
        self.lr_minus: float = 1e-4 # Learning rate for depression 0.0001
        self.decay_x: float = math.exp(-1.0 / 20.0) # Pre-synaptic trace decay
        self.decay_y: float = math.exp(-1.0 / 20.0) # Post-synaptic trace decay
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3)
        with torch.no_grad():
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(self.w_norm / weight_sum)
        self.neurons: excitatory_neurons = excitatory_neurons(theta_plus, theta_decay_tau_steps, theta_max, num_neurons, wta)
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * inhibition_mV
        self.inh_matrix.fill_diagonal_(0.0)
        self.prev_spikes: torch.Tensor = torch.empty(0)
        self.x_trace: torch.Tensor = torch.empty(0)
        self.y_trace: torch.Tensor = torch.empty(0)
        # NEW (results_for_cnn_replication), all off by default (baseline behaviour):
        #   explain   --l2-explain: explaining away (needs k winners, --l2-k). Each winner claims the input lines its template
        #             covers; every other neuron only receives the lines no winner has claimed yet, and each winner learns from
        #             what the OTHER winners leave. See _explained_drive / _explained_post.
        #   lat_lr    --l2-lat-lr: learned lateral inhibition. After every batch the inhibition between two neurons grows in
        #             proportion to how often they fired on the same image, and shrinks slowly otherwise (lat_target = the
        #             fraction of neurons that should fire per image; lat_max = ceiling in mV). See update_lateral.
        self.explain: bool = False
        self._explain_state: tuple | None = None
        self.lat_lr: float = 0.0; self.lat_target: float = 0.0125; self.lat_max: float = 200.0

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.neurons.reset(batch_size, device)
        self.inh_matrix  = self.inh_matrix.to(device)
        self.prev_spikes = torch.zeros(batch_size, self.num_neurons, device=device)
        # one presynaptic trace per input line: 1,024 layer-1 lines here (784 pixels in the base file)
        self.x_trace     = torch.zeros(batch_size, self.n_inputs, device=device)
        self.y_trace     = torch.zeros(batch_size, self.num_neurons, device=device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        x_exc: torch.Tensor = self._explained_drive(input_spikes) if self.explain else self.synapses(input_spikes)
        x_inh: torch.Tensor = torch.matmul(self.prev_spikes, self.inh_matrix)
        if self.neurons.k_winners:
            # NEW (--l2-k): winners are never inhibited (by each other); the other neurons only once their image has its k
            # winners -- until then nobody is inhibited, so the next-best neurons can still reach threshold and join.
            w = self.neurons.winners
            assert w is not None
            full = (w.sum(dim=1, keepdim=True) >= self.neurons.k_winners).float()
            x_inh = x_inh * (~w).float() * full
        current_spikes: torch.Tensor = self.neurons(x_exc, x_inh, learning=learning)
        if learning:
            self.x_trace = (self.x_trace * self.decay_x) + input_spikes
            self.y_trace = (self.y_trace * self.decay_y) + current_spikes
            batch_size: int = input_spikes.shape[0]
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace) / batch_size
            if self.explain:   # NEW (--l2-explain): each spiking winner potentiates only on what the other winners leave
                post_interaction = self._explained_post(current_spikes) / batch_size
            delta_w_plus: torch.Tensor = (self.lr_plus ) * post_interaction * (self.w_max - self.synapses.weight)
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes) / batch_size
            delta_w_minus = (self.lr_minus ) * pre_interaction * self.synapses.weight
            self.synapses.weight.add_(delta_w_plus - delta_w_minus)
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(self.w_norm / weight_sum)
        self.prev_spikes = current_spikes
        return current_spikes

    # ---- NEW (results_for_cnn_replication): explaining away and learned lateral inhibition --------------------------------
    def _unit_templates(self) -> torch.Tensor:
        # each neuron's weights scaled so its strongest line is 1: "how much of line j neuron i claims when it wins"
        W = self.synapses.weight
        return W / W.max(dim=1, keepdim=True).values.clamp(min=1e-12)

    def _explained_drive(self, input_spikes: torch.Tensor) -> torch.Tensor:
        # Explaining away. S = the sum of the current winners' unit templates (B, n_inputs): how much of each input line the
        # winners already account for. A non-winner receives each line's spikes scaled by (1 - S), i.e. only what no winner
        # has claimed, so the next winner is the neuron that best matches the REST of the image. A winner receives each line
        # scaled by (1 - S + its own claim): its own template is not removed for it, the other winners' are. Winners are the
        # k-winner set of excitatory_neurons as of the previous millisecond (none at the start of an image: everyone sees
        # everything, as without this option).
        w = self.neurons.winners
        assert w is not None
        T = self._unit_templates()
        S = w.float() @ T
        x_exc = self.synapses(input_spikes * (1.0 - S).clamp(min=0.0))
        b, i = w.nonzero(as_tuple=True)
        if b.numel():
            g = (1.0 - S[b] + T[i]).clamp(0.0, 1.0)
            x_exc[b, i] = (input_spikes[b] * g * self.synapses.weight[i]).sum(dim=1)
        self._explain_state = (S, T, w.clone())
        return x_exc

    def _explained_post(self, current_spikes: torch.Tensor) -> torch.Tensor:
        # The potentiation term of STDP under explaining away: for each neuron that spiked this millisecond, its presynaptic
        # traces times the share of each line the OTHER winners leave (a neuron that joined this millisecond: what all earlier
        # winners leave). Unscaled by the batch size (the caller divides). Shape (N, n_inputs), like matmul(spikes.T, x_trace).
        assert self._explain_state is not None
        S, T, w_prev = self._explain_state
        post = torch.zeros_like(self.synapses.weight)
        b, i = current_spikes.nonzero(as_tuple=True)
        if b.numel():
            g = (1.0 - S[b] + T[i] * w_prev[b, i].float().unsqueeze(1)).clamp(0.0, 1.0)
            post.index_add_(0, i, self.x_trace[b] * g)
        return post

    @torch.no_grad()
    def update_lateral(self, counts: torch.Tensor) -> None:
        # Learned lateral inhibition (anti-Hebbian, after Foldiak 1990), once per batch. counts = (B, N) spikes per image.
        # co[i, j] = fraction of the batch's images on which i and j both fired. inhibition[i, j] += lat_lr x (co - target^2):
        # pairs that fire together more often than two independent neurons at the target rate would inhibit each other more,
        # pushing them toward different images or different parts of an image; the rest slowly lose inhibition. Kept in
        # [0, lat_max] mV, no self-inhibition.
        y = (counts > 0).float()
        co = y.t() @ y / max(1, y.shape[0])
        self.inh_matrix += self.lat_lr * (co - self.lat_target ** 2)
        self.inh_matrix.clamp_(min=0.0, max=self.lat_max)
        self.inh_matrix.fill_diagonal_(0.0)


# =====================================================================================================================
# Layer 1: the feature layer (new)
# =====================================================================================================================
class FeatureLayer(nn.Module):
    """P window positions x F feature neurons on a 28x28 input. See the module docstring for the design.

    Shapes: patches (B, P, K); templates W (F, K) if shared else (P*F, K); membrane v, spikes, traces (B, P, F);
    theta (P*F,) per neuron or (F,) per feature. Output: spikes flattened to (B, P*F), neuron index = p*F + f.
    """
    # Constructor arguments (main() passes the command-line values):
    #   patch, stride          window size and step on the 28x28 image (7, 3 -> 8x8 = 64 windows; neighbours overlap by 4 px)
    #   n_features (F)         neurons per window (16)
    #   shared                 True = one set of F templates used at every window (the failed shared design);
    #                          False = every (window, feature) neuron has its own template (the default from the command line)
    #   theta_plus             adaptive-threshold increment, applied as theta_plus / 16 per spike like the base file (0.1)
    #   theta_decay_tau_steps  time constant of theta's relaxation toward -52 mV, in TIMESTEPS (main converts from examples)
    #   theta_shared           one theta per feature index, shared by all windows, instead of one per neuron (off by default)
    #   w_norm, w_max          each template is rescaled to sum to w_norm after every update and clamped to [0, w_max] (12, 1.5)
    #   lr_plus, lr_minus      STDP rates, the base file's 1e-2 and 1e-4
    #   inhibition_mV          what each spike costs the OTHER features of the same window on the next step (200)
    #   stdp_reduce            shared only: sum the updates of all windows ("sum") or average them ("mean")
    #   wta                    per-window tie-break, see forward()
    def __init__(self, patch: int = 7, stride: int = 3, n_features: int = 16, shared: bool = True,
                 theta_plus: float = 0.1, theta_decay_tau_steps: float = 1e7, theta_shared: bool = False,
                 w_norm: float = 9.8, w_max: float = 1.0, lr_plus: float = 1e-2, lr_minus: float = 1e-4,
                 inhibition_mV: float = 200.0, stdp_reduce: str = "sum", image_side: int = 28, wta: bool = False):
        super().__init__()
        self.patch, self.stride, self.F, self.shared = patch, stride, n_features, shared
        self.wta: bool = wta   # within-position tie-break: when several features of a window cross in the same step, only the highest fires
        # K = pixels per window (49); side = windows per row/column = (28 - 7) // 3 + 1 = 8; P = 64 windows; N = P*F = 1,024 neurons.
        self.K: int = patch * patch
        self.side: int = (image_side - patch) // stride + 1
        self.P: int = self.side * self.side
        self.N: int = self.P * self.F
        self.theta_shared: bool = theta_shared
        self.w_norm, self.w_max, self.lr_plus, self.lr_minus = w_norm, w_max, lr_plus, lr_minus
        self.inhibition: float = inhibition_mV
        self.stdp_reduce: str = stdp_reduce
        # index table: idx[p, k] = flat pixel index (row-major 28x28) of the k-th pixel of window p
        idx = []
        # Window p = py * side + px (row-major over the 8x8 grid) has its top-left pixel at image row py*stride, column px*stride,
        # i.e. flat index base = (py*stride)*28 + px*stride. Its 49 pixels, row-major inside the window, are base + r*28 + c for
        # r, c in 0..6. E.g. window 0 covers rows 0-6 / cols 0-6, window 1 rows 0-6 / cols 3-9, window 8 rows 3-9 / cols 0-6.
        for py in range(self.side):
            for px in range(self.side):
                base = (py * stride) * image_side + px * stride
                idx.append([base + r * image_side + c for r in range(patch) for c in range(patch)])
        # a buffer, not a parameter: it moves with .to(device) and is saved with the model, but is never trained
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))          # (P, K)
        # Initial templates: uniform in [0.01, 0.3], then each row rescaled to sum to w_norm (12 -> mean weight 12 / 49 = 0.245):
        # the base file's initialisation with a different row sum. W has F rows if shared, N = 1,024 rows if unshared, where row
        # p*F + f belongs to neuron (window p, feature f). It is an nn.Parameter only so that .to(device) and saving work; it has
        # no gradient and is changed only by the STDP code in forward().
        n_templates = self.F if shared else self.N
        W = torch.empty(n_templates, self.K).uniform_(0.01, 0.3)
        W = W * (w_norm / W.sum(dim=1, keepdim=True))
        self.W: nn.Parameter = nn.Parameter(W, requires_grad=False)
        # the same cell constants as excitatory_neurons
        self.theta_base: float = -52.0
        self.theta_plus: float = theta_plus
        self.theta_decay: float = math.exp(-1.0 / theta_decay_tau_steps)
        self.E_rest: float = -65.0
        self.decay: float = math.exp(-1.0 / 100)
        self.decay_x: float = math.exp(-1.0 / 20.0)
        self.decay_y: float = math.exp(-1.0 / 20.0)
        # theta = the adaptive threshold (absolute, in mV): one per neuron (N,), or one per feature (F,) with theta_shared.
        # Unlike v it is NOT reset between batches: it is the long-term memory of how often each neuron has fired.
        self.theta: torch.Tensor = torch.ones(self.F if theta_shared else self.N) * self.theta_base
        # per-presentation state; reset() creates these with the right batch size
        self.v: torch.Tensor = torch.empty(0); self.prev_spikes: torch.Tensor = torch.empty(0)
        self.x_trace: torch.Tensor = torch.empty(0); self.y_trace: torch.Tensor = torch.empty(0)

    @property
    def num_neurons(self) -> int:
        return self.N

    def templates(self) -> torch.Tensor:
        """(n_templates, K) weights; the template of neuron (p, f) is row f (shared) or row p*F+f (unshared)."""
        return self.W.detach()

    def reset(self, batch_size: int, device: torch.device) -> None:
        # Called before every batch (= every 350-ms presentation). theta keeps its value; everything else restarts:
        #   v (B, P, F)            membrane potential of every neuron for every image, at rest (-65 mV)
        #   prev_spikes (B, P, F)  last step's spikes, used for the within-window inhibition
        #   x_trace (B, P, K)      presynaptic trace of the 49 pixels of each window (a pixel inside several windows has one copy per
        #                          window; the copies are identical, this layout just keeps the per-window einsums simple)
        #   y_trace (B, P, F)      postsynaptic trace of every neuron
        self.theta = self.theta.to(device)
        self.v = torch.ones(batch_size, self.P, self.F, device=device) * self.E_rest
        self.prev_spikes = torch.zeros(batch_size, self.P, self.F, device=device)
        self.x_trace = torch.zeros(batch_size, self.P, self.K, device=device)
        self.y_trace = torch.zeros(batch_size, self.P, self.F, device=device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        # One millisecond. input_spikes: (B, 784) of 0/1, the Poisson pixel spikes of this ms for each image in the batch.
        # Returns the (B, N) 0/1 spikes of the 1,024 neurons, neuron index p*F + f.
        B = input_spikes.shape[0]
        # Gather each window's pixels: advanced indexing with idx (P, K) turns (B, 784) into (B, P, K).
        patches = input_spikes[:, self.idx]                                            # (B, P, K): the 49 pixels of each window
        # Feed-forward drive in mV for this ms. Unshared: x_exc[b, p, f] = sum over k of patches[b, p, k] * W[p*F + f, k], i.e. each
        # neuron sees only its own window's 49 pixels through its own template (W viewed as (P, F, K), einsum over k).
        # Shared: the same F templates are applied to all 64 windows.
        if self.shared:
            x_exc = torch.matmul(patches, self.W.t())                                  # (B, P, F): template f against every window
        else:
            x_exc = torch.einsum("bpk,pfk->bpf", patches, self.W.view(self.P, self.F, self.K))
        # lateral inhibition WITHIN a position: a neuron is hit by every other feature at its position that fired last step
        # ps.sum(dim=2) counts the features of each window that spiked last step; subtracting ps removes a neuron's own spike. So a
        # neuron that fired alone gets no inhibition and the 15 others in its window get -200 mV each. Windows never inhibit each
        # other. Because inhibition comes from the PREVIOUS step, features that cross threshold in the same ms all fire (a tie).
        ps = self.prev_spikes
        x_inh = self.inhibition * (ps.sum(dim=2, keepdim=True) - ps)                  # (B, P, F)
        # ---- LIF, same equations as excitatory_neurons.forward ----
        # Leak: v relaxes toward E_rest = -65 mV with tau_m = 100 ms (one step = 1 ms). theta relaxes toward -52 mV with
        # theta_decay_tau_steps, but only while learning, so a frozen layer 1 keeps fixed thresholds. Then this ms's input is added.
        self.v = self.decay * (self.v - self.E_rest) + self.E_rest
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh
        # theta reshaped to broadcast over the batch: (1, 1, F) with one theta per feature, (1, P, F) with one per neuron
        theta_b = self.theta.view(1, 1, self.F) if self.theta_shared else self.theta.view(1, self.P, self.F)
        spikes = (self.v >= theta_b).float()
        # Mechanics of the tie-break: non-spiking entries of a copy of v are set to -inf, argmax over the feature axis picks the
        # highest crossing feature per (image, window) -> (B, P), scatter_ builds a one-hot mask over the 16 features, and the
        # product keeps only that spike. In a window where nothing crossed, argmax gives 0 but that spike is 0, so nothing changes.
        if self.wta:
            # The 16 features of a window see the identical input and inhibition only arrives next step, so once their
            # templates are similar they cross threshold in the same millisecond and all fire ("volley"). Resolve it
            # the way the layer-2 option does: per window, only the crossing feature with the highest potential fires.
            competing = self.v.clone(); competing[spikes == 0.0] = -float("inf")
            top_f = torch.argmax(competing, dim=2)                                       # (B, P)
            mask = torch.zeros_like(spikes); mask.scatter_(2, top_f.unsqueeze(2), 1.0)
            spikes = spikes * mask
        # neurons that spiked go back to rest (no refractory period, as in the base file)
        self.v[spikes == 1.0] = self.E_rest
        if learning:
            # Adaptive threshold: each spike raises its neuron's theta by theta_plus / 16 (spikes are summed over the batch first, as in
            # the base file). With theta_shared the increment is the feature's spikes averaged over the 64 windows.
            if self.theta_shared:
                self.theta += spikes.sum(dim=0).mean(dim=0) * self.theta_plus / 16      # (F,): the feature's spikes, averaged over positions
            else:
                self.theta += spikes.sum(dim=0).reshape(-1) * self.theta_plus / 16     # (P*F,): each neuron's own spikes
            # ---- STDP, the D&C rule per neuron, reduced into the templates ----
            # Traces, as in the base file: decay by e^(-1/20) per ms (tau 20 ms), +1 for every spike this ms.
            # x_trace = how recently each pixel of each window fired; y_trace = how recently each layer-1 neuron fired.
            self.x_trace = self.x_trace * self.decay_x + patches
            self.y_trace = self.y_trace * self.decay_y + spikes
            # Shared: the same rule, but the requests of every window where feature f fired are summed into ONE template (F, K) -- what
            # averaged unrelated patches into a blob in the shared run. div = B, or B * P with stdp_reduce = "mean".
            if self.shared:
                post = torch.einsum("bpf,bpk->fk", spikes, self.x_trace)              # (F, K): potentiation requests, summed over positions
                pre = torch.einsum("bpf,bpk->fk", self.y_trace, patches)
                div = B * (self.P if self.stdp_reduce == "mean" else 1)
                W = self.W
                dW = self.lr_plus * (post / div) * (self.w_max - W) - self.lr_minus * (pre / div) * W
            else:
                # Unshared (the default). For each neuron (p, f), averaged over the batch:
                #   post[p, f, k] = sum_b spikes[b, p, f] * x_trace[b, p, k]   when the neuron spikes, credit the pixels of ITS window that
                #                                                               fired recently (potentiation)
                #   pre[p, f, k]  = sum_b y_trace[b, p, f] * patches[b, p, k]   when a pixel of its window spikes, debit it in proportion to
                #                                                               how recently the neuron fired (depression)
                #   dW = lr_plus * post * (w_max - W) - lr_minus * pre * W      the base file's rule, neuron by neuron
                # A neuron changes only from its own window's activity, so its template imprints exactly the patches it wins.
                post = torch.einsum("bpf,bpk->pfk", spikes, self.x_trace) / B          # (P, F, K): each neuron its own
                pre = torch.einsum("bpf,bpk->pfk", self.y_trace, patches) / B
                W = self.W.view(self.P, self.F, self.K)
                dW = (self.lr_plus * post * (self.w_max - W) - self.lr_minus * pre * W).view(self.N, self.K)
            # Apply (no gradient tracking, W is a Parameter): add the update, clamp to [0, w_max], and rescale every template so its 49
            # weights sum to w_norm again (the base file's normalisation, with 12 instead of 78.4).
            with torch.no_grad():
                self.W.add_(dW)
                self.W.clamp_(min=0.0, max=self.w_max)
                self.W.mul_(self.w_norm / self.W.sum(dim=1, keepdim=True))
        # keep this step's spikes for next step's inhibition; return them flattened to (B, P*F), index p*F + f
        self.prev_spikes = spikes
        return spikes.reshape(B, self.N)


def pool2x2(spikes: torch.Tensor, side: int, F: int) -> torch.Tensor:
    """Spike-OR over 2x2 blocks of positions per feature: (B, side*side*F) -> (B, (side//2)*(side//2)*F)."""
    # spikes: (B, side*side*F), index (py*side + px)*F + f. View as (B, side, side, F); crop to an even side (a no-op for the
    # default side 8; with an odd side the cropped tensor would need .contiguous() before the next .view); split each spatial
    # axis into (side/2 blocks, 2 within a block) and take the max over the two within-block axes. The max of 0/1 values is a
    # logical OR: pooled feature f of block (Y, X) spikes this ms if f spiked in any of the windows (2Y..2Y+1, 2X..2X+1).
    # 8x8 windows x 16 features = 1,024 lines -> 4x4 x 16 = 256 lines.
    B = spikes.shape[0]
    s = spikes.view(B, side, side, F)[:, : side // 2 * 2, : side // 2 * 2, :]
    s = s.view(B, side // 2, 2, side // 2, 2, F).amax(dim=(2, 4))
    return s.reshape(B, -1)


class TwoLayerNetwork(nn.Module):
    """Layer 1 (optional) -> optional 2x2 pooling -> layer 2. Exposes the single-layer interface the estimators use:
    reset(batch_size, device), forward(input_spikes, learning) -> layer-2 spikes, num_neurons, synapses (layer 2)."""
    # Wraps layer 1 (None for --no-l1), the optional pooling and layer 2, so that the rest of the code -- including the base
    # file's estimators, written for the single-layer network -- can treat the whole thing as one network.
    def __init__(self, l1: FeatureLayer | None, l2: DiehlAndCookNetwork, pool2: bool = False):
        super().__init__()
        self.l1, self.l2, self.pool2 = l1, l2, pool2
        # learn_l1 / learn_l2 switch learning per layer: stage A trains layer 1 only (learn_l1 True, learn_l2 False); after the
        # freeze, stage B trains layer 2 only. The estimators also pass learning=False, which switches both off.
        self.learn_l1: bool = l1 is not None     # stage A: True; frozen afterwards
        self.learn_l2: bool = True
        self.last_l1_spikes: torch.Tensor | None = None   # (B, N1) of the last timestep (for the stage-B usage stats)

    # The estimators and the plotting code read model.num_neurons / model.synapses / model.neurons; these forward to layer 2,
    # so they see the same interface as the single-layer DiehlAndCookNetwork.
    @property
    def num_neurons(self) -> int:
        return self.l2.num_neurons

    @property
    def synapses(self):
        return self.l2.synapses

    @property
    def neurons(self):
        return self.l2.neurons

    # input lines into layer 2: 784 pixels without layer 1, (side/2)^2 * F with pooling, else P * F (1,024)
    def l2_input_size(self) -> int:
        if self.l1 is None: return 784
        return (self.l1.side // 2) ** 2 * self.l1.F if self.pool2 else self.l1.N

    def reset(self, batch_size: int, device: torch.device) -> None:
        if self.l1 is not None: self.l1.reset(batch_size, device)
        self.l2.reset(batch_size, device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        # One millisecond of the whole network. Layer 1 turns this ms's pixel spikes into this ms's layer-1 spikes (B, 1,024), and
        # layer 2 receives them in the SAME step: there is no synaptic delay between the layers. Each layer learns only if the
        # caller asked for learning AND its own learn_* flag is on.
        x = input_spikes
        if self.l1 is not None:
            x = self.l1(x, learning=learning and self.learn_l1)
            self.last_l1_spikes = x
            if self.pool2: x = pool2x2(x, self.l1.side, self.l1.F)
        return self.l2(x, learning=learning and self.learn_l2)


# =====================================================================================================================
# Estimators: copied verbatim from test4.py / hypo_1.py (they only use model.reset, model(...), model.num_neurons)
# =====================================================================================================================
@torch.no_grad()
def assign_neuron_labels(model, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    model.eval()
    spike_counts: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        model.reset(batch_size, device)
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
        for digit in range(10):
            mask = (targets == digit)
            if mask.any():
                spike_counts[:, digit] += batch_neuron_spikes[mask].sum(dim=0) / mask.sum().float()
    neuron_assignments: torch.Tensor = torch.argmax(spike_counts, dim=1)
    total_spikes = spike_counts.sum(dim=1)
    neuron_assignments[total_spikes == 0] = -1
    return neuron_assignments


@torch.no_grad()
def evaluate_network(model, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int,
                     return_matrix: bool = False):
    model.eval()
    correct_predictions: int = 0
    total_predictions:   int = 0
    conf_matrix = torch.zeros(10, 10, dtype=torch.int32, device=device)
    assignment_matrix: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    valid_mask = (neuron_assignments != -1)
    safe_assignments = torch.where(valid_mask, neuron_assignments, torch.zeros_like(neuron_assignments))
    assignment_matrix.scatter_(1, safe_assignments.unsqueeze(1), 1.0)
    assignment_matrix[~valid_mask] = 0.0
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        model.reset(batch_size, device)
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
        class_votes: torch.Tensor = torch.matmul(batch_neuron_spikes, assignment_matrix)
        predictions: torch.Tensor = torch.argmax(class_votes, dim=1)
        for true_label, predicted_label in zip(targets, predictions):
            conf_matrix[true_label.long(), predicted_label.long()] += 1
        correct_predictions += (predictions == targets).sum().item()
        total_predictions += batch_size
    accuracy = (correct_predictions / total_predictions) * 100.0
    if return_matrix:
        return accuracy, conf_matrix.cpu().numpy() # type: ignore
    return accuracy


@torch.no_grad()
def layer2_test_stats(model, data_loader: DataLoader, device: torch.device, num_steps: int) -> dict:
    # NEW (results_for_cnn_replication): one pass over the scoring images (test 0-999) with learning off, keeping each image's
    # layer-2 spike counts. Returns
    #   sta2 (N, 784)        spike-triggered average: the mean input image each neuron fired for, weighted by its spike count
    #                        (whole-digit prototypes look like digits here; part-detectors show a stroke or loop)
    #   sta2_count (N,)      total spikes of each neuron on these images (0 = never fired)
    #   class_counts2 (N,10) each neuron's spikes per true class: class purity = top class / total (~1 for a one-digit
    #                        prototype, lower for a part several digits share)
    #   distinct_test (I,)   distinct layer-2 neurons that fired on each image; spikes_test (I,) its layer-2 spikes
    model.eval()
    n = model.num_neurons
    sta = torch.zeros(n, 784, device=device); cnt = torch.zeros(n, device=device); cls = torch.zeros(n, 10, device=device)
    distinct, spikes = [], []
    for data, targets in data_loader:
        B = data.shape[0]; data, targets = data.to(device), targets.to(device)
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        model.reset(B, device)
        counts = torch.zeros(B, n, device=device)
        for t in range(num_steps):
            counts += model(spike_data[t], learning=False)
        sta += counts.t() @ data.view(B, 784).float()
        cnt += counts.sum(0)
        onehot = torch.zeros(B, 10, device=device); onehot[torch.arange(B, device=device), targets.long()] = 1.0
        cls += counts.t() @ onehot
        distinct.append((counts > 0).sum(1).float().cpu()); spikes.append(counts.sum(1).cpu())
    sta = sta / cnt.clamp(min=1).unsqueeze(1)
    return {"sta2": sta.cpu().numpy(), "sta2_count": cnt.cpu().numpy(), "class_counts2": cls.cpu().numpy(),
            "distinct_test": torch.cat(distinct).numpy(), "spikes_test": torch.cat(spikes).numpy()}


# =====================================================================================================================
# Helpers
# =====================================================================================================================
def template_sharpness(W: torch.Tensor, k: int) -> torch.Tensor:
    """Share of each row's mass on its k largest entries (CPU, float32). k = n_inputs / 7.84 keeps the flat-row value at ~0.23."""
    # (rows, inputs) -> for each row, the sum of its k largest weights / the sum of all its weights. For 784-pixel templates
    # k = 100: a fresh uniform row scores ~0.23, a learned digit template 0.4-0.8. Used for layer 2 (k = 131 for 1,024 inputs)
    # and still logged for layer 1 (k = 6) for comparison with earlier runs.
    Wc = W.detach().float().cpu()
    return Wc.topk(k, dim=1).values.sum(dim=1) / Wc.sum(dim=1)


def template_structure(W: torch.Tensor, side: int) -> torch.Tensor:
    """Spatial structure of each side x side template: the 1-pixel autocorrelation of the mean-centred template (average of
    the vertical and horizontal shifts). A random template scores ~0 (p95 ≈ 0.16); a learned stroke band scores 0.5-0.9.
    This replaces the top-k share for layer 1: a stroke band 3-4 px wide covers ~20 of 49 pixels, so its top-6 share can never
    reach the 0.35 that marks a trained pixel template, which made every band look "flat"."""
    # T: (rows, side, side), each template minus its own mean. cos(a, b) is the cosine similarity of two stacks of images.
    # T[:, 1:, :] vs T[:, :-1, :] compares every pixel with the one above it; T[:, :, 1:] vs T[:, :, :-1] with the one to its
    # left. Smooth strokes -> neighbouring weights alike -> high score; independent random weights -> ~0.
    T = W.detach().float().cpu().view(-1, side, side); T = T - T.mean(dim=(1, 2), keepdim=True)
    def cos(a, b): return (a * b).sum(dim=(1, 2)) / ((a * a).sum(dim=(1, 2)).sqrt() * (b * b).sum(dim=(1, 2)).sqrt() + 1e-12)
    return 0.5 * (cos(T[:, 1:, :], T[:, :-1, :]) + cos(T[:, :, 1:], T[:, :, :-1]))


def cosine_to_init(W: torch.Tensor, W_init: torch.Tensor) -> torch.Tensor:
    """Cosine between each row and the row it started from: 1.0 = never changed. Criterion-free 'has this neuron learned',
    for inputs (the layer-1 code) whose sparsity caps the top-k share far below the pixel-template value."""
    # both (rows, inputs); one cosine per row, returned on the CPU. Two all-positive rows are never orthogonal, so even a fully
    # learned row keeps a cosine of ~0.4-0.6 to its random start; the 0.95 cut used in stage B only asks "has it moved at all".
    a = W.detach().float(); b = W_init.detach().float().to(a.device)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).cpu()


def shifted_test_set(n: int, shift: int, seed: int = 1234, dataset=None) -> TensorDataset:
    """The first n test images, each translated by a fixed random (dx, dy) in [-shift, shift]^2 (zero fill).
    dataset: the test set to shift (default: MNIST's; main() passes Fashion-MNIST's with --dataset fmnist)."""
    ds = test_dataset if dataset is None else dataset
    # fixed seed: every snapshot of every run is evaluated on the same 1,000 shifted images
    g = np.random.default_rng(seed)
    imgs = torch.stack([ds[i][0] for i in range(n)])             # (n, 1, 28, 28)
    labels = torch.tensor([int(ds[i][1]) for i in range(n)])
    out = torch.zeros_like(imgs)
    for i in range(n):
        # For dy > 0 the digit moves down: destination rows dy..27 receive source rows 0..27-dy (dx likewise for columns), and the
        # rows/columns left uncovered stay 0. Up to --shift (2) pixels in each direction.
        dx, dy = int(g.integers(-shift, shift + 1)), int(g.integers(-shift, shift + 1))
        src = imgs[i, 0]
        ys, ye = max(0, dy), min(28, 28 + dy); xs, xe = max(0, dx), min(28, 28 + dx)
        out[i, 0, ys:ye, xs:xe] = src[ys - dy:ye - dy, xs - dx:xe - dx]
    return TensorDataset(out, labels)


# =====================================================================================================================
# Stage A: train layer 1 alone
# =====================================================================================================================
class Layer1Readout(nn.Module):
    """Layer 1 on its own, shaped like an output layer for the base file's estimators (they only use reset(), forward(x,
    learning=...) and num_neurons). During stage A layer 1 is the network's last trained layer, so this is how it is scored."""
    def __init__(self, l1: FeatureLayer):
        super().__init__()
        self.l1 = l1
        self.num_neurons: int = l1.N          # 1,024: every (window, feature) neuron gets its own label

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.l1.reset(batch_size, device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = False) -> torch.Tensor:
        return self.l1(input_spikes, learning=False)   # never learns: templates and thetas are untouched by scoring


def load_datasets(name: str, input_norm: bool = False):
    """(train, test) for --dataset. "mnist": the module's MNIST (loaded at import). "fmnist": Fashion-MNIST -- the same format
    (28x28 grey images in [0, 1], 10 classes, 60,000 / 10,000), read from (downloaded to) <repo>/data/FashionMNIST.
    Classes: 0 T-shirt/top, 1 trouser, 2 pullover, 3 dress, 4 coat, 5 sandal, 6 shirt, 7 sneaker, 8 bag, 9 ankle boot.
    input_norm (--input-norm): every image is scaled so its mean intensity equals the training set's (MNIST 0.131, Fashion-MNIST
    0.286), so light and heavy images send the same number of Poisson pixel spikes on average; pixels are capped at 10 (spike
    probability 0.1 x 10 = 1 per ms). Applied in the dataset's transform, so training, calibration, scoring and the shifted
    test set (built from the test set's items) all see the normalised images."""
    if name not in ("mnist", "fmnist"):
        raise ValueError(f"unknown dataset {name!r}")
    if name == "mnist" and not input_norm:
        return train_dataset, test_dataset
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    tf = transforms.ToTensor()
    if input_norm:
        target = float(cls(root=str(DATA_ROOT), train=True, download=True).data.float().mean() / 255.0)   # the training set's mean intensity
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x * (target / x.mean().clamp(min=1e-6))).clamp(max=10.0))])
    return (cls(root=str(DATA_ROOT), train=True, download=True, transform=tf),
            cls(root=str(DATA_ROOT), train=False, download=True, transform=tf))


def layer1_accuracy(l1: FeatureLayer, readout: tuple, device: torch.device, num_steps: int) -> tuple[float, float, int]:
    """Held-out accuracy with layer 1 as the output layer: the stage-B estimator (labels from train images 0-1999, accuracy
    on test images 0-999 and on their shifted copies) applied to the 1,024 layer-1 neurons. Each neuron is labelled with the
    digit it fires most for; an image is classified by summing the spikes of the neurons carrying each label.
    Returns (accuracy %, shifted accuracy %, neurons that received a label)."""
    assign_loader, eval_loader, shifted_loader = readout
    r = Layer1Readout(l1)
    labels = assign_neuron_labels(r, assign_loader, device, num_steps)
    acc = float(evaluate_network(r, eval_loader, labels, device, num_steps))
    acc_s = float(evaluate_network(r, shifted_loader, labels, device, num_steps)) if shifted_loader is not None else float("nan")
    return acc, acc_s, int((labels >= 0).sum())


def train_layer1(model: TwoLayerNetwork, loader: DataLoader, device: torch.device, cfg: dict, readout: tuple | None = None) -> dict:
    # Stage A: train layer 1 alone on its own shuffled stream (loader = l1_loader), learning on in layer 1 only, for at most
    # cfg["l1_examples"] images, then freeze it. Layer 2 is not run here at all. Every cfg["l1_snapshot"] batches it measures
    # the layer and may stop early (stop rule below). Returns the history dict that is saved into run.npz.
    l1 = model.l1; assert l1 is not None
    num_steps = cfg["num_steps"]; snap_every = cfg["l1_snapshot"]; max_examples = cfg["l1_examples"]
    # k for the old top-k criterion on 49-input templates: round(49 / 7.84) = 6 (logged only for comparison)
    k1 = max(3, int(round(l1.K / 7.84)))
    hist = {"points": [], "sharp": [], "struct": [], "theta_med": [], "theta_p90": [], "usage": [], "dW": [], "active_frac": [], "latency": [], "reservoir": [],
            "distinct": [], "tie": [], "spikes_per_window": []}   # per active window: distinct features that fired, features in its first spiking ms, spikes
    # Stage-A accuracy with layer 1 as the output layer (readout = (assign_loader, eval_loader, shifted_loader); None = not
    # scored): taken after the first batch, then every cfg["snapshot_interval"] batches (the stage-B cadence, 3,200 images),
    # and once more at the freeze, so the accuracy curve runs from image 0 to the end of stage B.
    hist.update(acc_points=[], acc=[], acc_shifted=[])
    # Snapshot-to-snapshot bookkeeping:
    #   W_prev                         templates at the previous snapshot, for the size of each template's change |dW|
    #   usage                          spikes per feature index (summed over windows and images) since the last snapshot
    #   n_windows_ink, n_windows_fired (image, window) pairs that received any input / that also produced a layer-1 spike
    #   lat_sum                        sum over active windows of the ms of their first spike (-> mean latency)
    #   distinct_sum                   sum over active windows of how many different features fired during the presentation
    #   tie_sum                        sum over active windows of how many features fired in the window's FIRST spiking ms
    #                                  (1 = no tie)
    #   spk_sum                        total layer-1 spikes in active windows
    W_prev = l1.templates().clone()
    usage = torch.zeros(l1.F, device=device)             # spikes per feature since the last snapshot
    n_windows_ink = 0; n_windows_fired = 0; lat_sum = 0.0; distinct_sum = 0.0; tie_sum = 0.0; spk_sum = 0.0
    seen = 0; batch_i = 0; t0 = time.time(); converged_at = None
    model.learn_l1 = True; model.learn_l2 = False

    def score_layer1() -> None:
        # Scoring draws its own Poisson spikes from torch's global generator, so a run made with this code does not reproduce
        # the training spikes of an older run with the same seed (the image order, set by the loader's own generator, is the same).
        acc, acc_s, n_lab = layer1_accuracy(l1, readout, device, num_steps)   # type: ignore[arg-type]
        hist["acc_points"].append(seen); hist["acc"].append(acc); hist["acc_shifted"].append(acc_s)
        print(f"[layer 1] {seen:>6d} ex | layer 1 as the output layer: held-out accuracy {acc:5.1f}% (shifted {acc_s:5.1f}%), "
              f"{n_lab}/{l1.N} neurons labelled")

    # at most one pass over the 60,000 images; the loop also stops at l1_examples or at the stop rule
    for data, targets in loader:
        B = data.shape[0]; data = data.to(device)
        # Poisson input for 350 ms: each pixel fires in each ms with probability 0.1 x intensity -> (T, B, 784) of 0/1
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        # reset only layer 1's per-presentation state (its thetas carry over)
        l1.reset(B, device)
        # Per-presentation diagnostics, all (B, P): fired = the window has spiked at least once so far; first_t = ms of its first
        # spike (NaN if none yet).
        fired = torch.zeros(B, l1.P, dtype=torch.bool, device=device)
        first_t = torch.full((B, l1.P), float("nan"), device=device)
        ink = (spike_data.sum(0)[:, l1.idx].sum(2) > 0)                                       # (B, P): windows with any input spike
        cnt = torch.zeros(B, l1.P, l1.F, device=device)                                       # spikes per feature in this presentation
        first_group = torch.zeros(B, l1.P, device=device)                                     # features that fired in the window's first spiking ms
        # The presentation, 1 ms at a time. s: (B, P, F). newly marks windows whose first spike is in this ms; for those, record
        # the ms and how many of their features fired in it.
        for t in range(num_steps):
            s = l1(spike_data[t], learning=True).view(B, l1.P, l1.F)
            any_f = s.sum(2) > 0
            newly = any_f & ~fired
            first_t = torch.where(newly, torch.full_like(first_t, float(t)), first_t)
            first_group = torch.where(newly, s.sum(2), first_group)
            fired |= any_f
            usage += s.sum(dim=(0, 1)); cnt += s
        # after the presentation: accumulate the diagnostics over windows that had input and fired (act)
        act = fired & ink
        n_windows_ink += int(ink.sum()); n_windows_fired += int(act.sum())
        lat_sum += float(torch.nan_to_num(first_t[act], nan=0.0).sum())
        distinct_sum += float((cnt[act] > 0).sum(1).float().sum()); tie_sum += float(first_group[act].sum()); spk_sum += float(cnt[act].sum())
        seen += B; batch_i += 1
        # snapshot every snap_every batches (20 batches = 640 images by default)
        if batch_i % snap_every == 0:
            W = l1.templates(); sharp = template_sharpness(W, k1).numpy(); struct = template_structure(W, l1.patch).numpy()
            dW = torch.linalg.norm((W - W_prev).float().cpu(), dim=1).numpy(); W_prev = W.clone()
            th = l1.theta.detach().float().cpu().numpy()
            # the reservoir = templates whose structure score is still below 0.4, i.e. still close to their random start
            res = int((struct < 0.4).sum())                      # unstructured templates (windows that never see ink stay here)
            hist["points"].append(seen); hist["sharp"].append(sharp.astype(np.float32)); hist["struct"].append(struct.astype(np.float32)); hist["dW"].append(dW.astype(np.float32))
            hist["theta_med"].append(float(np.median(th))); hist["theta_p90"].append(float(np.percentile(th, 90)))
            hist["usage"].append((usage / max(1, snap_every * B)).cpu().numpy().astype(np.float32))   # spikes per example, per feature
            hist["active_frac"].append(n_windows_fired / max(1, n_windows_ink)); hist["latency"].append(lat_sum / max(1, n_windows_fired))
            hist["reservoir"].append(res)
            nf = max(1, n_windows_fired)
            hist["distinct"].append(distinct_sum / nf); hist["tie"].append(tie_sum / nf); hist["spikes_per_window"].append(spk_sum / nf)
            print(f"[layer 1] {seen:>6d} ex | unstructured templates {res:3d}/{len(sharp)} | structure med {np.median(struct):.2f} | θ med {np.median(th):6.1f} p90 {np.percentile(th, 90):6.1f} | "
                  f"windows fired {100 * hist['active_frac'][-1]:4.0f}% latency {hist['latency'][-1]:5.1f} ms | per active window: {hist['spikes_per_window'][-1]:4.1f} spikes from "
                  f"{hist['distinct'][-1]:4.1f} features, {hist['tie'][-1]:4.1f} in its first spiking ms | ‖ΔW‖ max {dW.max():.2f} | {(time.time() - t0) / 60:4.1f} min")
            usage.zero_(); n_windows_ink = n_windows_fired = 0; lat_sum = 0.0; distinct_sum = tie_sum = spk_sum = 0.0
            # "converged" = no unstructured template left. This never happens in practice (border windows that never see ink keep
            # random templates); the stop rule below is what ends stage A.
            if res == 0 and converged_at is None: converged_at = seen
            # Stop rule: at least 6 snapshots, the reservoir is not smaller than it was 5 snapshots ago, AND the median theta moved by
            # less than 0.3 mV over those 5 snapshots -> the layer has stopped changing. --l1-no-early-stop switches it off.
            if (cfg["l1_stop_stationary"] and len(hist["reservoir"]) >= 6 and hist["reservoir"][-1] >= hist["reservoir"][-6]
                    and abs(hist["theta_med"][-1] - hist["theta_med"][-6]) < 0.3):
                print(f"[layer 1] stationary: the reservoir has not shrunk for 5 snapshots ({res} unstructured templates left, mostly windows that never see ink) "
                      f"and the median theta moved < 0.3 mV -> freezing at {seen} examples"); break
        if readout is not None and (batch_i == 1 or batch_i % cfg["snapshot_interval"] == 0): score_layer1()
        if seen >= max_examples: break
    # final score at the freeze point (unless that batch was just scored), so the stage-A curve ends where stage B begins
    if readout is not None and (not hist["acc_points"] or hist["acc_points"][-1] != seen): score_layer1()
    # Freeze: from here on TwoLayerNetwork calls layer 1 with learning=False -- no STDP, no theta increments, no theta decay.
    # The within-window inhibition and the tie-break stay on: they are part of how the layer computes its output.
    model.learn_l1 = False   # FREEZE
    hist["converged_at"] = converged_at; hist["frozen_at"] = seen; hist["minutes"] = (time.time() - t0) / 60
    return hist


def calibrate_layer2(model: TwoLayerNetwork, loader: DataLoader, device: torch.device, num_steps: int, target_drive: float, n_batches: int = 5) -> dict:
    """Measure the rate at which the frozen layer 1 (after pooling) feeds layer 2 -- total input spikes per ms per
    presentation -- and set layer 2's weight scale so that a flat layer-2 neuron receives target_drive mV/ms on average
    (the pixel network gets ~1.2: 12 input spikes/ms x mean weight 0.1). Mean weight m = target_drive / rate;
    row sum = m x inputs; w_max = 10 m (the base file's 1.0 / 0.1 ratio). Rescales the existing uniform weights."""
    # Example (16 features, default drive): layer 1 sends 0.954 spikes/ms per presentation summed over its 1,024 lines, so
    # m = 1.2 / 0.954 = 1.258, row sum = 1.258 x 1,024 = 1,288 and w_max = 12.58; a layer-1 spike onto a fully learned
    # synapse is then worth ~10 mV. --l2-drive changes target_drive and so scales all three by the same factor.
    assert model.l1 is not None
    model.learn_l1 = False; model.learn_l2 = False
    # run n_batches batches (160 images) through the frozen layer 1 (and the pooling, if on), counting every spike it sends
    tot_spikes = 0.0; tot_pres = 0
    with torch.no_grad():
        for bi, (data, _) in enumerate(loader):
            if bi >= n_batches: break
            B = data.shape[0]; data = data.to(device)
            spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
            model.reset(B, device)
            for t in range(num_steps):
                x = model.l1(spike_data[t], learning=False)
                if model.pool2: x = pool2x2(x, model.l1.side, model.l1.F)
                tot_spikes += float(x.sum())
            tot_pres += B
    rate = tot_spikes / max(1, tot_pres) / num_steps                      # input spikes per ms per presentation, summed over lines
    # mean weight that gives target_drive mV/ms on average to a neuron whose weights are all equal (a fresh, near-uniform row)
    m = target_drive / max(rate, 1e-6)
    # New row sum and ceiling. The existing weights already sum to the provisional row sum (old_norm), so multiplying them by
    # w_norm / old_norm keeps their random pattern and gives every row the new sum.
    l2 = model.l2; old_norm = l2.w_norm
    l2.w_norm = m * l2.n_inputs; l2.w_max = 10.0 * m
    with torch.no_grad():
        l2.synapses.weight.mul_(l2.w_norm / old_norm)
    info = {"l2_input_rate_per_ms": rate, "l2_mean_weight": m, "l2_norm": l2.w_norm, "l2_wmax": l2.w_max, "target_drive": target_drive}
    print(f"[calibration] layer 1 -> layer 2: {rate:.2f} input spikes/ms per presentation over {l2.n_inputs} lines; mean weight set to {m:.3f} (row sum {l2.w_norm:.1f}, w_max {l2.w_max:.2f}) for {target_drive} mV/ms initial drive")
    return info


# =====================================================================================================================
# Stage B: train layer 2 (the test4 measurement set, single i.i.d. stream) + the shifted evaluation
# =====================================================================================================================
def train_layer2(model: TwoLayerNetwork, train_loader, assign_loader, eval_loader, shifted_loader, device, cfg: dict, run_dir: Path, l1_hist: dict | None):
    # Stage B: train layer 2 for num_epochs passes over the 60,000 training images, layer 1 frozen, and take the test4
    # snapshot set every snapshot_interval batches. Writes run.npz, run.json and the figures into run_dir.
    n = model.num_neurons; num_epochs, num_steps, snapshot_interval = cfg["num_epochs"], cfg["num_steps"], cfg["snapshot_interval"]
    # k for the top-k share on 1,024 inputs: round(1,024 / 7.84) = 131, the analogue of the single layer's top 100 of 784
    k2 = max(3, int(round(model.l2.n_inputs / 7.84)))
    P, A, S, R, H, C, L = [], [], [], [], [], [], []             # points, acc, shifted acc, reservoir, hijack, conf, assignments
    WS, TH, SH, RC, L1U = [], [], [], [], []                       # window spikes, theta, sharpness, recruit steps, layer-1 usage
    DS = []                                                         # NEW: distinct layer-2 neurons per image (last batch) at each snapshot
    # recruit_step[j] = examples seen when neuron j's template first reached a top-k share of 0.35 (NaN until then)
    recruit_step = np.full(n, np.nan)
    W_prev = model.l2.synapses.weight.detach().clone(); W_init = W_prev.clone(); CI, RI = [], []   # cosine to the initial row per snapshot, reservoir by that criterion
    # spikes_since: layer-2 spikes per neuron since the last snapshot ("window spikes"); l1_usage: layer-1 spikes per feature
    # index since the last snapshot (None for --no-l1); gbi = global batch index across epochs
    spikes_since = torch.zeros(n, device=device); l1_usage = torch.zeros(model.l1.F, device=device) if model.l1 is not None else None
    total = 0; gbi = 0; t0 = time.time()
    model.learn_l1 = False; model.learn_l2 = True
    for epoch in range(num_epochs):
        for data, targets in train_loader:
            B = data.shape[0]; data = data.to(device)
            # (T, B, 784) Poisson pixel spikes, as in the base file
            spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
            model.reset(B, device)
            # bts: layer-2 spike counts per image over this presentation, (B, 400)
            bts = torch.zeros(B, n, device=device)
            for t in range(num_steps):
                # learning on: layer 2 learns (learn_l2 True), layer 1 runs frozen (learn_l1 False)
                out = model(spike_data[t])
                bts += out
                if l1_usage is not None: l1_usage += model.last_l1_spikes.view(B, -1, model.l1.F).sum(dim=(0, 1))   # type: ignore
            spikes_since += bts.sum(0); total += B
            if model.l2.lat_lr: model.l2.update_lateral(bts)                    # NEW (--l2-lat-lr): learned lateral inhibition
            # snapshot at batch 0 and then every snapshot_interval batches (100 batches = 3,200 images by default)
            if gbi % snapshot_interval == 0:
                # labels from train images 0-1999, accuracy on test images 0-999 and on their shifted copies, learning off: the
                # hypothesis-work estimator
                labels = assign_neuron_labels(model, assign_loader, device, num_steps)
                acc, cm = evaluate_network(model, eval_loader, labels, device, num_steps, return_matrix=True)   # type: ignore
                acc_s = evaluate_network(model, shifted_loader, labels, device, num_steps) if shifted_loader is not None else float("nan")
                with torch.no_grad():
                    # Weight diagnostics: top-k share and theta of every neuron; dw = how far each row moved since the last snapshot; hijacker
                    # index = share of all that movement carried by the 20 rows that moved most (the hypothesis-work measure).
                    W = model.l2.synapses.weight.detach()
                    sharp = template_sharpness(W, k2).numpy(); th = model.l2.neurons.theta.detach().float().cpu().numpy().copy()   # type: ignore
                    dw = torch.linalg.norm((W - W_prev).float().cpu(), dim=1); tot = float(dw.sum())
                    hij = float(torch.topk(dw, min(20, n)).values.sum()) / tot if tot > 0 else float("nan"); W_prev = W.clone()
                    ws = spikes_since.cpu().numpy().copy(); spikes_since.zero_()
                # Two "untrained" counts: the old one (top-k share < 0.35, which sparse inputs cap) and the criterion-free one (rows whose
                # cosine to their own initial weights is still > 0.95).
                newly = (sharp >= 0.35) & np.isnan(recruit_step); recruit_step[newly] = total
                res = int((sharp < 0.35).sum())
                ci = cosine_to_init(W, W_init).numpy(); res_init = int((ci > 0.95).sum()); CI.append(ci.astype(np.float32)); RI.append(res_init)
                P.append(total); A.append(float(acc)); S.append(float(acc_s)); R.append(res); H.append(hij)
                C.append(np.asarray(cm).astype(np.int32)); L.append(labels.cpu().numpy().astype(np.int16))
                WS.append(ws.astype(np.float32)); TH.append(th.astype(np.float32)); SH.append(sharp.astype(np.float32)); RC.append(recruit_step.copy())
                if l1_usage is not None: L1U.append(l1_usage.cpu().numpy().astype(np.float32)); l1_usage.zero_()
                # Console readouts. distinct = mean, over the images of the LAST batch, of how many different layer-2 neurons fired (1 = one
                # neuron took the presentation). The spikes-per-presentation numbers divide by snapshot_interval x B; at the very first
                # snapshot (batch 0) only one batch has been counted, which is why the first logged value is ~1/100 of the real one.
                distinct = float((bts > 0).sum(dim=1).float().mean())
                DS.append(distinct)
                l1_rate = (float(L1U[-1].sum()) / max(1, snapshot_interval * B)) if L1U else float("nan")
                print(f"[layer 2] {total:>7d} ex | held-out acc {acc:5.1f}% shifted {acc_s:5.1f}% | reservoir {res:3d} (unmoved rows {res_init:3d}, cos-to-init med {np.median(ci):.2f}) | hijacker idx {hij:4.2f} | "
                      f"L2 spikes/presentation {ws.sum() / max(1, snapshot_interval * B):5.1f} from {distinct:4.1f} distinct neurons (last batch) | L1 spikes/presentation {l1_rate:5.1f} | {(time.time() - t0) / 60:5.1f} min")
            gbi += 1
    print("Training complete")
    # NEW (results_for_cnn_replication): what each layer-2 neuron fires for on the 1,000 test digits (see layer2_test_stats)
    fs = layer2_test_stats(model, eval_loader, device, num_steps)
    fired = fs["sta2_count"] > 0
    pur = fs["class_counts2"].max(1) / np.maximum(fs["class_counts2"].sum(1), 1)
    active = fs["sta2_count"] >= 10
    print(f"[layer 2] test digits: {fs['distinct_test'].mean():.1f} distinct neurons and {fs['spikes_test'].mean():.1f} spikes per image; "
          f"{int(fired.sum())}/{n} neurons fired; median class purity {np.median(pur[active]) if active.any() else float('nan'):.2f}")
    # Summary for run.json and the plotting code (multi_layer_plots.py, adapted from test2/test4). spike_time = when the accuracy
    # curve shows the hypothesis-work "spike" (NaN if none); final accuracy = mean of the last 3 snapshots. Several keys here
    # and in the npz (intervention_step_actual, anneal_step_actual, n_refill, theta_plus_late, cooper_*, imprinted, refilled)
    # are placeholders those figures expect; they are unused in this script.
    spike_time = spike_time_from_curve(P, A)
    out = dict(cfg); out.update({
        "spike_time": spike_time, "intervention_step_actual": None, "anneal_step_actual": None,
        "theta_base": -52.0, "theta_plus": float(model.l2.neurons.theta_plus), "inhibition_mV": float(model.l2.inh_matrix[0, 1].item()),
        "untrained_threshold": 0.35, "top_k": k2, "n_refill": 0, "theta_plus_late": None,
        "num_neurons": n, "l2_inputs": model.l2.n_inputs, "device": str(device), "minutes": (time.time() - t0) / 60,
        "final_accuracy": float(np.mean(A[-3:])), "final_accuracy_shifted": float(np.nanmean(S[-3:])),
        "test_distinct_per_image": float(fs["distinct_test"].mean()), "test_spikes_per_image": float(fs["spikes_test"].mean()),
        "test_neurons_fired": int(fired.sum()),
        "class_purity_median": float(np.median(pur[active])) if active.any() else None,
        "lat_inh_final_mean": (float(model.l2.inh_matrix.sum() / max(1, n * (n - 1)))) if model.l2.lat_lr else None,
        "lat_inh_final_frac_at_max": (float((model.l2.inh_matrix >= 0.9 * model.l2.lat_max).sum()) / max(1, n * (n - 1))) if model.l2.lat_lr else None,
        "l1": None if l1_hist is None else {"converged_at": l1_hist["converged_at"], "frozen_at": l1_hist["frozen_at"], "minutes": l1_hist["minutes"],
                                            "P": model.l1.P, "F": model.l1.F, "K": model.l1.K, "side": model.l1.side},   # type: ignore
    })
    if out["l1"] is not None and l1_hist is not None and l1_hist.get("acc"):   # layer 1's own accuracy at the freeze point
        out["l1"]["readout_accuracy_at_freeze"] = l1_hist["acc"][-1]; out["l1"]["readout_accuracy_shifted_at_freeze"] = l1_hist["acc_shifted"][-1]
    empty = np.zeros(0, dtype=np.float32)
    npz = dict(points=np.array(P), accuracy=np.array(A), accuracy_shifted=np.array(S),
               cooper_points=empty, cooper_all=empty, cooper_proportion=empty,
               reservoir_count=np.array(R), hijack_index=np.array(H), window_spikes=np.array(WS), theta_snap=np.array(TH), sharp_snap=np.array(SH),
               assignments_snap=np.array(L), conf_snap=np.array(C), recruit_step_snap=np.array(RC),
               imprinted=np.zeros(n, bool), refilled=np.zeros(n, bool),
               cos_init_snap=np.array(CI), reservoir_init=np.array(RI), W2_init=W_init.float().cpu().numpy().astype(np.float16),
               W2=model.l2.synapses.weight.detach().float().cpu().numpy().astype(np.float16))
    # NEW (results_for_cnn_replication): distinct neurons per image at each snapshot, and the final test-digit statistics
    npz.update(distinct_snap=np.array(DS, dtype=np.float32), sta2=fs["sta2"].astype(np.float16), sta2_count=fs["sta2_count"].astype(np.float32),
               class_counts2=fs["class_counts2"].astype(np.float32), distinct_test=fs["distinct_test"].astype(np.float32),
               spikes_test=fs["spikes_test"].astype(np.float32))
    if model.l2.lat_lr: npz["inh2"] = model.l2.inh_matrix.detach().float().cpu().numpy().astype(np.float16)   # the learned inhibition (mV)
    # layer-1 arrays (templates, thetas, the stage-A history) are saved too, so a run can be analysed or replayed later
    if model.l1 is not None and l1_hist is not None:
        npz.update(W1=model.l1.templates().float().cpu().numpy().astype(np.float32), l1_theta=model.l1.theta.detach().float().cpu().numpy().astype(np.float32),
                   l1_usage_snap=np.array(L1U), l1_points=np.array(l1_hist["points"]), l1_sharp=np.array(l1_hist["sharp"]), l1_struct=np.array(l1_hist["struct"]), l1_dW=np.array(l1_hist["dW"]),
                   l1_theta_med=np.array(l1_hist["theta_med"]), l1_theta_p90=np.array(l1_hist["theta_p90"]), l1_usage=np.array(l1_hist["usage"]),
                   l1_active_frac=np.array(l1_hist["active_frac"]), l1_latency=np.array(l1_hist["latency"]), l1_reservoir=np.array(l1_hist["reservoir"]),
                   l1_distinct=np.array(l1_hist["distinct"]), l1_tie=np.array(l1_hist["tie"]), l1_spikes_per_window=np.array(l1_hist["spikes_per_window"]))
        # stage-A accuracy with layer 1 as the output layer (x = layer-1 training images; empty if stage A was not scored)
        npz.update(l1_acc_points=np.array(l1_hist.get("acc_points", [])), l1_accuracy=np.array(l1_hist.get("acc", [])),
                   l1_accuracy_shifted=np.array(l1_hist.get("acc_shifted", [])))
    # run.npz holds every array, run.json the config plus the summary numbers; then the F-figures are drawn
    np.savez(run_dir / "run.npz", **npz)
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    make_run_figures(run_dir)
    try:   # README.md in the run folder: every setting, layer by layer, and the command line (describe_runs.py); never fails the run
        import importlib.util
        _spec = importlib.util.spec_from_file_location("two_layer_describe_runs", Path(__file__).resolve().parent / "describe_runs.py")
        _describe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_describe)      # type: ignore[union-attr]
        _describe.write_run_readme(run_dir, sys.argv[1:], torch.cuda.get_device_name(device) if torch.device(device).type == "cuda"
                                   else torch.device(device).type)
    except Exception as e:                                                                                # noqa: BLE001
        print(f"(run README not written: {e})")
    print(f"\nrun finished in {out['minutes']:.1f} min | final acc {out['final_accuracy']:.1f}% shifted {out['final_accuracy_shifted']:.1f}% | spike time {spike_time if spike_time == spike_time else 'none'} | {run_dir}")


# =====================================================================================================================
def main() -> None:
    # Command-line flags. The defaults are the batch-1 design; the batch-specific values come from unity/submit_multi_layer.sh.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- layer 1 ----
    p.add_argument("--no-l1", action="store_true", help="single-layer reference: pixels straight to layer 2 (784 inputs, norm 78.4, wmax 1)")
    p.add_argument("--patch", type=int, default=7); p.add_argument("--stride", type=int, default=3); p.add_argument("--features", type=int, default=16)
    p.add_argument("--shared", action="store_true", help="tie the templates across positions (the shared design; default here is unshared / locally connected)")
    # --l1-theta-tau is in training EXAMPLES; it is converted to timesteps in the build section below
    p.add_argument("--l1-theta-plus", type=float, default=0.1); p.add_argument("--l1-theta-tau", type=float, default=2000.0, help="layer-1 theta decay in examples (0 = off)")
    p.add_argument("--l1-theta-shared", action="store_true", help="one theta per feature (global conscience) instead of per neuron")
    p.add_argument("--l1-norm", type=float, default=12.0, help="layer-1 template row sum (mean weight ~0.25 over 49 inputs)"); p.add_argument("--l1-wmax", type=float, default=1.5)
    p.add_argument("--l1-inhibition", type=float, default=200.0); p.add_argument("--l1-stdp-reduce", choices=["sum", "mean"], default="sum")
    p.add_argument("--l1-examples", type=int, default=30000, help="stage A length (max); each template sees one window per image"); p.add_argument("--l1-snapshot", type=int, default=20, help="batches between layer-1 snapshots")
    p.add_argument("--l1-stop-stationary", action="store_true", default=True); p.add_argument("--l1-no-early-stop", dest="l1_stop_stationary", action="store_false")
    p.add_argument("--pool2", action="store_true", help="2x2 spike-OR pooling between the layers")
    # ---- layer 2 ----
    # --theta-tau (layer 2) is also in examples; None = no decay (the base file's 1e7-timestep constant)
    p.add_argument("--neurons", type=int, default=400); p.add_argument("--theta-plus", type=float, default=0.5); p.add_argument("--theta-tau", type=float, default=None)
    p.add_argument("--inhibition", type=float, default=200.0)
    p.add_argument("--l2-explain", action="store_true", help="explaining away in layer 2 (needs --l2-k): each winner claims the input "
                   "lines its template covers; other neurons only receive unclaimed input, and each winner learns from what the other "
                   "winners leave. Default: off")
    p.add_argument("--l2-lat-lr", type=float, default=0.0, help="learned lateral inhibition in layer 2: mV added per batch per unit of "
                   "co-activation (fraction of images on which two neurons both fired), minus lat-target^2. 0 = fixed inhibition (default)")
    p.add_argument("--l2-lat-target", type=float, default=0.0125, help="target fraction of layer-2 neurons firing per image (5/400)")
    p.add_argument("--l2-lat-max", type=float, default=200.0, help="ceiling of the learned inhibition (mV)")
    p.add_argument("--l2-k", type=int, default=None, help="k winners per image in layer 2: the first k neurons to reach threshold share "
                   "the image and fire freely (no inhibition among them, none at all until there are k); every other neuron is "
                   "silenced. Replaces --l2-wta's one-spike-per-millisecond rule. Default: off (baseline behaviour)")
    p.add_argument("--l2-wta", action="store_true", help="hard winner-take-all per timestep in layer 2 (resolves simultaneous crossings caused by layer 1's synchronous bursts); off = baseline behaviour")
    p.add_argument("--l1-readout-inhibition", type=float, default=None, help="after stage A, run layer 1 with this within-window inhibition (mV) for layer-2 training and evaluation instead of --l1-inhibition; e.g. 17.5 lets a runner-up feature fire too, so codes of two instances of a class overlap more")
    p.add_argument("--l1-wta", action="store_true", help="within-window tie-break in layer 1: when several features of a window cross threshold in the same step, only the highest fires; off = baseline behaviour")
    p.add_argument("--l2-drive", type=float, default=1.2, help="initial drive (mV/ms) a flat layer-2 neuron should receive; layer 2's weight scale is calibrated from layer 1's measured output rate to give this")
    p.add_argument("--l2-norm", type=float, default=None, help="override the calibrated row sum"); p.add_argument("--l2-wmax", type=float, default=None, help="override the calibrated w_max")
    p.add_argument("--l2-lr-mult", type=float, default=1.0, help="multiply layer 2's STDP learning rates (lr+ 1e-2, lr- 1e-4) by this factor; 1 = baseline behaviour")
    # ---- run control ----
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--shift", type=int, default=2, help="max |dx|,|dy| of the shifted test set (0 = none)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=350); p.add_argument("--snapshot-interval", type=int, default=100)
    p.add_argument("--tag", default=""); p.add_argument("--smoke", action="store_true"); p.add_argument("--replot", default=None, metavar="RUN_DIR")
    p.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist", help="mnist, or fmnist = Fashion-MNIST (same format; "
                   "downloaded to <repo>/data on first use)")
    p.add_argument("--input-norm", action="store_true", help="scale every image to the training set's mean intensity (equal total "
                   "pixel input for light and heavy images)")
    p.add_argument("--results-dir", default="results", help="folder that receives the run folder: relative to this script's folder "
                   "(e.g. results_2, results/smoke_tests) or absolute")
    args = p.parse_args()
    # --replot only redraws the figures of an existing run folder
    if args.replot:
        print("figures regenerated:", make_run_figures(args.replot)); return

    # Seeds for Python, numpy and torch (weight initialisation, Poisson spikes, shuffling). MPS and CUDA still draw different
    # random numbers from the same seed.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.backends.mps.is_available(): device = torch.device("mps"); print("Using MPS.")
    elif torch.cuda.is_available(): device = torch.device("cuda"); print("Using CUDA.")
    else: device = torch.device("cpu"); print("Using CPU.")

    # --smoke: tiny settings to check the code path in a few minutes (640 training images, 25-ms presentations)
    train_ds, test_ds = load_datasets(args.dataset, args.input_norm)
    if args.smoke:
        train_subset = Subset(train_ds, range(0, 32 * 20)); assign_subset = Subset(train_ds, range(0, 128)); eval_subset = Subset(test_ds, range(0, 64))
        assign_bs = eval_bs = 128; args.epochs, args.steps, args.snapshot_interval = 1, 25, 5
        args.l1_examples, args.l1_snapshot = 32 * 14, 2; n_shift = 64
    else:
        train_subset = train_ds; assign_subset = Subset(train_ds, range(0, 2000)); eval_subset = Subset(test_ds, range(0, 1000))
        assign_bs = eval_bs = 512; n_shift = 1000
    # Loaders. train_loader feeds stage B; l1_loader feeds stage A and the calibration, shuffled with a different seed (seed +
    # 1000) so the two stages see the images in different orders. assign/eval: train 0-1999 and test 0-999 in a fixed order
    # (the hypothesis-work estimator); shifted_loader: the same test images, shifted.
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=gen)
    l1_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1000))
    assign_loader = DataLoader(assign_subset, batch_size=assign_bs, shuffle=False); eval_loader = DataLoader(eval_subset, batch_size=eval_bs, shuffle=False)
    shifted_loader = DataLoader(shifted_test_set(n_shift, args.shift, dataset=test_ds), batch_size=eval_bs, shuffle=False) if args.shift > 0 else None

    # ---- build ----
    # --no-l1: the single-layer reference, i.e. the base network (784 inputs, row sum 78.4, w_max 1.0) with this script's
    # evaluation
    if args.no_l1:
        l1 = None; n_in = 784; l2_norm = args.l2_norm or 78.4; l2_wmax = args.l2_wmax or 1.0
    else:
        # theta decays once per timestep and a batch of 32 images is 350 timesteps, so tau examples = tau x 350 / 32 timesteps;
        # tau <= 0 means 1e7 timesteps, i.e. no decay in practice. The layer-1 learning rates are the base file's 1e-2 / 1e-4.
        tau1 = 1e7 if args.l1_theta_tau <= 0 else args.l1_theta_tau * args.steps / 32
        l1 = FeatureLayer(args.patch, args.stride, args.features, bool(args.shared), args.l1_theta_plus, tau1, args.l1_theta_shared,
                          args.l1_norm, args.l1_wmax, 1e-2, 1e-4, args.l1_inhibition, args.l1_stdp_reduce, wta=bool(args.l1_wta))
        # Layer 2's input size: 256 pooled lines with --pool2, else 1,024. Its weight scale on the next line is provisional (mean
        # weight 0.1, w_max 1.0, the base file's values); calibrate_layer2() replaces it after stage A unless --l2-norm/--l2-wmax.
        n_in = (l1.side // 2) ** 2 * l1.F if args.pool2 else l1.N
        l2_norm = args.l2_norm or 0.1 * n_in; l2_wmax = args.l2_wmax or 1.0     # provisional; replaced by the calibration after stage A unless overridden
    # layer 2's theta decay, converted the same way; None -> 1e7 timesteps (the base file's value)
    tau2 = 1e7 if args.theta_tau is None else args.theta_tau * args.steps / 32
    l2 = DiehlAndCookNetwork(n_in, args.neurons, args.theta_plus, args.inhibition, tau2, None, l2_norm, l2_wmax, wta=bool(args.l2_wta))
    l2.lr_plus *= args.l2_lr_mult; l2.lr_minus *= args.l2_lr_mult      # both rates together, so the potentiation/depression ratio is unchanged
    if args.l2_k: l2.neurons.k_winners = int(args.l2_k)                # NEW: k winners per image (see excitatory_neurons._k_winners)
    if args.l2_explain:                                                # NEW: explaining away (see DiehlAndCookNetwork._explained_drive)
        if not args.l2_k: raise SystemExit("--l2-explain needs --l2-k (explaining away acts between an image's winners)")
        l2.explain = True
    l2.lat_lr, l2.lat_target, l2.lat_max = float(args.l2_lat_lr), float(args.l2_lat_target), float(args.l2_lat_max)   # NEW: learned inhibition
    model = TwoLayerNetwork(l1, l2, args.pool2).to(device)

    # Run folder name: architecture, layer-2 theta settings, epochs, seed, then one tag per non-default option and a time stamp,
    # e.g. unshared_p7s3f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_drv0.36_lr10_20261113-165859.
    variant = "singlelayer" if args.no_l1 else (f"{'shared' if args.shared else 'unshared'}_p{args.patch}s{args.stride}f{args.features}" + ("_pool2" if args.pool2 else ""))
    run_name = f"{variant}_n{args.neurons}_tp{args.theta_plus:g}" + (f"_tau{args.theta_tau / 1000:g}k" if args.theta_tau else "_nodecay") + f"_ep{args.epochs}_seed{args.seed}"
    if args.l1_wta: run_name += "_l1wta"
    if args.l1_readout_inhibition is not None and not args.no_l1: run_name += f"_rd{args.l1_readout_inhibition:g}"
    if args.l1_norm != 12.0 and not args.no_l1: run_name += f"_l1n{args.l1_norm:g}"                # layer-1 row sum (drive)
    if args.l1_theta_plus != 0.1 and not args.no_l1: run_name += f"_l1tp{args.l1_theta_plus:g}"    # layer-1 conscience
    if args.l2_wta: run_name += "_l2wta"
    if args.inhibition != 200.0: run_name += f"_inh{args.inhibition:g}"       # layer-2 lateral inhibition (mV)
    if args.l2_k: run_name += f"_k{args.l2_k}"                                 # k winners per image
    if args.l2_explain: run_name += "_ea"                                       # explaining away
    if args.l2_lat_lr: run_name += f"_lat{args.l2_lat_lr:g}" + (f"p{args.l2_lat_target:g}" if args.l2_lat_target != 0.0125 else "")   # learned inhibition
    if args.l2_drive != 1.2 and not args.no_l1: run_name += f"_drv{args.l2_drive:g}"
    if args.l2_lr_mult != 1.0: run_name += f"_lr{args.l2_lr_mult:g}"
    if args.dataset != "mnist": run_name += f"_{args.dataset}"
    if args.input_norm: run_name += "_inorm"
    if args.tag: run_name += f"_{args.tag}"
    if args.smoke: run_name += "_SMOKE"
    run_name += "_" + time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.results_dir) if Path(args.results_dir).is_absolute() else script_dir / args.results_dir   # default: results_root
    run_dir = out_root / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    # cfg is saved as run.json. Some keys (arm, sweep, multi, init, theta_init, anneal_at, ...) are fixed values that the
    # plotting and summary code shared with the hypothesis-work scripts expects.
    cfg = {"arm": "baseline", "sweep": True, "multi": True, "seed": args.seed, "num_epochs": args.epochs, "num_steps": args.steps, "snapshot_interval": args.snapshot_interval,
           "batch_size": 32, "smoke": bool(args.smoke), "tag": args.tag, "run_name": run_name, "init": "uniform", "theta_init": None, "theta_tau": args.theta_tau,
           "theta_plus_arg": args.theta_plus, "inhibition_arg": args.inhibition, "theta_max": None, "anneal_at": None, "theta_plus_late": None,
           "l2_lr_mult": args.l2_lr_mult, "l2_drive_arg": args.l2_drive, "l2_wta": bool(args.l2_wta), "l1_wta": bool(args.l1_wta), "l1_readout_inhibition": args.l1_readout_inhibition, "no_l1": bool(args.no_l1), "patch": args.patch, "stride": args.stride, "features": args.features, "shared": bool(args.shared), "pool2": bool(args.pool2),
           "l1_theta_plus": args.l1_theta_plus, "l1_theta_tau": args.l1_theta_tau, "l1_theta_shared": bool(args.l1_theta_shared), "l1_norm": args.l1_norm, "l1_wmax": args.l1_wmax,
           "l1_inhibition": args.l1_inhibition, "l1_stdp_reduce": args.l1_stdp_reduce, "l1_examples": args.l1_examples, "l1_snapshot": args.l1_snapshot,
           "l1_stop_stationary": bool(args.l1_stop_stationary), "l2_inputs": n_in, "l2_norm": l2_norm, "l2_wmax": l2_wmax, "shift": args.shift}
    cfg["dataset"] = args.dataset; cfg["input_norm"] = bool(args.input_norm)
    cfg["l2_k"] = args.l2_k
    cfg["l2_explain"] = bool(args.l2_explain); cfg["l2_lat_lr"] = args.l2_lat_lr; cfg["l2_lat_target"] = args.l2_lat_target; cfg["l2_lat_max"] = args.l2_lat_max
    print("config:", json.dumps(cfg))
    # stage A (skipped for --no-l1)
    l1_hist = train_layer1(model, l1_loader, device, cfg, (assign_loader, eval_loader, shifted_loader)) if l1 is not None else None
    if l1 is not None:
        # Optional readout inhibition: layer 1 learned with --l1-inhibition; from here on it runs with this value instead.
        if args.l1_readout_inhibition is not None:
            model.l1.inhibition = float(args.l1_readout_inhibition)      # learning is frozen; only the readout competition changes (calibration below sees it)
            print(f"[layer 1] readout: within-window inhibition set to {model.l1.inhibition:g} mV for stage B and evaluation (was {args.l1_inhibition:g} during learning)")
        # set layer 2's weight scale from layer 1's measured output rate (see calibrate_layer2)
        cal = calibrate_layer2(model, l1_loader, device, args.steps, args.l2_drive)
        if args.l2_norm is not None or args.l2_wmax is not None:      # explicit overrides win
            l2 = model.l2
            if args.l2_norm is not None:
                with torch.no_grad(): l2.synapses.weight.mul_(args.l2_norm / l2.w_norm)
                l2.w_norm = args.l2_norm
            if args.l2_wmax is not None: l2.w_max = args.l2_wmax
            print(f"[calibration] overridden: row sum {l2.w_norm:.1f}, w_max {l2.w_max:.2f}")
        # record the calibration and the final weight scale, and write run.json now, so a crash in stage B still leaves the config
        cfg.update(cal); cfg["l2_norm"] = model.l2.w_norm; cfg["l2_wmax"] = model.l2.w_max
        (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))
    # stage B (for --no-l1, the whole run)
    train_layer2(model, train_loader, assign_loader, eval_loader, shifted_loader, device, cfg, run_dir, l1_hist)


if __name__ == "__main__":
    main()
