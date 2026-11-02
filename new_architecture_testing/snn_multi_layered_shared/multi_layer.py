"""
multi_layer.py -- new_architecture_testing / snn_multi_layered_shared: a two-layer spiking network learned with STDP only.

WHY
  A Diehl & Cook neuron that sees all 784 pixels learns a whole-digit template, so a single layer is a prototype
  classifier whose ceiling is set by the number of prototypes (87-89% at 400). Giving layer-1 neurons LOCAL receptive
  fields forces them to learn strokes instead of digits; layer 2 then learns prototypes over "which stroke is where",
  a representation in which classes should separate better and which, with shared weights, tolerates translation.

ARCHITECTURE (defaults; every number is a flag)
  Layer 1 -- the feature layer
    * windows:  --patch 7 pixels, --stride 3  ->  side = (28-7)/3+1 = 8, P = 64 window positions, K = 49 inputs each.
    * neurons:  --features F = 16 per position, N1 = P*F = 1,024 LIF cells, each with its own membrane potential, spike
                train and (by default) its own theta. Same cell equations as the single layer (excitatory_neurons).
    * templates: SHARED by default: one (F, K) weight matrix; neuron (position p, feature f) uses template f whatever p
                is, so the 64 neurons of feature f are the same stroke detector pointed at 64 windows. --unshared gives
                every neuron its own template ((P*F, K) matrix; a locally-connected layer).
    * competition: the F neurons at a position inhibit each other (--l1-inhibition 200 mV, first to fire wins the window
                for the presentation and keeps firing at its match strength). Positions never inhibit each other.
    * learning: the D&C STDP rule per neuron (pre-trace potentiation on post spikes, weak depression, clamp to
                [0, --l1-wmax], template normalised to --l1-norm). With shared templates the updates of the 64 neurons
                that use template f are REDUCED into the one template: --l1-stdp-reduce sum (default: mean over the
                batch, sum over positions, so one win moves a template as much as one win moves a layer-2 neuron) or
                mean (also averaged over positions: 64x smaller steps).
    * theta:    per neuron by default (local conscience; raised by the neuron's own spikes only). --l1-theta-shared gives
                one theta per feature, raised by the feature's spikes averaged over positions (global conscience).
                --l1-theta-plus 0.1 and --l1-theta-tau 2000 examples (decay) by default: layer-1 neurons win ~10x more
                often than layer-2 neurons, so a short decay keeps their thresholds a few mV above base.
  Pooling (--pool2, off by default): spike-OR over 2x2 blocks of positions per feature, 8x8 -> 4x4, layer-2 input 256.
  Layer 2 -- the prototype layer
    * the 400-neuron D&C layer exactly as in test2/test4, reading the N1 (or pooled) layer-1 spike trains instead of
      pixels; theta_plus 0.5 (the fast regime), optional decay; --l2-norm (default 0.1 x inputs), --l2-wmax (default 4,
      because a learned layer-2 template concentrates on ~30 active input lines and would hit a clamp of 1).
  --no-l1: pixels go straight to layer 2 (n_inputs 784, norm 78.4, wmax 1) = the single-layer reference, evaluated
      with the same code and the same shifted test set.

TRAINING (sequential)
  Stage A: layer 1 alone, batches of 32 shuffled training images, up to --l1-examples (default 9,600); every
           --l1-snapshot batches record each template's sharpness (top-k share, k = K/7.84, threshold 0.35 as before),
           the reservoir of flat templates, theta, usage per feature, the size of the weight change, the fraction of
           windows that fired and their first-spike latency. Stop early once the reservoir has been 0 for 3 snapshots and
           the median theta moved < 0.3 mV over them (--l1-stop-stationary), then FREEZE layer 1 (no STDP, theta fixed).
  Stage B: layer 2 with the full snapshot machinery of test4 (held-out accuracy, confusion, assignments, reservoir,
           hijacker index, window spikes, theta, sharpness, recruitment) for --epochs epochs, plus at every snapshot the
           accuracy on a SHIFTED copy of the 1,000 test images (each shifted by a fixed random offset in
           [-shift, shift]^2, --shift 2): pixel templates should lose accuracy under shifts, stroke features less so.

OUTPUTS results/<run>/run.npz (everything above incl. the layer-1 templates W1 and layer-2 weights W2), run.json,
        F1..F7 (layer 2, as before, F1 with the shifted curve), F13 layer-2 templates projected back to image space,
        F14 the learned layer-1 templates, F15 the stage-A curves.
USAGE  python multi_layer.py --seed 0                   python multi_layer.py --unshared        python multi_layer.py --pool2
       python multi_layer.py --no-l1                    python multi_layer.py --smoke           python multi_layer.py --replot results/<run>
       python summarize_multi_layer.py                  bash run_multi_layer.sh
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

script_dir: Path = Path(__file__).resolve().parent
repo_root: Path = script_dir.parents[1]            # .../cooper
DATA_ROOT: Path = repo_root / "data"
results_root: Path = script_dir / "results"

train_dataset: datasets.MNIST = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transforms.ToTensor())
test_dataset: datasets.MNIST = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=transforms.ToTensor())


# =====================================================================================================================
# The LIF cell: copied verbatim from test4.py (excitatory_neurons). Used by layer 2; layer 1 re-implements the SAME
# equations inline because its theta may be per feature rather than per neuron (see FeatureLayer.forward).
# =====================================================================================================================
class excitatory_neurons(nn.Module):
    def __init__(self, theta_plus: float = 0.05, theta_decay_tau_steps: float = 1e7, theta_max: float | None = None,
                 num_neurons: int = 400, wta: bool = False):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = num_neurons
        self.wta: bool = wta   # hard winner-take-all per timestep (the base file's commented-out block); off = baseline behaviour
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = theta_plus
        self.theta_decay: float = math.exp(-1.0 / theta_decay_tau_steps)
        self.theta_max: float | None = theta_max
        self.v:     torch.Tensor | None = None
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

    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor, learning: bool = True) -> torch.Tensor:
        if self.v is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")
        self.v = self.decay * (self.v - self.E_rest) + self.E_rest
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh
        self.raw_v: torch.Tensor = self.v.clone()
        spikes: torch.Tensor = (self.v >= self.theta).float()
        if self.wta:
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
            if self.theta_max is not None:
                self.theta.clamp_(max=self.theta_max)
        return spikes


# =====================================================================================================================
# Layer 2: the Diehl & Cook layer of test4.py, verbatim except that the number of inputs (784), the row-normalisation
# target (78.4) and the weight ceiling (1.0) are constructor arguments.
# =====================================================================================================================
class DiehlAndCookNetwork(nn.Module):
    def __init__(self, n_inputs: int = 784, num_neurons: int = 400, theta_plus: float = 0.05, inhibition_mV: float = 200.0,
                 theta_decay_tau_steps: float = 1e7, theta_max: float | None = None, w_norm: float = 78.4, w_max: float = 1.0, wta: bool = False):
        super(DiehlAndCookNetwork, self).__init__()
        self.n_inputs: int = n_inputs
        self.num_neurons: int = num_neurons
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

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.neurons.reset(batch_size, device)
        self.inh_matrix  = self.inh_matrix.to(device)
        self.prev_spikes = torch.zeros(batch_size, self.num_neurons, device=device)
        self.x_trace     = torch.zeros(batch_size, self.n_inputs, device=device)
        self.y_trace     = torch.zeros(batch_size, self.num_neurons, device=device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        x_exc: torch.Tensor = self.synapses(input_spikes)
        x_inh: torch.Tensor = torch.matmul(self.prev_spikes, self.inh_matrix)
        current_spikes: torch.Tensor = self.neurons(x_exc, x_inh, learning=learning)
        if learning:
            self.x_trace = (self.x_trace * self.decay_x) + input_spikes
            self.y_trace = (self.y_trace * self.decay_y) + current_spikes
            batch_size: int = input_spikes.shape[0]
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace) / batch_size
            delta_w_plus: torch.Tensor = (self.lr_plus ) * post_interaction * (self.w_max - self.synapses.weight)
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes) / batch_size
            delta_w_minus = (self.lr_minus ) * pre_interaction * self.synapses.weight
            self.synapses.weight.add_(delta_w_plus - delta_w_minus)
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(self.w_norm / weight_sum)
        self.prev_spikes = current_spikes
        return current_spikes


# =====================================================================================================================
# Layer 1: the feature layer (new)
# =====================================================================================================================
class FeatureLayer(nn.Module):
    """P window positions x F feature neurons on a 28x28 input. See the module docstring for the design.

    Shapes: patches (B, P, K); templates W (F, K) if shared else (P*F, K); membrane v, spikes, traces (B, P, F);
    theta (P*F,) per neuron or (F,) per feature. Output: spikes flattened to (B, P*F), neuron index = p*F + f.
    """
    def __init__(self, patch: int = 7, stride: int = 3, n_features: int = 16, shared: bool = True,
                 theta_plus: float = 0.1, theta_decay_tau_steps: float = 1e7, theta_shared: bool = False,
                 w_norm: float = 9.8, w_max: float = 1.0, lr_plus: float = 1e-2, lr_minus: float = 1e-4,
                 inhibition_mV: float = 200.0, stdp_reduce: str = "sum", image_side: int = 28):
        super().__init__()
        self.patch, self.stride, self.F, self.shared = patch, stride, n_features, shared
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
        for py in range(self.side):
            for px in range(self.side):
                base = (py * stride) * image_side + px * stride
                idx.append([base + r * image_side + c for r in range(patch) for c in range(patch)])
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))          # (P, K)
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
        self.theta: torch.Tensor = torch.ones(self.F if theta_shared else self.N) * self.theta_base
        self.v: torch.Tensor = torch.empty(0); self.prev_spikes: torch.Tensor = torch.empty(0)
        self.x_trace: torch.Tensor = torch.empty(0); self.y_trace: torch.Tensor = torch.empty(0)

    @property
    def num_neurons(self) -> int:
        return self.N

    def templates(self) -> torch.Tensor:
        """(n_templates, K) weights; the template of neuron (p, f) is row f (shared) or row p*F+f (unshared)."""
        return self.W.detach()

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.theta = self.theta.to(device)
        self.v = torch.ones(batch_size, self.P, self.F, device=device) * self.E_rest
        self.prev_spikes = torch.zeros(batch_size, self.P, self.F, device=device)
        self.x_trace = torch.zeros(batch_size, self.P, self.K, device=device)
        self.y_trace = torch.zeros(batch_size, self.P, self.F, device=device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        B = input_spikes.shape[0]
        patches = input_spikes[:, self.idx]                                            # (B, P, K): the 49 pixels of each window
        if self.shared:
            x_exc = torch.matmul(patches, self.W.t())                                  # (B, P, F): template f against every window
        else:
            x_exc = torch.einsum("bpk,pfk->bpf", patches, self.W.view(self.P, self.F, self.K))
        # lateral inhibition WITHIN a position: a neuron is hit by every other feature at its position that fired last step
        ps = self.prev_spikes
        x_inh = self.inhibition * (ps.sum(dim=2, keepdim=True) - ps)                  # (B, P, F)
        # ---- LIF, same equations as excitatory_neurons.forward ----
        self.v = self.decay * (self.v - self.E_rest) + self.E_rest
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh
        theta_b = self.theta.view(1, 1, self.F) if self.theta_shared else self.theta.view(1, self.P, self.F)
        spikes = (self.v >= theta_b).float()
        self.v[spikes == 1.0] = self.E_rest
        if learning:
            if self.theta_shared:
                self.theta += spikes.sum(dim=0).mean(dim=0) * self.theta_plus / 16      # (F,): the feature's spikes, averaged over positions
            else:
                self.theta += spikes.sum(dim=0).reshape(-1) * self.theta_plus / 16     # (P*F,): each neuron's own spikes
            # ---- STDP, the D&C rule per neuron, reduced into the templates ----
            self.x_trace = self.x_trace * self.decay_x + patches
            self.y_trace = self.y_trace * self.decay_y + spikes
            if self.shared:
                post = torch.einsum("bpf,bpk->fk", spikes, self.x_trace)              # (F, K): potentiation requests, summed over positions
                pre = torch.einsum("bpf,bpk->fk", self.y_trace, patches)
                div = B * (self.P if self.stdp_reduce == "mean" else 1)
                W = self.W
                dW = self.lr_plus * (post / div) * (self.w_max - W) - self.lr_minus * (pre / div) * W
            else:
                post = torch.einsum("bpf,bpk->pfk", spikes, self.x_trace) / B          # (P, F, K): each neuron its own
                pre = torch.einsum("bpf,bpk->pfk", self.y_trace, patches) / B
                W = self.W.view(self.P, self.F, self.K)
                dW = (self.lr_plus * post * (self.w_max - W) - self.lr_minus * pre * W).view(self.N, self.K)
            with torch.no_grad():
                self.W.add_(dW)
                self.W.clamp_(min=0.0, max=self.w_max)
                self.W.mul_(self.w_norm / self.W.sum(dim=1, keepdim=True))
        self.prev_spikes = spikes
        return spikes.reshape(B, self.N)


def pool2x2(spikes: torch.Tensor, side: int, F: int) -> torch.Tensor:
    """Spike-OR over 2x2 blocks of positions per feature: (B, side*side*F) -> (B, (side//2)*(side//2)*F)."""
    B = spikes.shape[0]
    s = spikes.view(B, side, side, F)[:, : side // 2 * 2, : side // 2 * 2, :]
    s = s.view(B, side // 2, 2, side // 2, 2, F).amax(dim=(2, 4))
    return s.reshape(B, -1)


class TwoLayerNetwork(nn.Module):
    """Layer 1 (optional) -> optional 2x2 pooling -> layer 2. Exposes the single-layer interface the estimators use:
    reset(batch_size, device), forward(input_spikes, learning) -> layer-2 spikes, num_neurons, synapses (layer 2)."""
    def __init__(self, l1: FeatureLayer | None, l2: DiehlAndCookNetwork, pool2: bool = False):
        super().__init__()
        self.l1, self.l2, self.pool2 = l1, l2, pool2
        self.learn_l1: bool = l1 is not None     # stage A: True; frozen afterwards
        self.learn_l2: bool = True
        self.last_l1_spikes: torch.Tensor | None = None   # (B, N1) of the last timestep (for the stage-B usage stats)

    @property
    def num_neurons(self) -> int:
        return self.l2.num_neurons

    @property
    def synapses(self):
        return self.l2.synapses

    @property
    def neurons(self):
        return self.l2.neurons

    def l2_input_size(self) -> int:
        if self.l1 is None: return 784
        return (self.l1.side // 2) ** 2 * self.l1.F if self.pool2 else self.l1.N

    def reset(self, batch_size: int, device: torch.device) -> None:
        if self.l1 is not None: self.l1.reset(batch_size, device)
        self.l2.reset(batch_size, device)

    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
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


# =====================================================================================================================
# Helpers
# =====================================================================================================================
def template_sharpness(W: torch.Tensor, k: int) -> torch.Tensor:
    """Share of each row's mass on its k largest entries (CPU, float32). k = n_inputs / 7.84 keeps the flat-row value at ~0.23."""
    Wc = W.detach().float().cpu()
    return Wc.topk(k, dim=1).values.sum(dim=1) / Wc.sum(dim=1)


def shifted_test_set(n: int, shift: int, seed: int = 1234) -> TensorDataset:
    """The first n test images, each translated by a fixed random (dx, dy) in [-shift, shift]^2 (zero fill)."""
    g = np.random.default_rng(seed)
    imgs = torch.stack([test_dataset[i][0] for i in range(n)])             # (n, 1, 28, 28)
    labels = torch.tensor([int(test_dataset[i][1]) for i in range(n)])
    out = torch.zeros_like(imgs)
    for i in range(n):
        dx, dy = int(g.integers(-shift, shift + 1)), int(g.integers(-shift, shift + 1))
        src = imgs[i, 0]
        ys, ye = max(0, dy), min(28, 28 + dy); xs, xe = max(0, dx), min(28, 28 + dx)
        out[i, 0, ys:ye, xs:xe] = src[ys - dy:ye - dy, xs - dx:xe - dx]
    return TensorDataset(out, labels)


# =====================================================================================================================
# Stage A: train layer 1 alone
# =====================================================================================================================
def train_layer1(model: TwoLayerNetwork, loader: DataLoader, device: torch.device, cfg: dict) -> dict:
    l1 = model.l1; assert l1 is not None
    num_steps = cfg["num_steps"]; snap_every = cfg["l1_snapshot"]; max_examples = cfg["l1_examples"]
    k1 = max(3, int(round(l1.K / 7.84)))
    hist = {"points": [], "sharp": [], "theta_med": [], "theta_p90": [], "usage": [], "dW": [], "active_frac": [], "latency": [], "reservoir": []}
    W_prev = l1.templates().clone()
    usage = torch.zeros(l1.F, device=device)             # spikes per feature since the last snapshot
    n_windows_ink = 0; n_windows_fired = 0; lat_sum = 0.0
    seen = 0; batch_i = 0; t0 = time.time(); converged_at = None
    model.learn_l1 = True; model.learn_l2 = False
    for data, targets in loader:
        B = data.shape[0]; data = data.to(device)
        spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
        l1.reset(B, device)
        fired = torch.zeros(B, l1.P, dtype=torch.bool, device=device)
        first_t = torch.full((B, l1.P), float("nan"), device=device)
        ink = (spike_data.sum(0)[:, l1.idx].sum(2) > 0)                                       # (B, P): windows with any input spike
        for t in range(num_steps):
            s = l1(spike_data[t], learning=True).view(B, l1.P, l1.F)
            any_f = s.sum(2) > 0
            newly = any_f & ~fired
            first_t = torch.where(newly, torch.full_like(first_t, float(t)), first_t)
            fired |= any_f
            usage += s.sum(dim=(0, 1))
        n_windows_ink += int(ink.sum()); n_windows_fired += int((fired & ink).sum())
        lat_sum += float(torch.nan_to_num(first_t[fired & ink], nan=0.0).sum())
        seen += B; batch_i += 1
        if batch_i % snap_every == 0:
            W = l1.templates(); sharp = template_sharpness(W, k1).numpy()
            dW = torch.linalg.norm((W - W_prev).float().cpu(), dim=1).numpy(); W_prev = W.clone()
            th = l1.theta.detach().float().cpu().numpy()
            res = int((sharp < 0.35).sum())
            hist["points"].append(seen); hist["sharp"].append(sharp.astype(np.float32)); hist["dW"].append(dW.astype(np.float32))
            hist["theta_med"].append(float(np.median(th))); hist["theta_p90"].append(float(np.percentile(th, 90)))
            hist["usage"].append((usage / max(1, snap_every * B)).cpu().numpy().astype(np.float32))   # spikes per example, per feature
            hist["active_frac"].append(n_windows_fired / max(1, n_windows_ink)); hist["latency"].append(lat_sum / max(1, n_windows_fired))
            hist["reservoir"].append(res)
            print(f"[layer 1] {seen:>6d} ex | flat templates {res:3d}/{len(sharp)} | sharp med {np.median(sharp):.2f} | θ med {np.median(th):6.1f} p90 {np.percentile(th, 90):6.1f} | "
                  f"windows fired {100 * hist['active_frac'][-1]:4.0f}% latency {hist['latency'][-1]:5.1f} ms | ‖ΔW‖ max {dW.max():.2f} | {(time.time() - t0) / 60:4.1f} min")
            usage.zero_(); n_windows_ink = n_windows_fired = 0; lat_sum = 0.0
            if res == 0 and converged_at is None: converged_at = seen
            if (cfg["l1_stop_stationary"] and len(hist["reservoir"]) >= 3 and all(r == 0 for r in hist["reservoir"][-3:])
                    and abs(hist["theta_med"][-1] - hist["theta_med"][-3]) < 0.3):
                print(f"[layer 1] stationary: reservoir 0 for 3 snapshots and median theta moved < 0.3 mV over them -> freezing at {seen} examples"); break
        if seen >= max_examples: break
    model.learn_l1 = False   # FREEZE
    hist["converged_at"] = converged_at; hist["frozen_at"] = seen; hist["minutes"] = (time.time() - t0) / 60
    return hist


# =====================================================================================================================
# Stage B: train layer 2 (the test4 measurement set, single i.i.d. stream) + the shifted evaluation
# =====================================================================================================================
def train_layer2(model: TwoLayerNetwork, train_loader, assign_loader, eval_loader, shifted_loader, device, cfg: dict, run_dir: Path, l1_hist: dict | None):
    n = model.num_neurons; num_epochs, num_steps, snapshot_interval = cfg["num_epochs"], cfg["num_steps"], cfg["snapshot_interval"]
    k2 = max(3, int(round(model.l2.n_inputs / 7.84)))
    P, A, S, R, H, C, L = [], [], [], [], [], [], []             # points, acc, shifted acc, reservoir, hijack, conf, assignments
    WS, TH, SH, RC, L1U = [], [], [], [], []                       # window spikes, theta, sharpness, recruit steps, layer-1 usage
    recruit_step = np.full(n, np.nan)
    W_prev = model.l2.synapses.weight.detach().clone()
    spikes_since = torch.zeros(n, device=device); l1_usage = torch.zeros(model.l1.F, device=device) if model.l1 is not None else None
    total = 0; gbi = 0; t0 = time.time()
    model.learn_l1 = False; model.learn_l2 = True
    for epoch in range(num_epochs):
        for data, targets in train_loader:
            B = data.shape[0]; data = data.to(device)
            spike_data = spikegen.rate(data * 0.1, num_steps=num_steps).view(num_steps, B, 784)   # type: ignore
            model.reset(B, device)
            bts = torch.zeros(B, n, device=device)
            for t in range(num_steps):
                out = model(spike_data[t])
                bts += out
                if l1_usage is not None: l1_usage += model.last_l1_spikes.view(B, -1, model.l1.F).sum(dim=(0, 1))   # type: ignore
            spikes_since += bts.sum(0); total += B
            if gbi % snapshot_interval == 0:
                labels = assign_neuron_labels(model, assign_loader, device, num_steps)
                acc, cm = evaluate_network(model, eval_loader, labels, device, num_steps, return_matrix=True)   # type: ignore
                acc_s = evaluate_network(model, shifted_loader, labels, device, num_steps) if shifted_loader is not None else float("nan")
                with torch.no_grad():
                    W = model.l2.synapses.weight.detach()
                    sharp = template_sharpness(W, k2).numpy(); th = model.l2.neurons.theta.detach().float().cpu().numpy().copy()   # type: ignore
                    dw = torch.linalg.norm((W - W_prev).float().cpu(), dim=1); tot = float(dw.sum())
                    hij = float(torch.topk(dw, min(20, n)).values.sum()) / tot if tot > 0 else float("nan"); W_prev = W.clone()
                    ws = spikes_since.cpu().numpy().copy(); spikes_since.zero_()
                newly = (sharp >= 0.35) & np.isnan(recruit_step); recruit_step[newly] = total
                res = int((sharp < 0.35).sum())
                P.append(total); A.append(float(acc)); S.append(float(acc_s)); R.append(res); H.append(hij)
                C.append(np.asarray(cm).astype(np.int32)); L.append(labels.cpu().numpy().astype(np.int16))
                WS.append(ws.astype(np.float32)); TH.append(th.astype(np.float32)); SH.append(sharp.astype(np.float32)); RC.append(recruit_step.copy())
                if l1_usage is not None: L1U.append(l1_usage.cpu().numpy().astype(np.float32)); l1_usage.zero_()
                distinct = float((bts > 0).sum(dim=1).float().mean())
                l1_rate = (float(L1U[-1].sum()) / max(1, snapshot_interval * B)) if L1U else float("nan")
                print(f"[layer 2] {total:>7d} ex | held-out acc {acc:5.1f}% shifted {acc_s:5.1f}% | reservoir {res:3d} | hijacker idx {hij:4.2f} | "
                      f"L2 spikes/presentation {ws.sum() / max(1, snapshot_interval * B):5.1f} from {distinct:4.1f} distinct neurons (last batch) | L1 spikes/presentation {l1_rate:5.1f} | {(time.time() - t0) / 60:5.1f} min")
            gbi += 1
    print("Training complete")
    spike_time = spike_time_from_curve(P, A)
    out = dict(cfg); out.update({
        "spike_time": spike_time, "intervention_step_actual": None, "anneal_step_actual": None,
        "theta_base": -52.0, "theta_plus": float(model.l2.neurons.theta_plus), "inhibition_mV": float(model.l2.inh_matrix[0, 1].item()),
        "untrained_threshold": 0.35, "top_k": k2, "n_refill": 0, "theta_plus_late": None,
        "num_neurons": n, "l2_inputs": model.l2.n_inputs, "device": str(device), "minutes": (time.time() - t0) / 60,
        "final_accuracy": float(np.mean(A[-3:])), "final_accuracy_shifted": float(np.nanmean(S[-3:])),
        "l1": None if l1_hist is None else {"converged_at": l1_hist["converged_at"], "frozen_at": l1_hist["frozen_at"], "minutes": l1_hist["minutes"],
                                            "P": model.l1.P, "F": model.l1.F, "K": model.l1.K, "side": model.l1.side},   # type: ignore
    })
    empty = np.zeros(0, dtype=np.float32)
    npz = dict(points=np.array(P), accuracy=np.array(A), accuracy_shifted=np.array(S),
               cooper_points=empty, cooper_all=empty, cooper_proportion=empty,
               reservoir_count=np.array(R), hijack_index=np.array(H), window_spikes=np.array(WS), theta_snap=np.array(TH), sharp_snap=np.array(SH),
               assignments_snap=np.array(L), conf_snap=np.array(C), recruit_step_snap=np.array(RC),
               imprinted=np.zeros(n, bool), refilled=np.zeros(n, bool),
               W2=model.l2.synapses.weight.detach().float().cpu().numpy().astype(np.float16))
    if model.l1 is not None and l1_hist is not None:
        npz.update(W1=model.l1.templates().float().cpu().numpy().astype(np.float32), l1_theta=model.l1.theta.detach().float().cpu().numpy().astype(np.float32),
                   l1_usage_snap=np.array(L1U), l1_points=np.array(l1_hist["points"]), l1_sharp=np.array(l1_hist["sharp"]), l1_dW=np.array(l1_hist["dW"]),
                   l1_theta_med=np.array(l1_hist["theta_med"]), l1_theta_p90=np.array(l1_hist["theta_p90"]), l1_usage=np.array(l1_hist["usage"]),
                   l1_active_frac=np.array(l1_hist["active_frac"]), l1_latency=np.array(l1_hist["latency"]), l1_reservoir=np.array(l1_hist["reservoir"]))
    np.savez(run_dir / "run.npz", **npz)
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    make_run_figures(run_dir)
    print(f"\nrun finished in {out['minutes']:.1f} min | final acc {out['final_accuracy']:.1f}% shifted {out['final_accuracy_shifted']:.1f}% | spike time {spike_time if spike_time == spike_time else 'none'} | {run_dir}")


# =====================================================================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- layer 1 ----
    p.add_argument("--no-l1", action="store_true", help="single-layer reference: pixels straight to layer 2 (784 inputs, norm 78.4, wmax 1)")
    p.add_argument("--patch", type=int, default=7); p.add_argument("--stride", type=int, default=3); p.add_argument("--features", type=int, default=16)
    p.add_argument("--unshared", action="store_true", help="every layer-1 neuron its own template (locally connected) instead of shared templates")
    p.add_argument("--l1-theta-plus", type=float, default=0.1); p.add_argument("--l1-theta-tau", type=float, default=2000.0, help="layer-1 theta decay in examples (0 = off)")
    p.add_argument("--l1-theta-shared", action="store_true", help="one theta per feature (global conscience) instead of per neuron")
    p.add_argument("--l1-norm", type=float, default=9.8, help="layer-1 template row sum (mean weight 0.2 over 49 inputs)"); p.add_argument("--l1-wmax", type=float, default=1.0)
    p.add_argument("--l1-inhibition", type=float, default=200.0); p.add_argument("--l1-stdp-reduce", choices=["sum", "mean"], default="sum")
    p.add_argument("--l1-examples", type=int, default=9600, help="stage A length (max)"); p.add_argument("--l1-snapshot", type=int, default=10, help="batches between layer-1 snapshots")
    p.add_argument("--l1-stop-stationary", action="store_true", default=True); p.add_argument("--l1-no-early-stop", dest="l1_stop_stationary", action="store_false")
    p.add_argument("--pool2", action="store_true", help="2x2 spike-OR pooling between the layers")
    # ---- layer 2 ----
    p.add_argument("--neurons", type=int, default=400); p.add_argument("--theta-plus", type=float, default=0.5); p.add_argument("--theta-tau", type=float, default=None)
    p.add_argument("--inhibition", type=float, default=200.0)
    p.add_argument("--l2-wta", action="store_true", help="hard winner-take-all per timestep in layer 2 (resolves simultaneous crossings caused by layer 1's synchronous bursts); off = baseline behaviour"); p.add_argument("--l2-norm", type=float, default=None, help="default 0.1 x inputs (78.4 for pixels)")
    p.add_argument("--l2-wmax", type=float, default=None, help="default 4.0 with a feature layer, 1.0 for --no-l1")
    # ---- run control ----
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--shift", type=int, default=2, help="max |dx|,|dy| of the shifted test set (0 = none)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=350); p.add_argument("--snapshot-interval", type=int, default=100)
    p.add_argument("--tag", default=""); p.add_argument("--smoke", action="store_true"); p.add_argument("--replot", default=None, metavar="RUN_DIR")
    args = p.parse_args()
    if args.replot:
        print("figures regenerated:", make_run_figures(args.replot)); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.backends.mps.is_available(): device = torch.device("mps"); print("Using MPS.")
    elif torch.cuda.is_available(): device = torch.device("cuda"); print("Using CUDA.")
    else: device = torch.device("cpu"); print("Using CPU.")

    if args.smoke:
        train_subset = Subset(train_dataset, range(0, 32 * 20)); assign_subset = Subset(train_dataset, range(0, 128)); eval_subset = Subset(test_dataset, range(0, 64))
        assign_bs = eval_bs = 128; args.epochs, args.steps, args.snapshot_interval = 1, 25, 5
        args.l1_examples, args.l1_snapshot = 32 * 10, 2; n_shift = 64
    else:
        train_subset = train_dataset; assign_subset = Subset(train_dataset, range(0, 2000)); eval_subset = Subset(test_dataset, range(0, 1000))
        assign_bs = eval_bs = 512; n_shift = 1000
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=gen)
    l1_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1000))
    assign_loader = DataLoader(assign_subset, batch_size=assign_bs, shuffle=False); eval_loader = DataLoader(eval_subset, batch_size=eval_bs, shuffle=False)
    shifted_loader = DataLoader(shifted_test_set(n_shift, args.shift), batch_size=eval_bs, shuffle=False) if args.shift > 0 else None

    # ---- build ----
    if args.no_l1:
        l1 = None; n_in = 784; l2_norm = args.l2_norm or 78.4; l2_wmax = args.l2_wmax or 1.0
    else:
        tau1 = 1e7 if args.l1_theta_tau <= 0 else args.l1_theta_tau * args.steps / 32
        l1 = FeatureLayer(args.patch, args.stride, args.features, not args.unshared, args.l1_theta_plus, tau1, args.l1_theta_shared,
                          args.l1_norm, args.l1_wmax, 1e-2, 1e-4, args.l1_inhibition, args.l1_stdp_reduce)
        n_in = (l1.side // 2) ** 2 * l1.F if args.pool2 else l1.N
        l2_norm = args.l2_norm or 0.1 * n_in; l2_wmax = args.l2_wmax or 4.0
    tau2 = 1e7 if args.theta_tau is None else args.theta_tau * args.steps / 32
    l2 = DiehlAndCookNetwork(n_in, args.neurons, args.theta_plus, args.inhibition, tau2, None, l2_norm, l2_wmax, wta=bool(args.l2_wta))
    model = TwoLayerNetwork(l1, l2, args.pool2).to(device)

    variant = "singlelayer" if args.no_l1 else (f"{'unshared' if args.unshared else 'shared'}_p{args.patch}s{args.stride}f{args.features}" + ("_pool2" if args.pool2 else ""))
    run_name = f"{variant}_n{args.neurons}_tp{args.theta_plus:g}" + (f"_tau{args.theta_tau / 1000:g}k" if args.theta_tau else "_nodecay") + f"_ep{args.epochs}_seed{args.seed}"
    if args.tag: run_name += f"_{args.tag}"
    if args.smoke: run_name += "_SMOKE"
    run_name += "_" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"arm": "baseline", "sweep": True, "multi": True, "seed": args.seed, "num_epochs": args.epochs, "num_steps": args.steps, "snapshot_interval": args.snapshot_interval,
           "batch_size": 32, "smoke": bool(args.smoke), "tag": args.tag, "run_name": run_name, "init": "uniform", "theta_init": None, "theta_tau": args.theta_tau,
           "theta_plus_arg": args.theta_plus, "inhibition_arg": args.inhibition, "theta_max": None, "anneal_at": None, "theta_plus_late": None,
           "l2_wta": bool(args.l2_wta), "no_l1": bool(args.no_l1), "patch": args.patch, "stride": args.stride, "features": args.features, "shared": not args.unshared, "pool2": bool(args.pool2),
           "l1_theta_plus": args.l1_theta_plus, "l1_theta_tau": args.l1_theta_tau, "l1_theta_shared": bool(args.l1_theta_shared), "l1_norm": args.l1_norm, "l1_wmax": args.l1_wmax,
           "l1_inhibition": args.l1_inhibition, "l1_stdp_reduce": args.l1_stdp_reduce, "l1_examples": args.l1_examples, "l1_snapshot": args.l1_snapshot,
           "l1_stop_stationary": bool(args.l1_stop_stationary), "l2_inputs": n_in, "l2_norm": l2_norm, "l2_wmax": l2_wmax, "shift": args.shift}
    print("config:", json.dumps(cfg))
    l1_hist = train_layer1(model, l1_loader, device, cfg) if l1 is not None else None
    train_layer2(model, train_loader, assign_loader, eval_loader, shifted_loader, device, cfg, run_dir, l1_hist)


if __name__ == "__main__":
    main()
