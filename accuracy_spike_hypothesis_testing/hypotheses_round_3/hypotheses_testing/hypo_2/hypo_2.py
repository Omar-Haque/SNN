"""
hypo_2.py -- Round 3, Hypothesis 2: why "number of weights changed" FALLS at the spike in our measurement and RISES
in Cooper's -- both measured on the same network, in the same run.

Base code: src/snn_normalized replicate_results.py (model classes and estimators copied verbatim: 400 neurons, batch 32,
theta_plus 0.05 (sum/16), theta_decay 1e7, lr 1e-2 / 1e-4, 200 mV lateral inhibition, hard WTA off, no theta clamp).
ONE addition inside DiehlAndCookNetwork.forward(), marked "hypo_2": the raw STDP update of the timestep is stored in
self.last_delta_w before it is applied. Nothing else about the dynamics changes.

THE TWO METRICS (and a third, per batch)
  (A) "STDP update" count   -- our round-1/2 metric: number of entries of the RAW STDP update delta_w = dw_plus - dw_minus
                               with |delta_w| > thr, per timestep, summed over the 350 timesteps of a batch. Measured
                               BEFORE clamping and before the row normalisation to 78.4. A row of delta_w is non-zero
                               only where the neuron just spiked (potentiation) or has a live post-synaptic trace
                               (depression), so this is essentially (output spikes) x (active pixels).
  (B) "net change" count    -- Cooper's metric as we understand it: number of weights whose STORED value differs
                               between before and after the timestep, |W_after - W_before| > thr, i.e. AFTER clamp and
                               normalisation. The normalisation rescales a whole row whenever anything in that row moved,
                               so every one of the 784 weights of an active neuron counts: this is ~784 x (rows active
                               at that timestep).
  (C) "net change per batch"-- same as (B) but between the weights at the START and the END of a batch (one comparison
                               per batch). Cooper's x-axis was "training batch number", so this is the most likely
                               literal reading of his "number of weights changed per update": ~784 x (distinct neurons
                               that fired at least once in the batch).

PREDICTION (reservoir / hijacker mechanism)
  Before the spike a few low-theta recruits fire very often (short inter-spike intervals), so total spikes per
  presentation are HIGH but concentrated in FEW neurons; after the spike the winners are trained neurons with higher
  theta (fewer spikes) and the 32 images of a batch go to many different neurons. Hence:
      (A) falls at the spike (fewer spikes), (B) and (C) rise at the spike (more distinct active rows).
  As a check, the run also records total output spikes per batch and the number of distinct active neurons per batch:
  (A) should track the former, (C) should equal 784 x the latter.

Thresholds: thr = 1e-6 (what round 2 used) for all three counts, plus a stricter 1e-5 for (B) and (C) so that float32
rounding of the row normalisation (which moves untouched rows by ~1e-7) can be seen to be irrelevant.

OUTPUTS  results/<run>/run.npz (every per-batch series), run.json (config, spike time, pre/post means), F1..F4 png.
USAGE    python hypo_2.py [--seed 0] [--epochs 3] [--thr 1e-6]      python hypo_2.py --smoke      python hypo_2.py --replot results/<run>
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
from torch.utils.data import DataLoader, Subset
from snntorch import spikegen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir: Path = Path(__file__).resolve().parent
repo_root: Path = script_dir.parents[3]            # .../cooper  (hypo_2 -> hypotheses_testing -> hypotheses_round_3 -> accuracy_spike_hypothesis_testing -> cooper)
DATA_ROOT: Path = repo_root / "data"               # the base file uses root='./data' relative to the CWD; this is CWD-independent
results_root: Path = script_dir / "results"

# Import MNIST train and test
train_dataset: datasets.MNIST = datasets.MNIST(
    root=str(DATA_ROOT),
    train=True,
    download=True,
    transform=transforms.ToTensor() # convert images to torch tensors
)

test_dataset: datasets.MNIST = datasets.MNIST(
    root=str(DATA_ROOT),
    train=False,
    download=True,
    transform=transforms.ToTensor() # convert images to torch tensors
)

# =====================================================================================================================
# Model: copied verbatim from src/snn_normalized replicate_results.py (hard WTA commented out, 200 mV inhibition),
# plus ONE line in DiehlAndCookNetwork.forward() marked "hypo_2" that stores the raw STDP update in self.last_delta_w.
# =====================================================================================================================
# EXCITATORY NEURONS
# Excitatory neurons are an entire class, but inhibitory neurons can be simulated using matrices.
class excitatory_neurons(nn.Module):
    def __init__(self):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = 400  # number of excitatory neurons.
        
        # For the adaptive threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        # CURRENT: Changed theta_plus changed from 0.5 to 0.05
        self.theta_plus:  float = 0.05                  # prev: 0.1. a penalty added to the minimum fire threshold each time a neuron fires.
        self.theta_decay: float = math.exp(-1.0 / 1e7)
        
        # State variables for leaky integrate-and-fire model. Used as memory for neuron states.
        self.v:     torch.Tensor | None = None     # self.v[i] is the ith neuron's voltage.
        self.g_e:   torch.Tensor | None = None     # self.g_e[i] is the ith neuron's excitatory conductance.
        self.g_i:   torch.Tensor | None = None     # self.g_i[i] is the ith neuron's inhibitory conductance.
        self.theta: torch.Tensor | None = (torch.ones(self.num_neurons) * self.theta_base) # the current, adaptive threshold for each neuron
        
        # Decay multipliers (tau)
        self.tau_m :   float = 100.0
        self.decay_ge: float = math.exp(-1.0 / 1.0)
        self.decay_gi: float = math.exp(-1.0 / 2.0)
        
        # Reverse potentials (in mV)
        self.E_rest: float = -65.0  # Baseline voltage for each neuron
        self.E_exc:  float = 0.0    # Ceiling for excitatory neurons
        self.E_inh:  float = -100.0 # Floor for inhibitory neurons
        
        self.decay = math.exp(-1.0 / 100)
    
    
    def reset(self, batch_size: int, device: torch.device) -> None:
        """
        Clears the memory of the neurons.
        The "neurons" come to when the variables self.v, self.theta, self.g_e, and self.g_i are interpreted together.
        Must be called before starting the time loop for a new batch of images.
        """
        self.v     = torch.ones(batch_size, self.num_neurons, device=device) * self.E_rest # shape is (batch_size, num_neurons)
        self.g_e   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.g_i   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.theta = self.theta.to(device) # type: ignore
        # NEW: 5 ms refractory period for excitatory neurons.
        # self.refractory_timer: torch.Tensor = torch.zeros((batch_size, self.num_neurons), device=self.v.device) # type: ignore

    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor, learning: bool = True) -> torch.Tensor:
        if self.v is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")
        
        # NEW: refractory countdown
        # self.refractory_timer = torch.clamp(self.refractory_timer - 1.0, min=0.0)
        # is_refractory = (self.refractory_timer > 0.0)
        
        self.v = self.decay * (self.v - self.E_rest) + self.E_rest
        
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh
        
        # NEW: Bring refractory neurons back to baseline voltage.
        # self.v[is_refractory] = self.E_rest
        
        # save a copy of the voltage to plot a voltage and theta vs timesteps plot for neurons
        self.raw_v: torch.Tensor = self.v.clone()
        
        spikes: torch.Tensor = (self.v >= self.theta).float()
        # spikes[is_refractory] = 0.0  # NEW: Cannot spike during refractory period
        
        # WTA
        # competing_v = self.v.clone()
        # competing_v[spikes == 0.0] = -float("inf")
        # max_v_indices: torch.Tensor = torch.argmax(competing_v, dim=1)
        # wta_mask = torch.zeros_like(spikes)
        # wta_mask.scatter_(1, max_v_indices.unsqueeze(1), 1.0)
        # spikes = spikes * wta_mask
        
        # ones = spikes.sum(dim=1)
        # morethanone = (ones > 1).any().item()
        # if (morethanone):
        #     print("MORE THAN 1 NEURON SPIKED ON A SINGLE TIMESTEP")
        
        # NEW: Add spiked neurons into a refractory period
        # newly_spiked: torch.Tensor = (spikes == 1.0)
        # self.refractory_timer[newly_spiked] = 5.0
        self.v[spikes == 1.0] = self.E_rest
        
        if learning:
            # CURRENT: Changed .mean to .sum
            self.theta += (spikes.sum(dim=0) * self.theta_plus) / 16
            # self.theta.clamp_(max=0.0)

        return spikes

# Architecture of the entire Diehl and Cook network
# STDP is also performed within this class
class DiehlAndCookNetwork(nn.Module):
    def __init__(self):
        super(DiehlAndCookNetwork, self).__init__()
        
        self.num_neurons: int = 400
        
        self.synapses: nn.Linear = nn.Linear(784, self.num_neurons, bias=False) # connect 784 pixels to each excitatory neuron.
        self.synapses.weight.requires_grad = False # Using STDP instead
        
        # STDP parameters
        self.w_max:    float = 1.0  # weight upper limit
        self.lr_plus:  float = 1e-2 # Learning rate for potentiation 0.01
        self.lr_minus: float = 1e-4 # Learning rate for depression 0.0001
        
        # Trace decay multipliers
        self.decay_x: float = math.exp(-1.0 / 20.0) # Pre-synaptic trace decay
        self.decay_y: float = math.exp(-1.0 / 20.0) # Post-synaptic trace decay
        
        # Initialize weights uniformly between a and b
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3)
        with torch.no_grad():
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
        
        self.neurons: excitatory_neurons = excitatory_neurons()
        
        # LATERAL INHIBITION USING A 100 x 100 MATRIX
        # A penalty of 200mV
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * 200.0
        # A neuron doesn't inhibit itself
        self.inh_matrix.fill_diagonal_(0.0)
        
        # Track the previous time step's spikes
        self.prev_spikes: torch.Tensor = torch.empty(0)
        
        # Trace memory state storage
        self.x_trace: torch.Tensor = torch.empty(0) # traces of input pixels
        self.y_trace: torch.Tensor = torch.empty(0) # traces of output neurons
        self.last_delta_w: torch.Tensor = torch.empty(0) # hypo_2: raw STDP update of the last timestep (set in forward)
        
    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset the entire network's memory for a new batch. Do not reset the synapses."""
        self.neurons.reset(batch_size, device)
        self.inh_matrix  = self.inh_matrix.to(device)
        self.prev_spikes = torch.zeros(batch_size, self.num_neurons, device=device)
        self.x_trace     = torch.zeros(batch_size, 784, device=device)
        self.y_trace     = torch.zeros(batch_size, self.num_neurons, device=device)
    
    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        # input spikes shape: (batch_size, 784)
        # Calculate excitatory current
        x_exc: torch.Tensor = self.synapses(input_spikes)
        
        # Calculate inhibitory current (penalties from the previous timestep)
        # Inhibit the neurons that spiked previously
        x_inh: torch.Tensor = torch.matmul(self.prev_spikes, self.inh_matrix)
        
        # Advance time
        current_spikes: torch.Tensor = self.neurons(x_exc, x_inh, learning=learning)
        
        # STDP
        if learning:
            # Update the traces
            self.x_trace = (self.x_trace * self.decay_x) + input_spikes   # did this pixel fire recently?
            self.y_trace = (self.y_trace * self.decay_y) + current_spikes # did this neuron fire recently?
            
            # When output fired, look back at input trace (potentiation)
            batch_size: int = input_spikes.shape[0]
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace) / batch_size
            delta_w_plus: torch.Tensor = (self.lr_plus ) * post_interaction * (self.w_max - self.synapses.weight)
            
            # When input fired, look back at output trace (depression)
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes) / batch_size
            delta_w_minus = (self.lr_minus ) * pre_interaction * self.synapses.weight
            
            # Apply changes to synapses
            # hypo_2: keep the raw STDP update of this timestep (before clamp and normalisation) so the training loop can count it
            self.last_delta_w = delta_w_plus - delta_w_minus
            self.synapses.weight.add_(self.last_delta_w) # delta_w = η(x_pre − x_tar)(w_max − w)^μ
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            # BindsNET weight normalization
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
            
        self.prev_spikes = current_spikes
        return current_spikes
        
        # current_spikes has shape (batch_size, num_neurons)
        # Row i represents the ith image in our batch
        # Column j represents the jth excitatory neuron
        # current_spikes[i][j] has a value of either 1.0 or 0.0
        # 1.0 indicates neuron j successfully fired at millisecond t (see training and assignment loop) while looking at image i.
        # 0.0 indicates neuron j did not fire when looking at image i.
    

# =====================================================================================================================
# Estimators: copied verbatim from src/snn_normalized replicate_results.py
# =====================================================================================================================
# Cooper / BindsNET-style accuracy estimate.
# Port of bindsnet.evaluation.{assign_labels, all_activity, proportion_weighting} and of the bookkeeping in
# bindsnet/examples/mnist/batch_eth_mnist.py. Differences from assign_neuron_labels()/evaluate_network() below:
#   * uses the spikes recorded DURING TRAINING (plasticity on), not a separate no-learning pass
#   * each window of `update_interval` training samples is scored with the assignments computed from the
#     PREVIOUS windows; `rates` accumulate over the whole run (alpha = 1.0), so assignments lag
#   * "all" votes = spikes per class / number of neurons assigned to that class (classes with few assigned
#     neurons are amplified); "proportion" additionally weights each neuron by its class proportion
#   * neurons that have never fired have all-zero rates -> NaN proportions -> set to 0 -> argmax gives class 0,
#     i.e. BindsNET silently assigns every dormant neuron to digit 0 (no "unassigned" state)
#   * the very first window is scored with assignments = -1 everywhere, which is why BindsNET curves start ~10%
# Costs nothing extra: no additional presentations are run.
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def bindsnet_assign_labels(spikes: torch.Tensor, labels: torch.Tensor, n_labels: int, rates: torch.Tensor | None = None, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # spikes: [n_samples, n_neurons] spike counts (already summed over time); labels: [n_samples]
    n_neurons = spikes.size(1)
    if rates is None:
        rates = torch.zeros((n_neurons, n_labels), device=spikes.device)
    for i in range(n_labels):
        n_labeled = torch.sum(labels == i).float()
        if n_labeled > 0:
            indices = torch.nonzero(labels == i).view(-1)
            rates[:, i] = alpha * rates[:, i] + (torch.sum(spikes[indices], 0) / n_labeled)
    proportions = rates / rates.sum(1, keepdim=True)
    proportions[proportions != proportions] = 0  # NaNs (never-fired neurons) -> 0, exactly as BindsNET does
    assignments = torch.max(proportions, 1)[1]
    return assignments, proportions, rates

@torch.no_grad()
def bindsnet_all_activity(spikes: torch.Tensor, assignments: torch.Tensor, n_labels: int) -> torch.Tensor:
    n_samples = spikes.size(0)
    rates = torch.zeros((n_samples, n_labels), device=spikes.device)
    for i in range(n_labels):
        n_assigns = torch.sum(assignments == i).float()
        if n_assigns > 0:
            indices = torch.nonzero(assignments == i).view(-1)
            rates[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns
    return torch.sort(rates, dim=1, descending=True)[1][:, 0]

@torch.no_grad()
def bindsnet_proportion_weighting(spikes: torch.Tensor, assignments: torch.Tensor, proportions: torch.Tensor, n_labels: int) -> torch.Tensor:
    n_samples = spikes.size(0)
    rates = torch.zeros((n_samples, n_labels), device=spikes.device)
    for i in range(n_labels):
        n_assigns = torch.sum(assignments == i).float()
        if n_assigns > 0:
            indices = torch.nonzero(assignments == i).view(-1)
            rates[:, i] += torch.sum((proportions[:, i] * spikes)[:, indices], 1) / n_assigns
    return torch.sort(rates, dim=1, descending=True)[1][:, 0]

@torch.no_grad()
def assign_neuron_labels(model: DiehlAndCookNetwork, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    print("Assigning class labels to neurons")
    model.eval()
    
    # Create a tensor to count spikes: shape [num_neurons neurons, 10 digit classes]
    spike_counts: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # Find the total sum of white pixels for each image in the batch
        # image_sums = data.sum(dim=(1, 2, 3), keepdim=True) + 1e-5
        # Force every image to have the exact same total sum of 75.0
        # data = (data / image_sums) * 75.0
        
        # Generate the 350ms presentation spikes
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        
        # Track the total spikes fired by each neuron over the 350ms window
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        # Run the time loop with learning disabled
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False) # shape: [batch_size, num_neurons]
            batch_neuron_spikes += out_spikes
            
        # For each image in the batch, map the spikes to its target label
        for digit in range(10):
            mask = (targets == digit)
            if mask.any():
                # Divide by the number of images in the mask to get the average
                spike_counts[:, digit] += batch_neuron_spikes[mask].sum(dim=0) / mask.sum().float()
                  
    # Find which digit caused each neuron to fire the most
    # neuron_assignments[i] will hold the integer digit (0-9) for the i-th neuron
    neuron_assignments: torch.Tensor = torch.argmax(spike_counts, dim=1)
    # Find neurons that fired 0 times
    total_spikes = spike_counts.sum(dim=1)
    dead_neurons = (total_spikes == 0)
    neuron_assignments[dead_neurons] = -1
    
    print("Neuron assignments complete")
    # for digit in range(10):
    #     count = (neuron_assignments == digit).sum().item()
        # print(f"Digit {digit} was assigned to {count} neurons.")
        
    dead_count = (neuron_assignments == -1).sum().item()
    print(f"Unassigned neurons: {dead_count}")
        
    return neuron_assignments

@torch.no_grad()
def evaluate_network(model: DiehlAndCookNetwork, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int, return_matrix: bool = False) -> float | tuple[float, np.ndarray]:
    print("Evaluating network accuracy on test set...")
    model.eval()

    correct_predictions: int = 0
    total_predictions:   int = 0
    
    # Initialize a 10x10 confusion matrix
    conf_matrix = torch.zeros(10, 10, dtype=torch.int32, device=device)
    
    # Create a one-hot mapping matrix [num_neurons, 10]
    # This matrix routes spikes from specific neurons into their assigned digit classes.
    assignment_matrix: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    # Scatter the 1.0s, ignoring the -1 dead neurons
    valid_mask = (neuron_assignments != -1)
    safe_assignments = torch.where(valid_mask, neuron_assignments, torch.zeros_like(neuron_assignments))
    assignment_matrix.scatter_(1, safe_assignments.unsqueeze(1), 1.0)
    assignment_matrix[~valid_mask] = 0.0 # Erase the temporary zeros used for dead neurons
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # Find the total sum of white pixels for each image in the batch
        # image_sums = data.sum(dim=(1, 2, 3), keepdim=True) + 1e-5
        # Force every image to have the exact same total sum of 75.0
        # data = (data / image_sums) * 75.0
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        
        # Track the total spikes fired by each neuron over the 350ms window
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        # Classify all images in the batch
        # Result shape: [batch_size, 10]
        class_votes: torch.Tensor = torch.matmul(batch_neuron_spikes, assignment_matrix)
        
        # The network's final guess is the class with the highest total spikes (evaluated across the whole batch)
        predictions: torch.Tensor = torch.argmax(class_votes, dim=1)
        
        # Populate the confusion matrix
        for true_label, predicted_label in zip(targets, predictions):
            conf_matrix[true_label.long(), predicted_label.long()] += 1
        
        # Count how many predictions match the targets
        correct_predictions += (predictions == targets).sum().item()
        total_predictions += batch_size
    accuracy = (correct_predictions / total_predictions) * 100.0
        
    if return_matrix:
        return accuracy, conf_matrix.cpu().numpy() # type: ignore
    return accuracy


# =====================================================================================================================
# Hypothesis-specific code
# =====================================================================================================================
SPIKE_LEVEL = 78.0   # spike time = first snapshot after which held-out accuracy never drops below this again


def spike_time_from_curve(points, acc, level=SPIKE_LEVEL):
    points, acc = np.asarray(points, float), np.asarray(acc, float)
    if len(acc) < 3:
        return float('nan')
    for i in range(len(acc) - 1):
        if np.all(acc[i:] >= level):
            return float(points[i])
    return float('nan')


def moving_average(y, k):
    """Centred moving average over k points; the window shrinks at the two ends instead of sagging."""
    y = np.asarray(y, float)
    if len(y) < k or k <= 1:
        return y
    h = k // 2
    c = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(len(y)); lo = np.maximum(0, i - h); hi = np.minimum(len(y), i + h + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def train(
    model: "DiehlAndCookNetwork",
    train_loader: DataLoader,
    snapshot_assign_loader: DataLoader,
    snapshot_eval_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    run_dir: Path,
) -> "DiehlAndCookNetwork":
    n: int = model.num_neurons
    num_epochs, num_steps, snapshot_interval = cfg["num_epochs"], cfg["num_steps"], cfg["snapshot_interval"]
    thr, thr_strict = cfg["thr"], cfg["thr_strict"]

    total_points_processed: int = 0
    global_batch_idx: int = 0

    # Held-out snapshot histories (every snapshot_interval batches)
    points_history, accuracy_history = [], []

    # Per-batch histories (one entry per batch). Each is the per-timestep quantity summed over the batch's 350 timesteps,
    # except the two "per batch" ones which are a single start-vs-end comparison.
    batch_points = []
    cnt_stdp_hist = []            # (A) |delta_w_stdp| > thr, summed over timesteps
    cnt_net_hist = []             # (B) |W_after - W_before| > thr, summed over timesteps
    cnt_net_strict_hist = []      # (B) with thr_strict
    rows_net_hist = []            # rows (neurons) with any |W_after - W_before| > thr_strict, summed over timesteps  -> (B) / 784
    fro_stdp_hist = []            # Frobenius norm of delta_w_stdp, summed over timesteps
    fro_net_hist = []             # Frobenius norm of W_after - W_before, summed over timesteps
    cnt_batch_hist = []           # (C) |W_end - W_start| > thr (one comparison per batch)
    cnt_batch_strict_hist = []    # (C) with thr_strict
    cnt_exact_hist = []           # (D) exact inequality W_after != W_before (no threshold), summed over timesteps
    cnt_batch_exact_hist = []     # (D) exact inequality W_end != W_start (no threshold), once per batch
    spikes_hist = []              # total output spikes in the batch (all images, all neurons)
    active_hist = []              # distinct neurons that fired at least once in the batch

    t_start = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            batch_size: int = data.shape[0]
            data, targets = data.to(device), targets.to(device)

            # Generate poisson spike trains.
            spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps)  # type: ignore
            # Flatten spatial dimensions: shape becomes [num_steps, batch_size, 784]
            spike_data = spike_data.view(num_steps, batch_size, 784)

            # Reset the network memory for a new batch
            model.reset(batch_size, device)

            # On-device accumulators (one .item()-style sync per batch, at the end)
            acc_cnt_stdp = torch.zeros((), device=device)
            acc_cnt_net = torch.zeros((), device=device)
            acc_cnt_net_strict = torch.zeros((), device=device)
            acc_rows_net = torch.zeros((), device=device)
            acc_fro_stdp = torch.zeros((), device=device)
            acc_fro_net = torch.zeros((), device=device)
            acc_cnt_exact = torch.zeros((), device=device)
            batch_train_spikes = torch.zeros(batch_size, n, device=device)
            W_start = model.synapses.weight.detach().clone()      # weights at the start of the batch, for (C)

            for t in range(num_steps):
                W_before = model.synapses.weight.detach().clone()  # weights at the start of this timestep
                out_spikes: torch.Tensor = model(spike_data[t])    # inside: STDP update -> clamp -> normalise to 78.4
                dW_stdp = model.last_delta_w                       # (A) raw STDP update of this timestep (before clamp / normalisation)
                dW_net = model.synapses.weight.detach() - W_before # (B) what actually changed in the stored weights
                batch_train_spikes += out_spikes
                acc_cnt_stdp += (dW_stdp.abs() > thr).sum()
                acc_cnt_net += (dW_net.abs() > thr).sum()
                acc_cnt_net_strict += (dW_net.abs() > thr_strict).sum()
                acc_rows_net += (dW_net.abs() > thr_strict).any(dim=1).sum()
                acc_fro_stdp += torch.linalg.norm(dW_stdp)
                acc_fro_net += torch.linalg.norm(dW_net)
                acc_cnt_exact += (dW_net != 0).sum()               # (D) any change at all, even one ulp of float32 rounding

            dW_batch = model.synapses.weight.detach() - W_start    # (C) start vs end of the batch
            per_neuron = batch_train_spikes.sum(dim=0)             # (n,) spikes of each neuron in this batch
            stats = torch.stack([
                acc_cnt_stdp, acc_cnt_net, acc_cnt_net_strict, acc_rows_net, acc_fro_stdp, acc_fro_net,
                (dW_batch.abs() > thr).sum().float(), (dW_batch.abs() > thr_strict).sum().float(),
                per_neuron.sum(), (per_neuron > 0).sum().float(),
                acc_cnt_exact, (dW_batch != 0).sum().float(),
            ]).cpu().numpy()                                       # one device->host sync per batch
            total_points_processed += batch_size
            batch_points.append(total_points_processed)
            cnt_stdp_hist.append(stats[0]); cnt_net_hist.append(stats[1]); cnt_net_strict_hist.append(stats[2]); rows_net_hist.append(stats[3])
            fro_stdp_hist.append(stats[4]); fro_net_hist.append(stats[5]); cnt_batch_hist.append(stats[6]); cnt_batch_strict_hist.append(stats[7])
            spikes_hist.append(stats[8]); active_hist.append(stats[9]); cnt_exact_hist.append(stats[10]); cnt_batch_exact_hist.append(stats[11])

            # Estimated accuracy evaluation (held-out, as in the base file)
            if global_batch_idx % snapshot_interval == 0:
                temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
                snapshot_acc = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps, return_matrix=False)
                points_history.append(total_points_processed); accuracy_history.append(float(snapshot_acc))  # type: ignore
                print(f"{total_points_processed:>7d} ex | held-out acc {float(snapshot_acc):5.1f}% | (A) stdp {stats[0]:9.0f} | (B) net {stats[2]:9.0f} | "  # type: ignore
                      f"(C) batch {stats[7]:7.0f} | (D) exact {stats[10]:10.0f} / batch {stats[11]:7.0f} | spikes {stats[8]:6.0f} | active {stats[9]:3.0f} | {(time.time() - t_start) / 60:5.1f} min")
            global_batch_idx += 1
    print("Training complete")

    spike_time = spike_time_from_curve(points_history, accuracy_history)
    bp = np.array(batch_points)
    series = {
        "cnt_stdp": np.array(cnt_stdp_hist), "cnt_net": np.array(cnt_net_hist), "cnt_net_strict": np.array(cnt_net_strict_hist),
        "rows_net": np.array(rows_net_hist), "fro_stdp": np.array(fro_stdp_hist), "fro_net": np.array(fro_net_hist),
        "cnt_batch": np.array(cnt_batch_hist), "cnt_batch_strict": np.array(cnt_batch_strict_hist),
        "spikes": np.array(spikes_hist), "active": np.array(active_hist),
        "cnt_exact": np.array(cnt_exact_hist), "cnt_batch_exact": np.array(cnt_batch_exact_hist),
    }
    # Pre/post-spike means (pre: 60k .. spike-5k, post: spike+10k .. end) for the summary and run.json
    prepost = {}
    if spike_time == spike_time:
        pre = (bp >= 60000) & (bp < spike_time - 5000); post = bp > spike_time + 10000
        for k, v in series.items():
            prepost[k] = {"pre": float(v[pre].mean()) if pre.any() else None, "post": float(v[post].mean()) if post.any() else None}
    out = dict(cfg)
    out.update({"spike_time": spike_time, "prepost_means": prepost,
                "theta_plus": float(model.neurons.theta_plus), "theta_decay_tc": float(-1.0 / math.log(model.neurons.theta_decay)),
                "inhibition_mV": float(model.inh_matrix[0, 1].item()), "lr_plus": float(model.lr_plus), "lr_minus": float(model.lr_minus),
                "num_neurons": n, "num_weights": int(n * model.synapses.weight.shape[1]), "device": str(device), "minutes": (time.time() - t_start) / 60})
    np.savez(run_dir / "run.npz", points=np.array(points_history), accuracy=np.array(accuracy_history), batch_points=bp, **series)
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    make_figures(run_dir)
    print(f"\nrun finished in {out['minutes']:.1f} min | spike time: {spike_time if spike_time == spike_time else 'none'} | outputs: {run_dir}")
    for k in ("cnt_stdp", "cnt_net_strict", "cnt_batch_strict", "cnt_exact", "cnt_batch_exact", "spikes", "active"):
        if k in prepost and prepost[k]["pre"] and prepost[k]["post"]:
            print(f"  {k:<16s} pre {prepost[k]['pre']:12.1f}   post {prepost[k]['post']:12.1f}   ratio {prepost[k]['post'] / prepost[k]['pre']:.2f}")
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Figures (numpy + matplotlib only; `--replot` re-runs this on a finished run)
# ---------------------------------------------------------------------------------------------------------------------
def make_figures(run_dir):
    run_dir = Path(run_dir)
    d = dict(np.load(run_dir / "run.npz")); cfg = json.loads((run_dir / "run.json").read_text())
    pts, acc, bp = d["points"], d["accuracy"], d["batch_points"]
    spike = spike_time_from_curve(pts, acc)
    K = 25                                                     # batches in the moving average (25 x 32 = 800 examples)
    thr, thr_s = cfg["thr"], cfg["thr_strict"]

    def footer(fig):
        fig.text(0.995, 0.004, f"baseline, seed {cfg['seed']}  ·  thr = {thr:g} (strict {thr_s:g})", ha='right', va='bottom', fontsize=8.5, color='dimgray')

    def acc_twin(ax):
        ax2 = ax.twinx(); ax2.plot(pts, acc, color='blue', alpha=0.35, linewidth=1.5, label='held-out accuracy'); ax2.set_ylabel('Accuracy (%)', color='blue')
        if spike == spike: ax.axvline(spike, color='green', linestyle=':', linewidth=1.5, label=f"spike at {spike / 1000:.0f}k")
        return ax2

    def legend_both(ax, ax2, loc):
        # legend below the axes so it never covers data
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)

    # ---- F1: the two per-timestep counts, summed per batch ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bp, d["cnt_stdp"], color='tab:purple', alpha=0.15, linewidth=0.5)
    ax.plot(bp, moving_average(d["cnt_stdp"], K), color='tab:purple', linewidth=2, label=f'(A) STDP update, before normalisation: |Δw| > {thr:g}  [our round-2 metric]')
    ax.plot(bp, d["cnt_net_strict"], color='tab:red', alpha=0.15, linewidth=0.5)
    ax.plot(bp, moving_average(d["cnt_net_strict"], K), color='tab:red', linewidth=2, label=f'(B) net change, after normalisation: |ΔW| > {thr_s:g}')
    ax.plot(bp, moving_average(d["cnt_net"], K), color='tab:red', linewidth=1, linestyle='--', label=f'(B) same with |ΔW| > {thr:g}  (float-noise check)')
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('Weights changed per timestep, summed over the batch'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = acc_twin(ax); legend_both(ax, ax2, 'center right')
    ax.set_title('Weights changed per timestep: raw STDP update vs. stored weights', loc='left', fontsize=12, fontweight='bold')
    footer(fig); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F1_per_timestep_counts.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- F2: per-batch count (Cooper's likely metric) vs 784 x distinct active neurons ---------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bp, d["cnt_batch_strict"], color='tab:red', alpha=0.15, linewidth=0.5)
    ax.plot(bp, moving_average(d["cnt_batch_strict"], K), color='tab:red', linewidth=2, label=f'(C) weights changed per batch (start vs end, after normalisation): |ΔW| > {thr_s:g}')
    ax.plot(bp, moving_average(784 * d["active"], K), color='black', linewidth=1.2, linestyle='--', label='784 × distinct neurons that fired in the batch')
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('Weights changed per batch'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = acc_twin(ax); legend_both(ax, ax2, 'center right')
    ax.set_title("Weights changed per batch (Cooper's 'per update' metric as we read it)", loc='left', fontsize=12, fontweight='bold')
    footer(fig); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F2_per_batch_count.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- F3: Frobenius norms --------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bp, moving_average(d["fro_stdp"], K), color='tab:purple', linewidth=2, label='(A) ‖Δw_STDP‖_F summed over timesteps')
    ax.plot(bp, moving_average(d["fro_net"], K), color='tab:red', linewidth=2, label='(B) ‖W_after − W_before‖_F summed over timesteps')
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('Frobenius norm per batch'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = acc_twin(ax); legend_both(ax, ax2, 'center right')
    ax.set_title('Size of the weight change per batch', loc='left', fontsize=12, fontweight='bold')
    footer(fig); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F3_frobenius.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- F4: the drivers: total spikes per batch and distinct active neurons per batch -----------------------------
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax.plot(bp, d["spikes"], color='tab:orange', alpha=0.15, linewidth=0.5)
    ax.plot(bp, moving_average(d["spikes"], K), color='tab:orange', linewidth=2, label='output spikes per batch (32 images)')
    ax.set_ylabel('Spikes per batch'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = acc_twin(ax); legend_both(ax, ax2, 'center right')
    ax.set_title('What drives the two counts', loc='left', fontsize=12, fontweight='bold')
    bx.plot(bp, d["active"], color='black', alpha=0.15, linewidth=0.5)
    bx.plot(bp, moving_average(d["active"], K), color='black', linewidth=2, label='distinct neurons that fired in the batch')
    bx.set_ylabel('Active neurons per batch'); bx.set_xlabel('Number of training examples'); bx.grid(True, linestyle='--', alpha=0.5)
    bx2 = acc_twin(bx); legend_both(bx, bx2, 'center right')
    footer(fig); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F4_drivers.png', dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- F5: exact-inequality counts (no threshold) ----------------------------------------------------------------
    # If the per-timestep exact count sits near num_weights x 350 (~1.1e8 per batch), float32 rounding of the row
    # normalisation moves untouched rows too and an exact count is noise. If it sits near 784 x rows touched, it is a
    # participation count and should rise with the distinct-active-neuron curve.
    if "cnt_exact" in d:
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        ax.plot(bp, d["cnt_exact"], color='tab:brown', alpha=0.15, linewidth=0.5)
        ax.plot(bp, moving_average(d["cnt_exact"], K), color='tab:brown', linewidth=2, label='(D) W_after != W_before, per timestep, summed over the batch')
        ax.plot(bp, moving_average(784 * d["rows_net"], K), color='black', linewidth=1.2, linestyle='--', label='784 × rows with |ΔW| > strict thr, summed over timesteps')
        ax.axhline(cfg.get('num_weights', 313600) * cfg['num_steps'], color='gray', linewidth=0.8, linestyle=':', label='every weight, every timestep (pure noise level)')
        ax.set_ylabel('Weights changed (exact), per batch'); ax.grid(True, linestyle='--', alpha=0.5)
        ax2 = acc_twin(ax); legend_both(ax, ax2, 'center right')
        ax.set_title('Exact-inequality counts (no threshold)', loc='left', fontsize=12, fontweight='bold')
        bx.plot(bp, d["cnt_batch_exact"], color='tab:brown', alpha=0.15, linewidth=0.5)
        bx.plot(bp, moving_average(d["cnt_batch_exact"], K), color='tab:brown', linewidth=2, label='(D) W_end != W_start, once per batch')
        bx.plot(bp, moving_average(784 * d["active"], K), color='black', linewidth=1.2, linestyle='--', label='784 × distinct neurons that fired in the batch')
        bx.axhline(cfg.get('num_weights', 313600), color='gray', linewidth=0.8, linestyle=':', label='every weight (pure noise level)')
        bx.set_ylabel('Weights changed (exact), per batch'); bx.set_xlabel('Number of training examples'); bx.grid(True, linestyle='--', alpha=0.5)
        bx2 = acc_twin(bx); legend_both(bx, bx2, 'center right')
        footer(fig); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F5_exact_counts.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    return spike


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--snapshot-interval", type=int, default=100, help="batches between held-out snapshots")
    p.add_argument("--thr", type=float, default=1e-6, help="|change| threshold for counting a weight as changed (round 2 used 1e-6)")
    p.add_argument("--thr-strict", type=float, default=1e-5, help="stricter threshold for the after-normalisation counts (float-noise check)")
    p.add_argument("--tag", default="")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end check (~1 min)")
    p.add_argument("--replot", default=None, metavar="RUN_DIR")
    args = p.parse_args()

    if args.replot:
        print(f"figures regenerated in {args.replot} (spike time: {make_figures(args.replot)})"); return

    # Same seeding as hypo_1.py, so this run pairs with hypo_1's baseline of the same seed (same init, same data order).
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        device: torch.device = torch.device("mps"); print("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("Using CUDA.")
    else:
        device = torch.device("cpu"); print("Using CPU.")

    if args.smoke:
        train_subset = Subset(train_dataset, range(0, 32 * 30))
        assign_subset = Subset(train_dataset, range(0, 128)); eval_subset = Subset(test_dataset, range(0, 64)); assign_bs, eval_bs = 128, 64
        args.epochs, args.steps, args.snapshot_interval = 1, 25, 5
    else:
        train_subset = train_dataset
        # Small subsets of MNIST for running average: 2000 images to assign labels and 1000 images to test
        assign_subset = Subset(train_dataset, range(0, 2000)); eval_subset = Subset(test_dataset, range(0, 1000)); assign_bs = eval_bs = 512
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=gen)
    snapshot_assign_loader = DataLoader(assign_subset, batch_size=assign_bs, shuffle=False)
    snapshot_eval_loader = DataLoader(eval_subset, batch_size=eval_bs, shuffle=False)

    run_name = f"baseline_seed{args.seed}" + (f"_{args.tag}" if args.tag else "") + ("_SMOKE" if args.smoke else "") + "_" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"seed": args.seed, "num_epochs": args.epochs, "num_steps": args.steps, "snapshot_interval": args.snapshot_interval,
           "thr": args.thr, "thr_strict": args.thr_strict, "batch_size": 32, "smoke": bool(args.smoke), "tag": args.tag, "run_name": run_name}
    print("config:", json.dumps(cfg))

    model: DiehlAndCookNetwork = DiehlAndCookNetwork().to(device)
    model.train() # This probably doesn't make a difference.
    train(model, train_loader, snapshot_assign_loader, snapshot_eval_loader, device, cfg, run_dir)


if __name__ == "__main__":
    main()
