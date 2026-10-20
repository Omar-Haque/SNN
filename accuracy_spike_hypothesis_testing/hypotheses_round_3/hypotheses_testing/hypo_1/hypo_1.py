"""
hypo_1.py -- Round 3, Hypothesis 1: "reservoir exhaustion".

Base code: src/snn_normalized replicate_results.py (model classes and estimators copied verbatim; parameters are the
tried-and-tested spiking set, cross-checked against hypotheses_round_2/master_test.py: 400 neurons, batch 32,
theta_plus 0.05 (sum/16), theta_decay 1e7, lr 1e-2 / 1e-4, 200 mV lateral inhibition, hard WTA OFF, no theta clamp).

HYPOTHESIS
  Training contains a reservoir of untrained neurons (template still ~ its random init). A reservoir neuron is
  recruited when it wins a presentation outright: it imprints that image in one go and, having the lowest theta
  in the network plus a sharp single-image template, it wins everything resembling that image for ~10k examples
  -- it hijacks a class -- until its theta rises to the trained population's level and it settles. While the
  reservoir lasts there is always a current hijacker, so accuracy sits at ~55-65% and the biased class rotates.
  When the last untrained neuron is used up, the last hijacker settles within one lifetime, competition becomes
  fair, and accuracy jumps within ~5-10k examples and stays flat. Nobody needs to reach theta = 0: a hijacker only
  needs to rise to the trained population's level.

VARIATIONS (one per invocation)
  baseline      no intervention
  refill        at --at (default 110k): re-initialise --n-refill neurons, drawn at random from the LOWEST-theta
                half of the trained population (never the dominants), to uniform random weights, theta = -52 mV.
                Prediction: spike delayed by about n / (drain rate ~2.5 per 1k examples); plateau continues.
  pseudo_refill   same, but theta set to the trained median (untrained, but cannot win).  Prediction: no delay.
  sabotage        at --at (default 100k): raise every untrained neuron's theta to the trained median; weights
                untouched.  Prediction: spike within ~10k (slightly lower ceiling).  Alternative reading
                ("the jump IS the recruits coming online") predicts: no spike at all.
  imprint       at --at (default 100k): give every untrained neuron a random training image as its template
                (normalised to 78.4) and theta = trained median.  Prediction: spike within ~10k.

MEASUREMENTS at every snapshot (every --snapshot-interval batches), saved to results/<run>/run.npz + run.json
and plotted as F1..F7 (see README.md):
  held-out accuracy + confusion matrix + neuron assignments, BindsNET-style running accuracy, reservoir count,
  hijacker index, per-neuron window spikes, per-neuron theta, per-neuron template sharpness, recruitment times.

USAGE (from anywhere; MNIST is read from <repo>/data)
  python hypo_1.py --variation baseline --seed 0
  python hypo_1.py --variation refill --n-refill 50 --seed 0
  python hypo_1.py --variation pseudo_refill --n-refill 50 --seed 0
  python hypo_1.py --variation sabotage --seed 0
  python hypo_1.py --variation imprint --seed 0
  python hypo_1.py --smoke --variation refill          # ~1-2 min end-to-end check of the code path
  python hypo_1.py --replot results/<run_dir>    # regenerate F1..F7 for a finished run
  python summarize_hypo_1.py                     # overlay all finished runs -> summary.csv, S1..S3
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # hypo_1_plots.py sits next to this file; importable from any CWD
from hypo_1_plots import make_run_figures, spike_time_from_curve

script_dir: Path = Path(__file__).resolve().parent
repo_root: Path = script_dir.parents[3]            # .../cooper  (hypo_1 -> hypotheses_testing -> hypotheses_round_3 -> accuracy_spike_hypothesis_testing -> cooper)
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
# Model: copied verbatim from src/snn_normalized replicate_results.py (hard WTA commented out, 200 mV inhibition)
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
            self.synapses.weight.add_(delta_w_plus - delta_w_minus) # delta_w = η(x_pre − x_tar)(w_max − w)^μ
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
#
# Everything above this banner is the network and the two estimators, copied unchanged from the base file. Everything
# below implements the hypothesis test:
#
#   1. template_sharpness()   how we decide, from the weights alone, whether a neuron is "untrained" (still ~ its random
#                             init) or "trained" (has a digit-like template). This defines the reservoir.
#   2. apply_intervention()   the five variations (baseline / refill / pseudo_refill / sabotage / imprint): what is changed in the
#                             network, on which neurons, and at what point in training.
#   3. train()                the training loop of the base file plus, at every snapshot, the measurements the hypothesis
#                             makes predictions about (reservoir count, hijacker index, per-neuron window spikes /
#                             theta / sharpness / recruitment time), all saved to run.npz + run.json.
#   4. main()                 command-line variations, seeding, data loaders, run folder naming, smoke mode.

# A neuron's template is its row of the 784-wide weight matrix (one weight per input pixel), always normalised so that
# the row sums to 78.4. "Sharpness" is the fraction of that 78.4 that sits on the neuron's TOP_K largest weights:
#   * a freshly initialised row, uniform(0.01, 0.3) then normalised, puts ~23% of its mass on its top 100 pixels
#     (checked by simulation: mean 0.232, max 0.245 over 400 rows)
#   * a row that has imprinted a digit puts 50-80% of its mass on the ~100-150 stroke pixels of that digit
# so a single threshold between the two (0.35) separates "untrained" from "trained" with a wide margin on both sides.
VARIATION_NAME = {"baseline": "baseline", "refill": "refill", "refill_sham": "pseudo refill", "disarm": "sabotage", "imprint": "imprint"}  # display names; internal identifiers: pseudo refill = refill_sham, sabotage = disarm

TOP_K: int = 100                     # template "sharpness" = share of a neuron's 78.4 weight mass on its top-100 pixels
UNTRAINED_THRESHOLD: float = 0.35    # a fresh uniform(0.01, 0.3) row scores ~0.23 (max ~0.245); digit templates score 0.5-0.8


@torch.no_grad()
def template_sharpness(W: torch.Tensor) -> torch.Tensor:
    """Per-neuron share of weight mass on the TOP_K largest weights. Computed on the CPU (tiny) so that no MPS
    corner case can bite; returns a CPU tensor of shape [num_neurons]."""
    # W has shape (num_neurons, 784): row i is neuron i's template.
    # topk(...).values has shape (num_neurons, TOP_K): the TOP_K largest weights of each row.
    # Dividing their sum by the row sum (78.4 after normalisation) gives a number in (0, 1] per neuron.
    Wc = W.detach().float().cpu()
    return torch.topk(Wc, TOP_K, dim=1).values.sum(dim=1) / Wc.sum(dim=1)


@torch.no_grad()
def apply_intervention(arm: str, model: "DiehlAndCookNetwork", device: torch.device, rng: np.random.Generator,
                       imprint_dataset, n_refill: int, step: int) -> dict:
    """Modify the network in place according to `arm`. Returns a JSON-serialisable record of what was done."""
    # ---- Read the network's current state -------------------------------------------------------------------
    # W:     (num_neurons, 784) synaptic weights, modified IN PLACE below (it is an nn.Parameter with requires_grad=False)
    # theta: (num_neurons,)     adaptive thresholds in mV; -52 for a neuron that has never fired, higher for active ones
    W = model.synapses.weight
    model.neurons.theta = model.neurons.theta.to(device)  # type: ignore
    theta: torch.Tensor = model.neurons.theta              # type: ignore
    # Split the population into "trained" (digit-like template) and "reservoir" (template still ~ random init).
    # Masks are kept on the CPU; only the final index tensors are moved to the device.
    sharp = template_sharpness(W)                          # CPU
    trained = sharp >= UNTRAINED_THRESHOLD                 # CPU bool mask
    reservoir = ~trained
    theta_cpu = theta.detach().float().cpu()
    n_tr, n_res = int(trained.sum()), int(reservoir.sum())
    # The "trained median theta" is the reference level for the pseudo refill, sabotage and imprint variations: a neuron whose theta
    # sits at the median of the trained population is an ordinary competitor, not a hijacker (a hijacker's defining
    # property is the LOWEST theta in the network, i.e. theta_base).
    if n_tr > 0:
        theta_med = float(theta_cpu[trained].median())
    else:
        print("WARNING: no trained neurons at intervention time; using the median theta of all neurons")
        theta_med = float(theta_cpu.median())
    # Record of the intervention, written into run.json (theta of every neuron before and after, the chosen neurons).
    info = {"arm": arm, "step": int(step), "n_trained_before": n_tr, "n_reservoir_before": n_res,
            "theta_median_trained": theta_med, "theta_before": theta_cpu.numpy().tolist()}

    if arm in ("refill", "refill_sham"):
        # ---- REFILL: turn n_refill TRAINED neurons back into reservoir neurons ---------------------------------
        # Candidates are the lowest-theta half of the trained population. Rationale: the competing "theta-clock"
        # hypothesis says the spike happens when the DOMINANT (highest-theta) neurons' thresholds reach a critical
        # level; resetting dominants would delay the spike under that hypothesis too. Drawing only from the low-theta
        # half leaves every dominant untouched, so the theta-clock predicts NO delay here and the reservoir
        # hypothesis predicts a delay proportional to n_refill.
        cand = trained & (theta_cpu <= theta_med)          # lowest-theta half of the trained population: never the dominants
        cand_idx = torch.nonzero(cand).view(-1).numpy()
        # Fallbacks only matter in --smoke mode, where almost nothing is trained yet.
        if len(cand_idx) < n_refill:
            print(f"WARNING: only {len(cand_idx)} candidates in the lowest-theta half of the trained population; widening to all trained neurons")
            cand_idx = torch.nonzero(trained).view(-1).numpy()
        if len(cand_idx) < n_refill:
            print(f"WARNING: only {len(cand_idx)} trained neurons; widening to all neurons")
            cand_idx = np.arange(model.num_neurons)
        # Draw n_refill of them at random (seeded rng, so the choice is reproducible).
        idx = np.sort(rng.choice(cand_idx, size=min(n_refill, len(cand_idx)), replace=False))
        idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
        # New templates: exactly the initialisation used in DiehlAndCookNetwork.__init__ (uniform(0.01, 0.3), row
        # normalised to 78.4). Shape (n_refill, 784).
        new_rows = torch.empty(len(idx), W.shape[1], device=device).uniform_(0.01, 0.3)   # same init as DiehlAndCookNetwork.__init__
        new_rows.mul_(78.4 / new_rows.sum(dim=1, keepdim=True))
        W[idx_t] = new_rows
        # The ONLY difference between the two refill variations is the threshold the reset neurons get:
        #   refill      -> theta_base (-52 mV): the lowest theta in the network, i.e. a genuine reservoir neuron
        #   pseudo_refill -> the trained median: untrained template, but it cannot win a presentation, so it can never
        #                  be recruited and never hijack. Prediction: no delay. This isolates "low theta" as the
        #                  active ingredient and rules out "resetting 50 templates disturbs the network".
        theta[idx_t] = model.neurons.theta_base if arm == "refill" else theta_med
    elif arm in ("disarm", "imprint"):
        # ---- DISARM / IMPRINT: act on EVERY reservoir neuron ---------------------------------------------------
        idx = torch.nonzero(reservoir).view(-1).numpy()
        if len(idx) == 0:
            print("WARNING: reservoir is empty at intervention time; nothing to do")
        else:
            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            if arm == "imprint":
                # IMPRINT: give each reservoir neuron a random training image as its template (the image's 784 pixel
                # intensities, scaled so the row sums to 78.4 like every other template). The reservoir is emptied by
                # training it instantly. Together with theta = trained median (below) these neurons become ordinary
                # trained competitors, deliberately NOT hijackers.
                pick = rng.choice(len(imprint_dataset), size=len(idx), replace=False)
                # imprint_dataset[p] -> (image tensor of shape (1, 28, 28), label); stack -> (n, 1, 28, 28) -> view (n, 784)
                imgs = torch.stack([imprint_dataset[int(p)][0] for p in pick]).view(len(idx), -1).to(device)
                imgs = imgs * (78.4 / imgs.sum(dim=1, keepdim=True).clamp_min(1e-6))
                W[idx_t] = imgs
                info["imprint_labels"] = [int(imprint_dataset[int(p)][1]) for p in pick]
            # DISARM: weights untouched; only theta is raised to the trained median, so these neurons lose the one
            # property (lowest theta) that lets a reservoir neuron win a presentation and be recruited. The reservoir
            # still exists as untrained templates, but it can no longer supply hijackers. (Imprint also lands here:
            # its neurons get the same theta on top of their new templates.)
            theta[idx_t] = theta_med
    else:
        raise ValueError(f"unknown arm {arm!r}")

    info["idx"] = [int(i) for i in idx]
    info["theta_after"] = theta.detach().float().cpu().numpy().tolist()
    return info


def train(
    model: "DiehlAndCookNetwork",
    train_loader: DataLoader,
    snapshot_assign_loader: DataLoader,
    snapshot_eval_loader: DataLoader,
    imprint_dataset,
    device: torch.device,
    cfg: dict,
    run_dir: Path,
) -> "DiehlAndCookNetwork":
    # cfg holds the command-line configuration: variation, n_refill, intervention_at, seed, num_epochs, num_steps,
    # snapshot_interval (batches between snapshots; 100 batches x 32 = every 3,200 examples), smoke flag.
    rng = np.random.default_rng(cfg["seed"])
    n: int = model.num_neurons
    num_epochs, num_steps, snapshot_interval = cfg["num_epochs"], cfg["num_steps"], cfg["snapshot_interval"]
    arm: str = cfg["arm"]
    intervention_pending: bool = arm != "baseline"
    intervention_info: dict | None = None

    total_points_processed: int = 0   # number of training examples presented so far (the x-axis of every plot)
    global_batch_idx: int = 0         # counts batches across epochs (snapshots and the BindsNET window use this)

    # Held-out snapshot histories (one entry per snapshot; all saved to run.npz)
    points_history, accuracy_history = [], []                          # x-axis, held-out accuracy (%)
    conf_history, assignments_history = [], []                         # (10, 10) confusion matrix; (400,) digit per neuron, -1 = unassigned
    reservoir_history, hijack_history = [], []                         # number of untrained neurons; hijacker index
    window_spikes_history, theta_history, sharp_history, recruit_history = [], [], [], []   # per-neuron (400,) vectors

    # Per-neuron bookkeeping
    # recruit_step[i] = total_points_processed at the FIRST snapshot where neuron i's template was digit-like.
    # NaN until then. "Age" of a neuron at a later snapshot = total_points_processed - recruit_step[i]. Ages are
    # therefore quantised to the snapshot interval: a neuron recruited anywhere inside the current window has age 0.
    recruit_step = np.full(n, np.nan)                 # first snapshot at which the template was digit-like (NaN = not yet)
    imprinted = np.zeros(n, dtype=bool)               # set by the imprint variation (so the figures can mark those neurons)
    refilled = np.zeros(n, dtype=bool)                # set by the refill variations
    # W_prev: the weight matrix as it was at the previous snapshot. The "hijacker index" of a snapshot is computed
    # from W - W_prev: per-neuron L2 norm of the change over the window, and then the share of the total carried by
    # the 20 neurons that changed most. ~0.5 when a handful of fresh recruits do most of the learning, ~0.1 when
    # learning is spread over the population.
    W_prev = model.synapses.weight.detach().clone()   # weights at the previous snapshot (for the hijacker index)
    # spikes_since_snapshot[i] = output spikes of neuron i, summed over all images of all batches since the previous
    # snapshot (the "window spikes"). Kept on the device, read out and zeroed at each snapshot.
    spikes_since_snapshot = torch.zeros(n, device=device)

    # Cooper / BindsNET-style running estimate (identical to the base file; see bindsnet_* helpers)
    # update_interval = 250 examples rounded down to whole batches -> 7 batches = 224 examples at batch size 32.
    cooper_batch_size: int = train_loader.batch_size
    cooper_update_steps: int = max(1, int(250 / cooper_batch_size))
    cooper_update_interval: int = cooper_update_steps * cooper_batch_size
    cooper_spike_record = torch.zeros(cooper_update_interval, n, device=device)       # spike counts of the last 224 training images
    cooper_label_record = torch.zeros(cooper_update_interval, dtype=torch.long, device=device)   # their labels
    cooper_assignments = -torch.ones(n, dtype=torch.long, device=device)              # -1 = no assignment yet (first window scores ~10%)
    cooper_proportions = torch.zeros(n, 10, device=device)
    cooper_rates = torch.zeros(n, 10, device=device)                                  # accumulated over the whole run (alpha = 1)
    cooper_points_history, cooper_all_history, cooper_proportion_history = [], [], []

    t_start = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            batch_size: int = data.shape[0]
            data, targets = data.to(device), targets.to(device)

            # ---- intervention: applied once, at the first batch at or after cfg["intervention_at"] ----
            # total_points_processed is always a multiple of 32, so "at 100000" fires at exactly 100,000 and
            # "at 110000" at 110,016. Applied BEFORE this batch is presented.
            if intervention_pending and total_points_processed >= cfg["intervention_at"]:
                intervention_info = apply_intervention(arm, model, device, rng, imprint_dataset, cfg["n_refill"], total_points_processed)
                idx = np.array(intervention_info["idx"], dtype=int)
                if arm in ("refill", "refill_sham"):
                    # Reset neurons are untrained again: their recruitment time is cleared and will be set afresh
                    # when (if) they get recruited. The refill variation's neurons can be; the pseudo refill's never are.
                    recruit_step[idx] = np.nan; refilled[idx] = True
                elif arm == "imprint":
                    # Imprinted neurons are trained by construction, as of now.
                    recruit_step[idx] = total_points_processed; imprinted[idx] = True
                # Re-base W_prev so that the next hijacker index measures what the network LEARNED since the
                # intervention, not the (large, artificial) change the intervention itself made to the weights.
                W_prev = model.synapses.weight.detach().clone()   # the hijacker index must measure learning, not the intervention itself
                intervention_pending = False
                print(f"\n*** INTERVENTION '{VARIATION_NAME[arm]}' applied at {total_points_processed} examples on {len(idx)} neurons "
                      f"(trained before: {intervention_info['n_trained_before']}, reservoir before: {intervention_info['n_reservoir_before']}, "
                      f"trained median theta: {intervention_info['theta_median_trained']:.2f} mV) ***\n")

            # Generate poisson spike trains.
            spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps)  # type: ignore
            # Flatten spatial dimensions: shape becomes [num_steps, batch_size, 784]
            spike_data = spike_data.view(num_steps, batch_size, 784)

            # Reset the network memory for a new batch
            model.reset(batch_size, device)

            # Spike counts of this batch over the whole presentation (plasticity on)
            # batch_train_spikes has shape (batch_size, num_neurons): entry [b, i] = number of times neuron i fired
            # during the 350 ms presentation of image b. This is the same quantity evaluate_network() builds as
            # batch_neuron_spikes, but recorded from the TRAINING presentation (learning on) instead of a separate pass.
            batch_train_spikes = torch.zeros(batch_size, n, device=device)
            for t in range(num_steps):
                out_spikes: torch.Tensor = model(spike_data[t])   # (batch_size, num_neurons), 0/1
                batch_train_spikes += out_spikes
            spikes_since_snapshot += batch_train_spikes.sum(dim=0)   # (num_neurons,): add this batch's spikes, all images
            total_points_processed += batch_size

            # ---- BindsNET-style estimate: fill the window; score it with the previous windows' assignments; update ----
            # Costs nothing: it re-uses batch_train_spikes. See the bindsnet_* helpers for the differences from the
            # held-out estimate. Skipped for a partial last batch (never happens with 60000 / 32).
            if batch_size == cooper_batch_size:
                slot = (global_batch_idx % cooper_update_steps) * cooper_batch_size   # where this batch goes in the 224-row window
                cooper_spike_record[slot:slot + batch_size] = batch_train_spikes
                cooper_label_record[slot:slot + batch_size] = targets
                if (global_batch_idx + 1) % cooper_update_steps == 0:               # window full: score it, THEN update assignments
                    all_pred = bindsnet_all_activity(cooper_spike_record, cooper_assignments, 10)
                    prop_pred = bindsnet_proportion_weighting(cooper_spike_record, cooper_assignments, cooper_proportions, 10)
                    cooper_points_history.append(total_points_processed)
                    cooper_all_history.append(100.0 * (all_pred == cooper_label_record).float().mean().item())
                    cooper_proportion_history.append(100.0 * (prop_pred == cooper_label_record).float().mean().item())
                    cooper_assignments, cooper_proportions, cooper_rates = bindsnet_assign_labels(cooper_spike_record, cooper_label_record, 10, cooper_rates)

            # ---- snapshot: held-out accuracy + the three hypothesis measurements ----
            # Every snapshot_interval batches (default 100 -> every 3,200 examples; the first one after batch 0).
            if global_batch_idx % snapshot_interval == 0:
                # (a) Held-out accuracy exactly as in the base file: assign a digit to each neuron from 2,000 training
                #     images (learning off), then score 1,000 test images. Also keeps the 10x10 confusion matrix
                #     (rows = true digit, columns = predicted digit) and the per-neuron assignments.
                temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
                snapshot_acc, c_matrix = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps, return_matrix=True)  # type: ignore
                with torch.no_grad():
                    W = model.synapses.weight.detach()
                    # (b) Template sharpness of every neuron -> reservoir count and recruitment times.
                    sharp = template_sharpness(W).numpy()                                   # (num_neurons,)
                    theta_np = model.neurons.theta.detach().float().cpu().numpy().copy()  # type: ignore
                    # (c) Hijacker index: per-neuron L2 norm of the weight change over the window, then the share of
                    #     the total carried by the 20 neurons that changed most. (All on the CPU, it is tiny.)
                    dw = torch.linalg.norm((W - W_prev).float().cpu(), dim=1)             # (num_neurons,)
                    dw_total = float(dw.sum())
                    hijack = float(torch.topk(dw, min(20, n)).values.sum()) / dw_total if dw_total > 0 else float("nan")
                    W_prev = W.clone()
                    # (d) Window spikes: who fired how much since the previous snapshot. Read out, then zeroed.
                    ws = spikes_since_snapshot.cpu().numpy().copy()
                    spikes_since_snapshot.zero_()
                # A neuron is "recruited" at the first snapshot where its template is digit-like.
                newly = (sharp >= UNTRAINED_THRESHOLD) & np.isnan(recruit_step)
                recruit_step[newly] = total_points_processed
                reservoir = int((sharp < UNTRAINED_THRESHOLD).sum())
                # Append everything for this snapshot. Per-neuron vectors are stored as float32/int16 to keep run.npz small.
                points_history.append(total_points_processed); accuracy_history.append(float(snapshot_acc))
                conf_history.append(np.asarray(c_matrix).astype(np.int32)); assignments_history.append(temp_labels.cpu().numpy().astype(np.int16))
                reservoir_history.append(reservoir); hijack_history.append(hijack)
                window_spikes_history.append(ws.astype(np.float32)); theta_history.append(theta_np.astype(np.float32))
                sharp_history.append(sharp.astype(np.float32)); recruit_history.append(recruit_step.copy())
                print(f"[{VARIATION_NAME[arm]}] {total_points_processed:>7d} ex | held-out acc {snapshot_acc:5.1f}% | reservoir {reservoir:3d} | hijacker idx {hijack:4.2f} | {(time.time() - t_start) / 60:5.1f} min")
            global_batch_idx += 1
    print("Training complete")

    # ---- Save everything and draw the figures ----
    # Spike time: first snapshot after which held-out accuracy never drops below 78% again (NaN if there is none).
    spike_time = spike_time_from_curve(points_history, accuracy_history)
    out = dict(cfg)
    out.update({
        "intervention_step_actual": None if intervention_info is None else intervention_info["step"],
        "intervention_info": intervention_info,
        "spike_time": spike_time,
        "untrained_threshold": UNTRAINED_THRESHOLD, "top_k": TOP_K,
        "theta_base": float(model.neurons.theta_base), "theta_plus": float(model.neurons.theta_plus),
        "theta_decay_tc": float(-1.0 / math.log(model.neurons.theta_decay)),
        "inhibition_mV": float(model.inh_matrix[0, 1].item()),
        "lr_plus": float(model.lr_plus), "lr_minus": float(model.lr_minus), "num_neurons": n, "device": str(device),
        "minutes": (time.time() - t_start) / 60,
    })
    # run.npz: S = number of snapshots, N = number of neurons
    #   points (S,), accuracy (S,)                                   held-out curve
    #   cooper_points, cooper_all, cooper_proportion                 BindsNET-style curve (one point per 224 examples)
    #   reservoir_count (S,), hijack_index (S,)
    #   window_spikes (S, N), theta_snap (S, N), sharp_snap (S, N)   per-neuron state at each snapshot
    #   assignments_snap (S, N) int16 (-1 = unassigned), conf_snap (S, 10, 10)
    #   recruit_step_snap (S, N)                                     recruitment time of every neuron as known at each snapshot (NaN = untrained)
    #   imprinted (N,), refilled (N,)                                which neurons the intervention touched
    np.savez(run_dir / "run.npz",
             points=np.array(points_history), accuracy=np.array(accuracy_history),
             cooper_points=np.array(cooper_points_history), cooper_all=np.array(cooper_all_history), cooper_proportion=np.array(cooper_proportion_history),
             reservoir_count=np.array(reservoir_history), hijack_index=np.array(hijack_history),
             window_spikes=np.array(window_spikes_history), theta_snap=np.array(theta_history), sharp_snap=np.array(sharp_history),
             assignments_snap=np.array(assignments_history), conf_snap=np.array(conf_history), recruit_step_snap=np.array(recruit_history),
             imprinted=imprinted, refilled=refilled)
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    make_run_figures(run_dir)   # F1..F7, from run.npz + run.json only (so --replot can redo them without torch)
    print(f"\nrun finished in {out['minutes']:.1f} min | spike time: {spike_time if spike_time == spike_time else 'none'} | "
          f"outputs: {run_dir}")
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variation", "--arm", dest="arm", choices=["baseline", "refill", "pseudo_refill", "refill_sham", "sabotage", "disarm", "imprint"], default="baseline",
                   help="which variation to run (refill_sham / disarm are accepted as the old names of pseudo_refill / sabotage)")
    p.add_argument("--n-refill", type=int, default=50, help="neurons re-initialised by the refill variations")
    p.add_argument("--at", type=int, default=None, help="example count at which to intervene (default: 110000 for the refill variations, 100000 for sabotage/imprint)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps", type=int, default=350, help="timesteps (ms) per presentation")
    p.add_argument("--snapshot-interval", type=int, default=100, help="batches between held-out snapshots (100 = every 3200 examples)")
    p.add_argument("--tag", default="", help="extra text for the run folder name")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end check of the whole code path (~1-2 min)")
    p.add_argument("--replot", default=None, metavar="RUN_DIR", help="regenerate the figures of a finished run and exit")
    args = p.parse_args()
    if args.arm == "pseudo_refill":
        args.arm = "refill_sham"   # internal identifier (kept so that old run.json files stay readable)
    if args.arm == "sabotage":
        args.arm = "disarm"        # same

    # --replot: figures only, from a finished run's npz/json. Nothing is trained.
    if args.replot:
        st = make_run_figures(args.replot)
        print(f"figures regenerated in {args.replot} (spike time: {st})")
        return

    # Default intervention points: refill variations at 110k (reservoir ~50, still 20k before the baseline spike);
    # drain variations at 100k (reservoir ~80).
    if args.at is None:
        args.at = 110_000 if args.arm in ("refill", "refill_sham") else 100_000

    # Seeds: weight init and spike generation use torch's global RNG; the data order uses its own generator (below);
    # the intervention's random choices use numpy's `rng` inside train(). Same --seed => same init AND same data order,
    # so all seed-0 variations are identical to the seed-0 baseline up to the intervention (paired comparison).
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        device: torch.device = torch.device("mps"); print("Using MPS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("Using CUDA.")
    else:
        device = torch.device("cpu"); print("Using CPU.")

    if args.smoke:
        # Tiny configuration that still exercises every code path: 30 batches, 25 ms presentations, a snapshot every
        # 5 batches, the intervention after 12 batches, at most 5 refilled neurons. ~1-2 minutes. Run folders are
        # tagged _SMOKE and ignored by summarize_hypo_1.py.
        n_batches = 30
        train_subset = Subset(train_dataset, range(0, 32 * n_batches))
        assign_subset = Subset(train_dataset, range(0, 128)); eval_subset = Subset(test_dataset, range(0, 64))
        assign_bs, eval_bs = 128, 64
        args.epochs, args.steps, args.snapshot_interval = 1, 25, 5
        args.at = 32 * 12; args.n_refill = min(args.n_refill, 5)
    else:
        train_subset = train_dataset
        # Small subsets of MNIST for running average: 2000 images to assign labels and 1000 images to test
        assign_subset = Subset(train_dataset, range(0, 2000)); eval_subset = Subset(test_dataset, range(0, 1000))
        assign_bs = eval_bs = 512
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, generator=gen)
    snapshot_assign_loader = DataLoader(assign_subset, batch_size=assign_bs, shuffle=False)
    snapshot_eval_loader = DataLoader(eval_subset, batch_size=eval_bs, shuffle=False)

    # Run folder: results/<variation>[_n<refill>][_at<k>][_seed][_tag][_SMOKE]_<timestamp>/ -- never overwritten.
    run_name = VARIATION_NAME[args.arm].replace(" ", "_")
    if args.arm in ("refill", "refill_sham"): run_name += f"_n{args.n_refill}"
    if args.arm != "baseline": run_name += f"_at{args.at // 1000}k"
    run_name += f"_seed{args.seed}"
    if args.tag: run_name += f"_{args.tag}"
    if args.smoke: run_name += "_SMOKE"
    run_name += "_" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"arm": args.arm, "n_refill": args.n_refill, "intervention_at": args.at, "seed": args.seed,
           "num_epochs": args.epochs, "num_steps": args.steps, "snapshot_interval": args.snapshot_interval,
           "batch_size": 32, "smoke": bool(args.smoke), "tag": args.tag, "run_name": run_name}
    print("config:", json.dumps(cfg))

    model: DiehlAndCookNetwork = DiehlAndCookNetwork().to(device)
    model.train() # This probably doesn't make a difference.
    # train_dataset is passed as the pool of images the imprint variation draws its templates from.
    train(model, train_loader, snapshot_assign_loader, snapshot_eval_loader, train_dataset, device, cfg, run_dir)


if __name__ == "__main__":
    main()
