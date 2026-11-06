"""
snn_exp.py  --  your `snn_normalized replicate_results.py` made parameterisable for controlled experiments.

Everything marked  # [EXP]  is a change relative to the original script. The network maths (LIF neuron,
lateral inhibition, STDP, BindsNET-style normalisation) is unchanged unless a flag is given, and with the
default flags it reproduces the original configuration (400 neurons, 350 steps, data*0.1, batch 32,
theta_plus 0.05 / 16, inh 200).

[WTA] marks the addition for the inhibition / hard-WTA / presentation-time experiments:  --wta

Typical use (normally launched by run_experiments.py or run_wta_experiments.py):
    python snn_exp.py --name A_base --snapshot_interval 20 --save_spikes
    python snn_exp.py --name B_n100 --n_neurons 100
    python snn_exp.py --name W1_t250 --inh 17.5 --wta --num_steps 250
"""
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import math
import os
import json
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


script_dir: Path = Path(__file__).resolve().parent
output_dir: Path = script_dir / "results"   # [EXP] overwritten per run in __main__ (results/<name>/)

# [EXP] ------------------------------------------------------------------------------------------------
# Global encoding settings. encode_images() is the ONLY place where images become spikes, and it is used by
# training, label assignment and evaluation alike, so they can never drift apart.
#   rate      : spikegen.rate(pixel * intensity)                (your original: intensity = 0.1)
#   rate_norm : every image rescaled to the same total intensity `norm_total` first (your commented-out
#               "sum = 75" code), then spikegen.rate(pixel * intensity)
ENC = {"encoding": "rate", "intensity": 0.1, "norm_total": 75.0}

def encode_images(data: torch.Tensor, num_steps: int) -> torch.Tensor:
    x = data
    if ENC["encoding"] == "rate_norm":
        x = x / (x.sum(dim=(1, 2, 3), keepdim=True) + 1e-5) * ENC["norm_total"]
    elif ENC["encoding"] != "rate":
        raise ValueError(f"unknown encoding {ENC['encoding']}")
    x = (x * ENC["intensity"]).clamp(0.0, 1.0)
    return spikegen.rate(x, num_steps=num_steps)  # type: ignore

# [EXP] Held-out evaluation consumes random numbers (Poisson spikes). To make the evaluation frequency a
# truly independent factor, the RNG state is saved before every evaluation, the evaluation runs with a fixed
# seed (same input noise at every checkpoint -> smoother curve), and the training RNG state is restored.
EVAL_SEED = 12345

def save_rng(device: torch.device) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["dev"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        state["dev"] = torch.mps.get_rng_state()
    return state

def restore_rng(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if "dev" in state:
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["dev"], device)
        else:
            torch.mps.set_rng_state(state["dev"])

def build_loaders(batch_size: int, assign_n: int, eval_n: int):
    """[EXP] was module-level code; moved into a function so that importing this file has no side effects."""
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    # Small subsets of MNIST for running accuracy: assign labels on `assign_n` train images, score `eval_n` test images
    assign_subset = torch.utils.data.Subset(train_dataset, range(0, assign_n))
    eval_subset = torch.utils.data.Subset(test_dataset, range(0, eval_n))
    snapshot_assign_loader = DataLoader(assign_subset, batch_size=512, shuffle=False)
    snapshot_eval_loader = DataLoader(eval_subset, batch_size=512, shuffle=False)
    return train_loader, test_loader, snapshot_assign_loader, snapshot_eval_loader


# EXCITATORY NEURONS
# Excitatory neurons are an entire class, but inhibitory neurons can be simulated using matrices.
class excitatory_neurons(nn.Module):
    def __init__(self, num_neurons: int = 400, theta_plus: float = 0.05, theta_div: float = 16.0, refrac: int = 0,
                 wta_mode: str = "off"):  # [EXP] args  [WTA] wta_mode
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = num_neurons  # [EXP] number of excitatory neurons (was hard-coded 400)

        # For the adaptive threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = theta_plus           # [EXP] a penalty added to the fire threshold each time a neuron fires.
        self.theta_div:   float = theta_div            # [EXP] was the hard-coded "/ 16" in the theta update
        self.refrac:      int = refrac                 # [EXP] refractory period in steps (0 = off, as in your original)
        # [WTA] hard winner-take-all, at most ONE neuron spikes per sample per step. Modes:
        #   off      no WTA
        #   v        winner = highest v among supra-threshold neurons (your original commented-out code; it
        #            favours high-theta neurons and works against the adaptive threshold)
        #   vt       winner = highest (v - theta), i.e. the neuron furthest above ITS OWN threshold; losers keep v
        #            (they stay above threshold and fire one by one on the following steps)
        #   vt_reset as vt, but the losers are reset to E_rest (a real hard WTA: they are cleared, not queued)
        if wta_mode not in ("off", "v", "vt", "vt_reset"):
            raise ValueError(f"unknown wta_mode {wta_mode}")
        self.wta_mode:    str = wta_mode
        self.theta_decay: float = math.exp(-1.0 / 1e7)

        # State variables for leaky integrate-and-fire model. Used as memory for neuron states.
        self.v:     torch.Tensor | None = None     # self.v[i] is the ith neuron's voltage.
        self.g_e:   torch.Tensor | None = None     # self.g_e[i] is the ith neuron's excitatory conductance.
        self.g_i:   torch.Tensor | None = None     # self.g_i[i] is the ith neuron's inhibitory conductance.
        self.theta: torch.Tensor | None = (torch.ones(self.num_neurons) * self.theta_base) # the current, adaptive threshold for each neuron
        self.refractory_timer: torch.Tensor | None = None  # [EXP]

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
        Must be called before starting the time loop for a new batch of images.
        """
        self.v     = torch.ones(batch_size, self.num_neurons, device=device) * self.E_rest # shape is (batch_size, num_neurons)
        self.g_e   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.g_i   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.theta = self.theta.to(device) # type: ignore
        self.refractory_timer = torch.zeros(batch_size, self.num_neurons, device=device)  # [EXP]

    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor, learning: bool = True) -> torch.Tensor:
        if self.v is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")

        self.v = self.decay * (self.v - self.E_rest) + self.E_rest

        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh

        # [EXP] optional refractory period (your commented-out code, now switchable with --refrac N)
        if self.refrac > 0:
            self.refractory_timer = torch.clamp(self.refractory_timer - 1.0, min=0.0)  # type: ignore
            self.v = torch.where(self.refractory_timer > 0.0, torch.full_like(self.v, self.E_rest), self.v)

        # save a copy of the voltage to plot a voltage and theta vs timesteps plot for neurons
        self.raw_v: torch.Tensor = self.v.clone()

        spikes: torch.Tensor = (self.v >= self.theta).float()

        # [WTA] hard winner-take-all (your commented-out code, now switchable with --wta).
        # Among the neurons above threshold in a sample, only the one with the highest voltage spikes
        # (ties -> lowest index). Samples where nothing crossed threshold are untouched (spikes stay 0).
        if self.wta_mode != "off":
            score = self.v if self.wta_mode == "v" else (self.v - self.theta)   # [WTA] vt/vt_reset: relative to own threshold
            competing = torch.where(spikes > 0, score, torch.full_like(score, -float("inf")))
            winner = torch.argmax(competing, dim=1)
            wta_mask = torch.zeros_like(spikes)
            wta_mask.scatter_(1, winner.unsqueeze(1), 1.0)
            candidates = spikes
            spikes = spikes * wta_mask                       # samples with no candidate stay all-zero
            if self.wta_mode == "vt_reset":
                losers = (candidates - spikes) > 0           # crossed threshold but lost -> cleared
                self.v[losers] = self.E_rest

        self.v[spikes == 1.0] = self.E_rest
        if self.refrac > 0:  # [EXP]
            self.refractory_timer[spikes == 1.0] = float(self.refrac)  # type: ignore

        if learning:
            # [EXP] "/ 16" is now self.theta_div. Note: per spike, theta grows by theta_plus / theta_div.
            self.theta += (spikes.sum(dim=0) * self.theta_plus) / self.theta_div

        return spikes

# Architecture of the entire Diehl and Cook network
# STDP is also performed within this class
class DiehlAndCookNetwork(nn.Module):
    def __init__(self, num_neurons: int = 400, theta_plus: float = 0.05, theta_div: float = 16.0,
                 refrac: int = 0, inh: float = 200.0, lr_scale: float = 1.0, wta_mode: str = "off"):  # [EXP] args  [WTA] wta_mode
        super(DiehlAndCookNetwork, self).__init__()

        self.num_neurons: int = num_neurons  # [EXP]

        self.synapses: nn.Linear = nn.Linear(784, self.num_neurons, bias=False) # connect 784 pixels to each excitatory neuron.
        self.synapses.weight.requires_grad = False # Using STDP instead

        # STDP parameters
        self.w_max:    float = 1.0  # weight upper limit
        self.lr_plus:  float = 1e-2 * lr_scale # Learning rate for potentiation 0.01   [EXP] lr_scale
        self.lr_minus: float = 1e-4 * lr_scale # Learning rate for depression 0.0001   [EXP] lr_scale

        # Trace decay multipliers
        self.decay_x: float = math.exp(-1.0 / 20.0) # Pre-synaptic trace decay
        self.decay_y: float = math.exp(-1.0 / 20.0) # Post-synaptic trace decay

        # Initialize weights uniformly between a and b
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3)
        with torch.no_grad():
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)

        self.neurons: excitatory_neurons = excitatory_neurons(num_neurons, theta_plus, theta_div, refrac, wta_mode)  # [EXP]

        # LATERAL INHIBITION USING A num_neurons x num_neurons MATRIX
        # A penalty of `inh` mV  [EXP] (was hard-coded 200)
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * inh
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

def visualize_learned_templates(model: DiehlAndCookNetwork) -> None:
    print("Extracting weights and thresholds after training...")

    # Shape is (num_neurons, 784)
    weights: np.ndarray = model.synapses.weight.detach().cpu().numpy()

    # Shape becomes (num_neurons, 28, 28)
    weight_images: np.ndarray = weights.reshape(model.num_neurons, 28, 28)

    # Extract the current adaptive thresholds (theta) for all neurons
    thetas: np.ndarray = model.neurons.theta.detach().cpu().numpy() # type: ignore

    # [EXP] grid size follows the number of neurons (was fixed 20x20)
    side = int(math.ceil(math.sqrt(model.num_neurons)))
    fig, axes = plt.subplots(side, side, figsize=(18, 18), squeeze=False)
    fig.suptitle(f"Synaptic Weights Visualization for ({model.num_neurons} Neurons)", fontsize=20, y=0.95)
    plt.subplots_adjust(wspace=0.05, hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i >= model.num_neurons:  # [EXP] unused cells of the grid
            ax.axis('off')
            continue
        ax.imshow(weight_images[i], cmap='hot', interpolation='nearest')
        ax.set_title(f"θ: {thetas[i]:.1f}", fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(output_dir / f"theta_templates.png")
    plt.close(fig)

def visualize_confusion_matrix(conf_matrix: np.ndarray, step: int) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix at {step} Samples", fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks) # type: ignore
    plt.yticks(tick_marks, tick_marks) # type: ignore
    plt.ylabel('True Digit Class')
    plt.xlabel('Predicted Digit Class')

    # Add the text annotations inside the grid
    thresh = conf_matrix.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{step}.png")
    plt.close()

# ---------------------------------------------------------------------------------------------------------------
# Cooper / BindsNET-style accuracy estimate (unchanged).
# Port of bindsnet.evaluation.{assign_labels, all_activity, proportion_weighting} and of the bookkeeping in
# bindsnet/examples/mnist/batch_eth_mnist.py. It uses the spikes recorded DURING TRAINING (plasticity on); each
# window of `update_interval` training samples is scored with the assignments computed from the PREVIOUS windows.
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

        # [EXP] shared encoder (see encode_images)
        spike_data: torch.Tensor = encode_images(data, num_steps)
        spike_data = spike_data.view(num_steps, batch_size, 784)

        model.reset(batch_size, device)

        # Track the total spikes fired by each neuron over the presentation window
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
    neuron_assignments: torch.Tensor = torch.argmax(spike_counts, dim=1)
    # Find neurons that fired 0 times
    total_spikes = spike_counts.sum(dim=1)
    dead_neurons = (total_spikes == 0)
    neuron_assignments[dead_neurons] = -1

    print("Neuron assignments complete")
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
    assignment_matrix: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)

    # Scatter the 1.0s, ignoring the -1 dead neurons
    valid_mask = (neuron_assignments != -1)
    safe_assignments = torch.where(valid_mask, neuron_assignments, torch.zeros_like(neuron_assignments))
    assignment_matrix.scatter_(1, safe_assignments.unsqueeze(1), 1.0)
    assignment_matrix[~valid_mask] = 0.0 # Erase the temporary zeros used for dead neurons

    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)

        spike_data: torch.Tensor = encode_images(data, num_steps)  # [EXP] shared encoder
        spike_data = spike_data.view(num_steps, batch_size, 784)

        model.reset(batch_size, device)

        # Track the total spikes fired by each neuron over the presentation window
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)

        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes

        # Classify all images in the batch
        class_votes: torch.Tensor = torch.matmul(batch_neuron_spikes, assignment_matrix)  # [batch_size, 10]
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

def train(
    model: DiehlAndCookNetwork,
    train_loader: DataLoader,
    snapshot_assign_loader: DataLoader,
    snapshot_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 3,
    num_steps: int = 350,
    track_accuracy_curve: bool = False,      # held-out accuracy (assign on train subset, score on test subset)
    track_dominant_neuron: bool = True,
    generate_confusion_matrix: bool = False,
    visualize_templates: bool = False,
    # [EXP] new arguments ---------------------------------------------------------------------------------
    track_cooper: bool = True,               # Cooper/BindsNET running estimate; free (uses training spikes)
    snapshot_interval: int = 100,            # held-out eval every N batches (was hard-coded 100)
    dense_range: tuple[int, int] | None = None,  # (start_sample, end_sample): use dense_interval inside this range
    dense_interval: int | None = None,
    max_train_samples: int | None = None,    # stop early (smoke tests)
    save_spikes: bool = False,               # save per-sample training spike counts -> offline Cooper analysis
    config: dict | None = None,
) -> DiehlAndCookNetwork:

    # Tracking variables for running accuracy
    total_points_processed = 0
    points_history = []
    accuracy_history = []
    heldout_unassigned = []   # [EXP] number of neurons without assignment at each held-out checkpoint
    # [EXP] snapshot_interval is now an argument (was `snapshot_interval = 100` here)

    # Tracking variables for the Cooper / BindsNET-style running estimate (see bindsnet_* helpers above).
    # BindsNET: update_interval = 250 samples rounded down to whole batches -> 224 samples at batch size 32.
    cooper_batch_size: int = train_loader.batch_size
    cooper_update_steps: int = max(1, int(250 / cooper_batch_size))   # [EXP] max(1, ...) for large batches
    cooper_update_interval: int = cooper_update_steps * cooper_batch_size
    cooper_spike_record = torch.zeros(cooper_update_interval, model.num_neurons, device=device)
    cooper_label_record = torch.zeros(cooper_update_interval, dtype=torch.long, device=device)
    cooper_assignments = -torch.ones(model.num_neurons, dtype=torch.long, device=device)
    cooper_proportions = torch.zeros(model.num_neurons, 10, device=device)
    cooper_rates = torch.zeros(model.num_neurons, 10, device=device)
    cooper_points_history = []
    cooper_all_history = []
    cooper_proportion_history = []
    global_batch_idx = 0  # counts batches across epochs

    # [EXP] cheap diagnostics, logged once per Cooper window (they help explain WHY accuracy jumps) ------------
    diag: dict[str, list] = {k: [] for k in
        ["points", "silent_frac", "mean_spikes", "active_neurons", "theta_mean", "theta_min", "theta_max", "w_std"]}
    spike_store: list = []
    label_store: list = []
    done = False
    t_start = time.time()

    def current_interval() -> int:
        if dense_range is not None and dense_interval and dense_range[0] <= total_points_processed <= dense_range[1]:
            return dense_interval
        return snapshot_interval

    def save_curves() -> None:
        """Write all curves to results/<name>/curves.npz (atomically, so plots can be made while training runs)."""
        payload = dict(
            heldout_points=np.array(points_history), heldout_acc=np.array(accuracy_history),
            heldout_unassigned=np.array(heldout_unassigned),
            cooper_points=np.array(cooper_points_history), cooper_all=np.array(cooper_all_history),
            cooper_proportion=np.array(cooper_proportion_history),
            cooper_window=np.array(cooper_update_interval),
            elapsed_s=np.array(time.time() - t_start),
            config=json.dumps(config or {}, default=str),
        )
        payload.update({f"diag_{k}": np.array(v) for k, v in diag.items()})
        tmp = output_dir / "curves_tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, output_dir / "curves.npz")
    # ---------------------------------------------------------------------------------------------------------

    # Tracking variables for voltage/theta vs timesteps plot
    dominant_idx = None
    track_steps = []
    track_theta = []
    track_vmax = []
    track_spiked = []

    for epoch in range(num_epochs):
        if done:  # [EXP]
            break
        for batch_idx, (data, targets) in enumerate(train_loader):
            if max_train_samples is not None and total_points_processed >= max_train_samples:  # [EXP]
                done = True
                break
            batch_size: int = data.shape[0]
            data, targets = data.to(device), targets.to(device)

            # Generate poisson spike trains.   [EXP] shared encoder
            spike_data: torch.Tensor = encode_images(data, num_steps)
            # Flatten spatial dimensions: shape becomes [num_steps, batch_size, 784]
            spike_data = spike_data.view(num_steps, batch_size, 784)

            # Reset the network memory for a new batch
            model.reset(batch_size, device)

            # Spike counts of this batch over the whole presentation (plasticity on) for the Cooper / BindsNET-style estimate
            batch_train_spikes = torch.zeros(batch_size, model.num_neurons, device=device)

            # Identify the greediest neuron after 5,000 samples
            if track_dominant_neuron and dominant_idx is None and total_points_processed > 5000:
                dominant_idx = torch.argmin(model.neurons.theta).item() # type: ignore
                print(f"\n Tracking passive neuron #{dominant_idx}")

            # Track the highest voltage it reaches in this batch.
            highest_v_in_batch = torch.full((), -1000.0, device=device)
            dominant_spike_count = torch.zeros((), device=device)

            for t in range(num_steps):
                out_spikes: torch.Tensor = model(spike_data[t])
                batch_train_spikes += out_spikes
                if track_dominant_neuron and dominant_idx is not None:
                    highest_v_in_batch = torch.maximum(highest_v_in_batch, model.neurons.raw_v[:, dominant_idx].max())
                    dominant_spike_count += out_spikes[:, dominant_idx].sum()

            total_points_processed += batch_size

            if save_spikes:  # [EXP] per-sample spike counts of the training pass
                spike_store.append(batch_train_spikes.to(torch.int16).cpu().numpy())
                label_store.append(targets.to(torch.int8).cpu().numpy())

            # Cooper / BindsNET-style estimate: fill the window; every cooper_update_steps batches score it with the
            # assignments from the previous windows, then update the (cumulative) assignments.
            if track_cooper and batch_size == cooper_batch_size:   # [EXP] was `track_accuracy_curve and ...`
                slot = (global_batch_idx % cooper_update_steps) * cooper_batch_size
                cooper_spike_record[slot:slot + batch_size] = batch_train_spikes
                cooper_label_record[slot:slot + batch_size] = targets
                if (global_batch_idx + 1) % cooper_update_steps == 0:
                    all_pred = bindsnet_all_activity(cooper_spike_record, cooper_assignments, 10)
                    prop_pred = bindsnet_proportion_weighting(cooper_spike_record, cooper_assignments, cooper_proportions, 10)
                    cooper_points_history.append(total_points_processed)
                    cooper_all_history.append(100.0 * (all_pred == cooper_label_record).float().mean().item())
                    cooper_proportion_history.append(100.0 * (prop_pred == cooper_label_record).float().mean().item())
                    cooper_assignments, cooper_proportions, cooper_rates = bindsnet_assign_labels(cooper_spike_record, cooper_label_record, 10, cooper_rates)
                    # [EXP] diagnostics of this window
                    per_sample = cooper_spike_record.sum(dim=1)
                    th = model.neurons.theta
                    diag["points"].append(total_points_processed)
                    diag["silent_frac"].append((per_sample == 0).float().mean().item())   # samples on which NO neuron fired
                    diag["mean_spikes"].append(per_sample.mean().item())                  # total output spikes per sample
                    diag["active_neurons"].append(int((cooper_spike_record.sum(dim=0) > 0).sum().item()))
                    diag["theta_mean"].append(th.mean().item())    # type: ignore
                    diag["theta_min"].append(th.min().item())      # type: ignore
                    diag["theta_max"].append(th.max().item())      # type: ignore
                    diag["w_std"].append(model.synapses.weight.std().item())
                    if len(diag["points"]) % 50 == 0:
                        save_curves()
            global_batch_idx += 1

            if track_dominant_neuron and dominant_idx is not None:
                track_steps.append(total_points_processed)
                track_vmax.append(highest_v_in_batch.item())
                track_theta.append(model.neurons.theta[dominant_idx].item())
                track_spiked.append(dominant_spike_count.item() > 0)

            # Estimated accuracy evaluation
            if track_accuracy_curve and batch_idx % current_interval() == 0:   # [EXP] current_interval()
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Estimating current accuracy...")
                _rng_state = save_rng(device)   # [EXP] evaluation must not disturb the training RNG stream
                torch.manual_seed(EVAL_SEED)
                _t_eval = time.time()
                temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
                snapshot_acc = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps, return_matrix=False)
                restore_rng(_rng_state, device)
                print(snapshot_acc, f"(eval took {time.time() - _t_eval:.1f}s)")
                points_history.append(total_points_processed)
                accuracy_history.append(snapshot_acc)
                heldout_unassigned.append(int((temp_labels == -1).sum().item()))   # [EXP]
                save_curves()                                                       # [EXP]

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} processed. "
                      f"[{total_points_processed} samples, {time.time() - t_start:.0f}s elapsed]")
    print("Training complete")
    save_curves()   # [EXP]
    if save_spikes and spike_store:   # [EXP]
        np.savez_compressed(output_dir / "train_spikes.npz",
                            spikes=np.concatenate(spike_store).astype(np.int16),
                            labels=np.concatenate(label_store).astype(np.int8))

    # Plot accuracy curve
    if track_accuracy_curve and len(points_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(cooper_points_history, cooper_all_history, linestyle='-', color='tab:gray', linewidth=1, alpha=0.8, label=f'BindsNET "all" (training window of {cooper_update_interval})')
        plt.plot(cooper_points_history, cooper_proportion_history, linestyle='-', color='tab:orange', linewidth=1, alpha=0.8, label=f'BindsNET "proportion" (training window of {cooper_update_interval})')
        plt.plot(points_history, accuracy_history, linestyle='-', color='blue', linewidth=2, label='Held-out (assign on train subset, score on test subset)')
        plt.title(f"Accuracy Curve ({model.num_neurons} Neurons) | {output_dir.name}", fontsize=11)
        plt.xlabel("Number of Training Points Processed")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_dir / "accuracy_curve.png")   # [EXP] fixed name inside results/<name>/; raw data is in curves.npz
        plt.close()

    # Plot the neuron voltage-theta/time graph
    if track_dominant_neuron and dominant_idx is not None and len(track_steps) > 1:
        print(f"Plotting voltage plot for passive neuron #{dominant_idx}...")
        plt.figure(figsize=(12, 6))
        plt.plot(track_steps, track_vmax, color='blue', label='Max Voltage ($V_{max}$)', linewidth=2)
        points = np.array([track_steps, track_theta]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_colors = ['red' if spiked else 'green' for spiked in track_spiked[1:]]
        lc = LineCollection(segments, colors=segment_colors, linewidth=2) # type: ignore
        plt.gca().add_collection(lc)

        plt.title(f"Passive Neuron #{dominant_idx} Voltage-Theta/Time Graph", fontsize=16)
        plt.xlabel("Number of Training Points Processed", fontsize=12)
        plt.ylabel("Voltage (mV)", fontsize=12)
        plt.axvspan(115000, 130000, color='gray', alpha=0.2, label='Observed Accuracy Spike Window')

        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Max Voltage ($V_{max}$)'),
            Line2D([0], [0], color='red', lw=2, label=r'Threshold ($\theta$) - Spiking'),   # [EXP] raw string (\t was a tab)
            Line2D([0], [0], color='green', lw=2, label=r'Threshold ($\theta$) - Not Spiking'),
            plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.2, label='Observed Accuracy Spike Window') # type: ignore
        ]
        plt.legend(handles=legend_elements, fontsize=12, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"neuron_{dominant_idx}_intersection.png", dpi=300)
        plt.close()

    if generate_confusion_matrix:
        temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
        _, c_matrix = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps, return_matrix=True) # type: ignore
        visualize_confusion_matrix(c_matrix, total_points_processed)

    if visualize_templates:
        visualize_learned_templates(model)

    return model


if __name__ == '__main__':
    # [EXP] command line, so run_experiments.py can launch every configuration without editing this file
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="A_base", help="run name -> results/<name>/")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    # network / dynamics
    p.add_argument("--n_neurons", type=int, default=400)
    p.add_argument("--theta_plus", type=float, default=0.05)
    p.add_argument("--theta_div", type=float, default=16.0, help="theta += spikes*theta_plus/theta_div (original: 16)")
    p.add_argument("--refrac", type=int, default=0, help="refractory steps (0=off)")
    p.add_argument("--inh", type=float, default=200.0, help="lateral inhibition strength")
    p.add_argument("--wta_mode", default="off", choices=["off", "v", "vt", "vt_reset"],
                   help="[WTA] hard winner-take-all variant (see excitatory_neurons): off | v (old) | vt | vt_reset")
    p.add_argument("--wta", action="store_true", help="[WTA] deprecated alias for --wta_mode v")
    p.add_argument("--lr_scale", type=float, default=1.0, help="multiplies both STDP learning rates")
    # input
    p.add_argument("--num_steps", type=int, default=350)
    p.add_argument("--encoding", default="rate", choices=["rate", "rate_norm"])
    p.add_argument("--intensity", type=float, default=0.1, help="pixel*intensity = spike probability per step")
    p.add_argument("--norm_total", type=float, default=75.0)
    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train", type=int, default=None, help="stop after this many samples (smoke test)")
    # accuracy tracking
    p.add_argument("--snapshot_interval", type=int, default=50, help="held-out eval every N batches")
    p.add_argument("--dense_start", type=int, default=0)
    p.add_argument("--dense_end", type=int, default=0)
    p.add_argument("--dense_interval", type=int, default=0, help="eval every N batches for samples in [dense_start, dense_end]")
    p.add_argument("--assign_n", type=int, default=2000)
    p.add_argument("--eval_n", type=int, default=1000)
    p.add_argument("--no_heldout", action="store_true")
    p.add_argument("--no_cooper", action="store_true")
    p.add_argument("--save_spikes", action="store_true", help="save per-sample training spikes (offline Cooper analysis)")
    p.add_argument("--templates", action="store_true")
    p.add_argument("--confusion", action="store_true")
    args = p.parse_args()
    if args.wta and args.wta_mode == "off":
        args.wta_mode = "v"

    output_dir = script_dir / args.results_dir / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    ENC.update(encoding=args.encoding, intensity=args.intensity, norm_total=args.norm_total)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.backends.mps.is_available():
        device: torch.device = torch.device("mps")
        print("Using MPS.")
    elif torch.cuda.is_available():
        device: torch.device = torch.device("cuda")
        print("Using CUDA.")
    else:
        device: torch.device = torch.device("cpu")
        print("Using CPU.")

    train_loader, test_loader, snapshot_assign_loader, snapshot_eval_loader = build_loaders(
        args.batch_size, args.assign_n, args.eval_n)

    model: DiehlAndCookNetwork = DiehlAndCookNetwork(
        num_neurons=args.n_neurons, theta_plus=args.theta_plus, theta_div=args.theta_div,
        refrac=args.refrac, inh=args.inh, lr_scale=args.lr_scale, wta_mode=args.wta_mode).to(device)
    model.train() # This probably doesn't make a difference.
    train(
        model=model,
        train_loader=train_loader,
        snapshot_assign_loader=snapshot_assign_loader,
        snapshot_eval_loader=snapshot_eval_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=args.epochs,
        num_steps=args.num_steps,
        track_accuracy_curve=not args.no_heldout,
        track_dominant_neuron=False,
        generate_confusion_matrix=args.confusion,
        visualize_templates=args.templates,
        track_cooper=not args.no_cooper,
        snapshot_interval=args.snapshot_interval,
        dense_range=(args.dense_start, args.dense_end) if args.dense_interval > 0 else None,
        dense_interval=args.dense_interval if args.dense_interval > 0 else None,
        max_train_samples=args.max_train,
        save_spikes=args.save_spikes,
        config=vars(args),
    )