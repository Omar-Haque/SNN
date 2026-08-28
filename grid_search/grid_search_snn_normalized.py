import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import math
import numpy as np
import itertools
import csv
import os
# type annotations everywhere because I remember messing up one of my RL projects because of confusing tensors and numpy arrays.
from jaxtyping import Float, Int

# 1. Dataset Setup
# Import MNIST train and test
train_dataset: datasets.MNIST = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor() # convert images to torch tensors
)
test_dataset: datasets.MNIST = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor() # convert images to torch tensors
)

# 2. Refactored Excitatory Neurons (Accepts hyperparameters as arguments)
# EXCITATORY NEURONS
# Excitatory neurons are an entire class, but inhibitory neurons can be simulated using matrices.
class excitatory_neurons(nn.Module):
    def __init__(self, num_neurons: int, theta_plus: float, theta_decay_tau: float):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = num_neurons # number of excitatory neurons.
        
        # For the adaptive threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = theta_plus           # a penalty added to the minimum fire threshold each time a neuron fires.
        self.theta_decay: float = math.exp(-1.0 / theta_decay_tau)
        
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

    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor, learning: bool = True) -> torch.Tensor:
        if self.v is None or self.g_e is None or self.g_i is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")
        
        # Update conductances.
        self.g_e = (self.g_e * self.decay_ge) + x_exc
        self.g_i = (self.g_i * self.decay_gi) + x_inh
        
        # Update voltages.
        dV: torch.Tensor = (self.E_rest - self.v) + (self.g_e * (self.E_exc - self.v)) + (self.g_i * (self.E_inh - self.v)) # equation in the paper.
        dV /= self.tau_m
        self.v += dV
        
        # Check which neurons have fired. These are the neurons which crossed the threshold.
        spikes: torch.Tensor = (self.v >= self.theta).float()
        
        # Only the highest voltage neuron is allowed to fire.
        # This is to overcome the problem of multiple neurons crossing the threshold together, since time proceeds in a discretized 1.0 ms manner.
        competing_v = self.v.clone()
        competing_v[spikes == 0.0] = -1000.0 # Instantly disqualify non-spiking neurons
        
        # Only the highest voltage strictly among the eligible spiking neurons is allowed to fire.
        max_v_indices: torch.Tensor = torch.argmax(competing_v, dim=1)
        
        # Create a blank mask and put a 1.0 only at the winning index
        wta_mask = torch.zeros_like(spikes) # shape is (batch_size, num_neurons)
        wta_mask.scatter_(1, max_v_indices.unsqueeze(1), 1.0) # a 1.0 is filled in the column of the winning for each image in the batch.
        
        # Filter the spikes: you must have crossed the threshold AND won the WTA race
        spikes = spikes * wta_mask # "spikes" contains neurons which crossed the threshold. Out of these, let only winner remain. 
        
        # Reset the neurons which have spiked, back to E_rest voltage
        self.v[spikes == 1.0] = self.E_rest
        
        if learning:
            # Adaptive thresholding (simulates homeostasis)
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
            # TODO: Decide on mean or sum. mean fails to work with my current set of hyperparameters.
            self.theta = self.theta + (spikes.mean(dim=0) * self.theta_plus)
            self.theta.clamp_(max=0.0)
        
        return spikes

# 3. Refactored Network (Accepts hyperparameters)
# Architecture of the entire Diehl and Cook network
# STDP is also performed within this class
class DiehlAndCookNetwork(nn.Module):
    def __init__(self, num_neurons: int, lr_plus: float, lr_minus: float, 
                 theta_plus: float, theta_decay_tau: float, inh_penalty: float):
        super(DiehlAndCookNetwork, self).__init__()
        
        self.num_neurons: int = num_neurons
        self.synapses: nn.Linear = nn.Linear(784, self.num_neurons, bias=False) # connect 784 pixels to each excitatory neuron.
        self.synapses.weight.requires_grad = False # Using STDP instead
        
        # STDP parameters
        self.w_max:    float = 1.0    # weight upper limit
        self.lr_plus:  float = lr_plus   # Learning rate for potentiation
        self.lr_minus: float = lr_minus  # Learning rate for depression
        
        # Trace decay multipliers
        self.decay_x: float = math.exp(-1.0 / 20.0) # Pre-synaptic trace decay
        self.decay_y: float = math.exp(-1.0 / 20.0) # Post-synaptic trace decay
        
        # Initialize weights uniformly between a and b
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3)
        with torch.no_grad():
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
        
        self.neurons: excitatory_neurons = excitatory_neurons(num_neurons, theta_plus, theta_decay_tau)
        
        # LATERAL INHIBITION USING A MATRIX
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * inh_penalty
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
            # TODO: divide by batch_size? No, that would require a different set of hyperparameters.
            batch_size: int = input_spikes.shape[0]
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace) / batch_size
            delta_w_plus: torch.Tensor = (self.lr_plus * batch_size) * post_interaction * (self.w_max - self.synapses.weight)
            
            # When input fired, look back at output trace (depression)
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes) / batch_size
            delta_w_minus = (self.lr_minus * batch_size) * pre_interaction * self.synapses.weight
            
            # Apply changes to synapses
            self.synapses.weight.add_(delta_w_plus - delta_w_minus) # delta_w = η(x_pre − x_tar)(w_max − w)^μ
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            
            # BindsNET weight normalization
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
            
        self.prev_spikes = current_spikes
        return current_spikes

# 4. Evaluation Functions
@torch.no_grad()
def assign_neuron_labels(model: DiehlAndCookNetwork, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    model.eval()
    
    # Create a tensor to count spikes: shape [num_neurons neurons, 10 digit classes]
    spike_counts: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # Find the total sum of white pixels for each image in the batch
        image_sums = data.sum(dim=(1, 2, 3), keepdim=True) + 1e-5
        # Force every image to have the exact same total sum of 75.0
        data = (data / image_sums) * 75.0
        
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
    
    return neuron_assignments

@torch.no_grad()
def evaluate_network(model: DiehlAndCookNetwork, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int) -> float:
    model.eval()
    correct_predictions: int = 0
    total_predictions: int = 0
    
    # PRE-COMPUTE: Create a one-hot mapping matrix [num_neurons, 10]
    # This matrix routes spikes from specific neurons into their assigned digit buckets
    assignment_matrix: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    # Safely scatter the 1.0s, ignoring the -1 dead neurons
    valid_mask = (neuron_assignments != -1)
    safe_assignments = torch.where(valid_mask, neuron_assignments, torch.zeros_like(neuron_assignments))
    assignment_matrix.scatter_(1, safe_assignments.unsqueeze(1), 1.0)
    assignment_matrix[~valid_mask] = 0.0 # Erase the temporary zeros used for dead neurons
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # Find the total sum of white pixels for each image in the batch
        image_sums = data.sum(dim=(1, 2, 3), keepdim=True) + 1e-5
        # Force every image to have the exact same total sum of 75.0
        data = (data / image_sums) * 75.0
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        
        # Track the total spikes fired by each neuron over the 350ms window
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        # Classify ALL images in the batch using one clean matrix multiplication!
        # Result shape: [batch_size, 10]
        class_votes: torch.Tensor = torch.matmul(batch_neuron_spikes, assignment_matrix)
        
        # The network's final guess is the class with the highest total spikes (evaluated across the whole batch)
        predictions: torch.Tensor = torch.argmax(class_votes, dim=1)
        
        # Instantly count how many predictions match the targets
        correct_predictions += (predictions == targets).sum().item()
        total_predictions += batch_size
            
    return (correct_predictions / total_predictions) * 100.0

# 5. GRID SEARCH IMPLEMENTATION
def run_grid_search():
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # Standard settings
    num_steps: int = 350
    batch_size: int = 256
    num_neurons: int = 400
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # --- DEFINE YOUR SEARCH SPACE HERE ---
    param_grid = {
        'lr_plus': [0.01, 0.03],                 # Standard vs Higher LTP
        'lr_minus': [0.0001, 0.003],             # Weak vs Strong LTD
        'theta_plus': [0.05, 0.1, 0.5],          # Fatigue penalty
        'theta_decay_tau': [1e5, 1e7, 1e9],      # Decay speeds
        'inh_penalty': [120.0, 200.0]            # Lateral inhibition strength
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    print(f"Starting Grid Search with {len(combinations)} combinations.")
    
    # Create a CSV to log results incrementally
    csv_file = "grid_search_results_normalized.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write headers
            writer.writerow(list(keys) + ['Accuracy (%)', 'Dead Neurons'])

    best_acc = 0.0
    best_params = None

    for idx, params in enumerate(combinations):
        print(f"\n--- Testing Combination {idx+1}/{len(combinations)} ---")
        print(params)
        
        # Instantiate a FRESH model for this combination
        model = DiehlAndCookNetwork(
            num_neurons=num_neurons,
            lr_plus=params['lr_plus'],
            lr_minus=params['lr_minus'],
            theta_plus=params['theta_plus'],
            theta_decay_tau=params['theta_decay_tau'],
            inh_penalty=params['inh_penalty']
        ).to(device)
        
        # Train for 1 Epoch (Grid search benchmark)
        num_epochs: int = 1
        model.train()
        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(train_loader):
                current_batch_size = data.shape[0]
                data = data.to(device)
                
                # Find the total sum of white pixels for each image in the batch
                image_sums = data.sum(dim=(1, 2, 3), keepdim=True) + 1e-5
                # Force every image to have the exact same total sum of 75.0
                data = (data / image_sums) * 75.0
                
                spike_data = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
                spike_data = spike_data.view(num_steps, current_batch_size, 784)
                
                model.reset(current_batch_size, device)
                for t in range(num_steps):
                    model(spike_data[t])
                
        # Evaluate
        print("Assigning labels and evaluating...")
        neuron_labels = assign_neuron_labels(model, train_loader, device, num_steps)
        dead_count = (neuron_labels == -1).sum().item()
        
        accuracy = evaluate_network(model, test_loader, neuron_labels, device, num_steps)
        print(f"Accuracy: {accuracy:.2f}% | Dead Neurons: {dead_count}")
        
        # Log to CSV in case of a crash
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list(params.values()) + [accuracy, dead_count])
            
        # Track best
        if accuracy > best_acc:
            best_acc = accuracy
            best_params = params
            
    print("\n======================================")
    print(f"GRID SEARCH COMPLETE. Best Accuracy: {best_acc:.2f}%")
    print(f"Best Parameters: {best_params}")
    print("======================================")

if __name__ == "__main__":
    run_grid_search()