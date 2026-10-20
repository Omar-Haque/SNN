# Hypothesis 2: At smaller theta_plus values, the spike happens later, and earlier for larger theta_plus values
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import math
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from pathlib import Path
import random


script_dir: Path = Path(__file__).parent

# Import MNIST train and test
train_dataset: datasets.MNIST = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor() # convert images to torch tensors
)

test_dataset: datasets.MNIST = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor() # convert images to torch tensors
)

# Load the MNIST dataset
train_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader:  DataLoader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Small subsets of MNIST for running average
# Use 2000 images to assign labels and 1000 images to test
assign_subset = torch.utils.data.Subset(train_dataset, range(0, 2000))
eval_subset   = torch.utils.data.Subset(test_dataset, range(0, 1000))
snapshot_assign_loader = DataLoader(assign_subset, batch_size=512, shuffle=False)
snapshot_eval_loader   = DataLoader(eval_subset, batch_size=512, shuffle=False)

# EXCITATORY NEURONS
class excitatory_neurons(nn.Module):
    def __init__(self, theta_plus: float = 0.0):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = 400  # number of excitatory neurons.
        
        # For the adaptive threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = theta_plus           # penalty added to the minimum fire threshold each time a neuron fires.
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
        self.v     = torch.ones(batch_size, self.num_neurons, device=device) * self.E_rest
        self.g_e   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.g_i   = torch.zeros(batch_size, self.num_neurons, device=device)
        self.theta = self.theta.to(device) 

    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor, learning: bool = True) -> torch.Tensor:
        if self.v is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")
        
        self.v = self.decay * (self.v - self.E_rest) + self.E_rest
        
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        self.v += x_exc - x_inh
        
        self.raw_v: torch.Tensor = self.v.clone()
        
        spikes: torch.Tensor = (self.v >= self.theta).float()
        self.v[spikes == 1.0] = self.E_rest
        
        if learning:
            self.theta += (spikes.sum(dim=0) * self.theta_plus) / 16

        return spikes


# Architecture of the entire Diehl and Cook network
class DiehlAndCookNetwork(nn.Module):
    def __init__(self, theta_plus: float = 0.0):
        super(DiehlAndCookNetwork, self).__init__()
        
        self.num_neurons: int = 400
        
        self.synapses: nn.Linear = nn.Linear(784, self.num_neurons, bias=False)
        self.synapses.weight.requires_grad = False
        
        # STDP parameters
        self.w_max:    float = 1.0
        self.lr_plus:  float = 1e-2
        self.lr_minus: float = 1e-4
        
        # Trace decay multipliers
        self.decay_x: float = math.exp(-1.0 / 20.0)
        self.decay_y: float = math.exp(-1.0 / 20.0)
        
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3)
        with torch.no_grad():
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
        
        self.neurons: excitatory_neurons = excitatory_neurons(theta_plus=theta_plus)
        
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * 200.0
        self.inh_matrix.fill_diagonal_(0.0)
        
        self.prev_spikes: torch.Tensor = torch.empty(0)
        self.x_trace: torch.Tensor = torch.empty(0)
        self.y_trace: torch.Tensor = torch.empty(0)
        
    def reset(self, batch_size: int, device: torch.device) -> None:
        self.neurons.reset(batch_size, device)
        self.inh_matrix  = self.inh_matrix.to(device)
        self.prev_spikes = torch.zeros(batch_size, self.num_neurons, device=device)
        self.x_trace     = torch.zeros(batch_size, 784, device=device)
        self.y_trace     = torch.zeros(batch_size, self.num_neurons, device=device)
    
    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_exc: torch.Tensor = self.synapses(input_spikes)
        x_inh: torch.Tensor = torch.matmul(self.prev_spikes, self.inh_matrix)
        
        current_spikes: torch.Tensor = self.neurons(x_exc, x_inh, learning=learning)
        delta_w: torch.Tensor | None = None
        
        if learning:
            self.x_trace = (self.x_trace * self.decay_x) + input_spikes
            self.y_trace = (self.y_trace * self.decay_y) + current_spikes
            
            batch_size: int = input_spikes.shape[0]
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace) / batch_size
            delta_w_plus: torch.Tensor = (self.lr_plus) * post_interaction * (self.w_max - self.synapses.weight)
            
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes) / batch_size
            delta_w_minus = (self.lr_minus) * pre_interaction * self.synapses.weight
            delta_w = delta_w_plus - delta_w_minus
            
            self.synapses.weight.add_(delta_w)
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
            
        self.prev_spikes = current_spikes
        return current_spikes, delta_w
        
def visualize_learned_templates(model: DiehlAndCookNetwork, theta_plus_val: float) -> None:
    print(f"Extracting weights and thresholds after training (theta_plus={theta_plus_val})...")
    weights: np.ndarray = model.synapses.weight.detach().cpu().numpy()
    weight_images: np.ndarray = weights.reshape(model.num_neurons, 28, 28)
    thetas: np.ndarray = model.neurons.theta.detach().cpu().numpy() 
    
    fig, axes = plt.subplots(20, 20, figsize=(18, 18))
    fig.suptitle(f"Synaptic Weights Visualization (theta_plus={theta_plus_val})", fontsize=20, y=0.95)
    plt.subplots_adjust(wspace=0.05, hspace=0.4) 
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(weight_images[i], cmap='hot', interpolation='nearest')
        ax.set_title(f"θ: {thetas[i]:.1f}", fontsize=8, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(script_dir / f"theta_templates_theta_{theta_plus_val}.png")
    plt.close(fig)

@torch.no_grad()
def assign_neuron_labels(model: DiehlAndCookNetwork, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    model.eval()
    spike_counts: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) 
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        for t in range(num_steps):
            out_spikes, _ = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        for digit in range(10):
            mask = (targets == digit)
            if mask.any():
                spike_counts[:, digit] += batch_neuron_spikes[mask].sum(dim=0) / mask.sum().float()
                  
    neuron_assignments: torch.Tensor = torch.argmax(spike_counts, dim=1)
    total_spikes = spike_counts.sum(dim=1)
    dead_neurons = (total_spikes == 0)
    neuron_assignments[dead_neurons] = -1
    return neuron_assignments

@torch.no_grad()
def evaluate_network(model: DiehlAndCookNetwork, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int, return_matrix: bool = False) -> float | tuple[float, np.ndarray]:
    model.eval()
    correct_tensor = torch.tensor(0.0, device=device)
    total_predictions: int = 0
    
    conf_matrix = torch.zeros(10, 10, dtype=torch.int32, device=device)
    assignment_matrix: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    valid_mask = (neuron_assignments != -1)
    safe_assignments = torch.where(valid_mask, neuron_assignments, torch.zeros_like(neuron_assignments))
    assignment_matrix.scatter_(1, safe_assignments.unsqueeze(1), 1.0)
    assignment_matrix[~valid_mask] = 0.0
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) 
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        for t in range(num_steps):
            out_spikes, _ = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        class_votes: torch.Tensor = torch.matmul(batch_neuron_spikes, assignment_matrix)
        predictions: torch.Tensor = torch.argmax(class_votes, dim=1)
        
        for true_label, predicted_label in zip(targets, predictions):
            conf_matrix[true_label.long(), predicted_label.long()] += 1
        
        correct_tensor += (predictions == targets).sum()
        total_predictions += batch_size
        
    accuracy = (correct_tensor.float() / total_predictions).item() * 100.0
    if return_matrix:
        return accuracy, conf_matrix.cpu().numpy() 
    return accuracy


def train(
    model: DiehlAndCookNetwork,
    train_loader: DataLoader,
    snapshot_assign_loader: DataLoader,
    snapshot_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    theta_plus_val: float,
    num_epochs: int = 5,
    num_steps: int = 350,
    track_accuracy_curve: bool = True
) -> tuple[DiehlAndCookNetwork, list, list]:
    
    run_seed = random.randint(0, 2**31 - 1)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    
    total_points_processed = 0
    points_history_per_snapshot = []
    accuracy_history = []
    snapshot_interval = 100

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            batch_size: int = data.shape[0]
            data, targets = data.to(device), targets.to(device)
            
            spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) 
            spike_data = spike_data.view(num_steps, batch_size, 784)

            model.reset(batch_size, device)
            
            for t in range(num_steps):
                out_spikes, _ = model(spike_data[t])
                
            total_points_processed += batch_size
            
            if track_accuracy_curve and batch_idx % snapshot_interval == 0:
                print(f"Theta+: {theta_plus_val} | Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Estimating current accuracy...")
                temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
                snapshot_acc = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps, return_matrix=False)
                print(f"Accuracy: {snapshot_acc:.2f}%")
                points_history_per_snapshot.append(total_points_processed)
                accuracy_history.append(snapshot_acc)
        
            if batch_idx % 100 == 0:
                print(f"Theta+: {theta_plus_val} | Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} processed.")
                
    print(f"Training complete for theta_plus = {theta_plus_val}")
    return model, points_history_per_snapshot, accuracy_history


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device: torch.device = torch.device("mps")
        print("Using MPS.")
    elif torch.cuda.is_available():
        device: torch.device = torch.device("cuda")
        print("Using CUDA.")
    else:
        device: torch.device = torch.device("cpu")
        print("Using CPU.")
    
    theta_plus_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    num_epochs_per_run = 5

    for theta_val in theta_plus_values:
        print(f"\n================ Running training with theta_plus = {theta_val} ================")
        model: DiehlAndCookNetwork = DiehlAndCookNetwork(theta_plus=theta_val).to(device)
        model.train()
        
        model, points_history, accuracy_history = train(
            model=model,
            train_loader=train_loader,
            snapshot_assign_loader=snapshot_assign_loader,
            snapshot_eval_loader=snapshot_eval_loader,
            test_loader=test_loader,
            device=device,
            theta_plus_val=theta_val,
            num_epochs=num_epochs_per_run,
            num_steps=350,
            track_accuracy_curve=True
        )
        
        # Save individual accuracy-time curve directly in script_dir
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(points_history, accuracy_history, color='blue', label=f'Theta+ = {theta_val} Accuracy', linewidth=2)
        ax.xaxis.set_major_locator(MultipleLocator(50000))
        ax.xaxis.set_minor_locator(MultipleLocator(5000))
        ax.grid(True, which='major', linestyle='--', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)
        plt.title(f"Accuracy / No. of Training Examples (theta_plus = {theta_val}, {num_epochs_per_run} Epochs)", fontsize=14)
        plt.xlabel("Number of Training Points", fontsize=12)
        plt.ylabel("Accuracy (%) on Validation Subset", fontsize=12)
        plt.legend(loc='lower right', fontsize=11)
        plt.tight_layout()
        plt.savefig(script_dir / f"theta_plus_{theta_val}_accuracy-time.png", dpi=300)
        plt.close(fig)
        
        # Save learned templates directly in script_dir
        visualize_learned_templates(model, theta_val)