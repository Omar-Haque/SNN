import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import math
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int

train_dataset: datasets.MNIST = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset: datasets.MNIST = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader:  DataLoader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

assign_subset = torch.utils.data.Subset(train_dataset, range(0, 2000))
eval_subset   = torch.utils.data.Subset(test_dataset, range(0, 1000))
snapshot_assign_loader = DataLoader(assign_subset, batch_size=512, shuffle=False)
snapshot_eval_loader   = DataLoader(eval_subset, batch_size=512, shuffle=False)

class excitatory_neurons(nn.Module):
    def __init__(self):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = 400
        
        self.theta_base:  float = -52.0
        self.theta_plus:  float = 0.05
        self.theta_decay: float = math.exp(-1.0 / 1e7)
        
        self.v:     torch.Tensor | None = None
        self.g_e:   torch.Tensor | None = None
        self.g_i:   torch.Tensor | None = None
        self.theta: torch.Tensor | None = (torch.ones(self.num_neurons) * self.theta_base)
        
        self.tau_m :   float = 100.0
        self.decay_ge: float = math.exp(-1.0 / 1.0)
        self.decay_gi: float = math.exp(-1.0 / 2.0)
        
        self.E_rest: float = -65.0
        self.E_exc:  float = 0.0
        self.E_inh:  float = -100.0
        
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

class DiehlAndCookNetwork(nn.Module):
    def __init__(self):
        super(DiehlAndCookNetwork, self).__init__()
        
        self.num_neurons: int = 400
        
        self.synapses: nn.Linear = nn.Linear(784, self.num_neurons, bias=False)
        self.synapses.weight.requires_grad = False
        
        self.w_max:    float = 1.0
        self.lr_plus:  float = 5e-2
        self.lr_minus: float = 5e-4
        
        self.decay_x: float = math.exp(-1.0 / 20.0)
        self.decay_y: float = math.exp(-1.0 / 20.0)
        
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3)
        with torch.no_grad():
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
        
        self.neurons: excitatory_neurons = excitatory_neurons()
        
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
    
    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        x_exc: torch.Tensor = self.synapses(input_spikes)
        x_inh: torch.Tensor = torch.matmul(self.prev_spikes, self.inh_matrix)
        
        current_spikes: torch.Tensor = self.neurons(x_exc, x_inh, learning=learning)
        
        if learning:
            self.x_trace = (self.x_trace * self.decay_x) + input_spikes
            self.y_trace = (self.y_trace * self.decay_y) + current_spikes
            
            batch_size: int = input_spikes.shape[0]
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace) / batch_size
            delta_w_plus: torch.Tensor = (self.lr_plus) * post_interaction * (self.w_max - self.synapses.weight)
            
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes) / batch_size
            delta_w_minus = (self.lr_minus) * pre_interaction * self.synapses.weight
            
            self.synapses.weight.add_(delta_w_plus - delta_w_minus)
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
            
        self.prev_spikes = current_spikes
        return current_spikes

@torch.no_grad()
def assign_neuron_labels(model: DiehlAndCookNetwork, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    print("Assigning class labels to neurons")
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
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        for digit in range(10):
            mask = (targets == digit)
            if mask.any():
                spike_counts[:, digit] += batch_neuron_spikes[mask].sum(dim=0) / mask.sum().float()
                  
    neuron_assignments: torch.Tensor = torch.argmax(spike_counts, dim=1)
    total_spikes = spike_counts.sum(dim=1)
    dead_neurons = (total_spikes == 0)
    neuron_assignments[dead_neurons] = -1
    
    print("Neuron assignments complete")
    return neuron_assignments

@torch.no_grad()
def evaluate_network(model: DiehlAndCookNetwork, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int, return_matrix: bool = False) -> float | tuple[float, np.ndarray]:
    print("Evaluating network accuracy on test set...")
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
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) 
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        class_votes: torch.Tensor = torch.matmul(batch_neuron_spikes, assignment_matrix)
        predictions: torch.Tensor = torch.argmax(class_votes, dim=1)
        
        correct_predictions += (predictions == targets).sum().item()
        total_predictions += batch_size
        
    accuracy = (correct_predictions / total_predictions) * 100.0
    if return_matrix:
        return accuracy, conf_matrix.cpu().numpy() 
    return accuracy

if torch.backends.mps.is_available():
    device: torch.device = torch.device("mps")
    print("Using MPS.")
elif torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
    print("Using CUDA.")
else:
    device: torch.device = torch.device("cpu")
    print("Using CPU.")

model: DiehlAndCookNetwork = DiehlAndCookNetwork().to(device)

num_epochs:  int = 3
num_steps:   int = 350

model.train()

total_points_processed = 0

norm_points_history = []
frobenius_norm_history = []

accuracy_points_history = []
accuracy_history = []

snapshot_interval = 100 

for epoch in range(num_epochs):
    data: Float[torch.Tensor, "batch channels height width"]
    targets: Int[torch.Tensor, "batch"]
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) 
        spike_data = spike_data.view(num_steps, batch_size, 784)

        model.reset(batch_size, device)
        
        batch_frobenius_sum: float = 0.0
        
        for t in range(num_steps):
            w_before_step: torch.Tensor = model.synapses.weight.detach().clone()
            current_input_spikes: torch.Tensor = spike_data[t]
            
            out_spikes: torch.Tensor = model(current_input_spikes)
            
            weight_diff = model.synapses.weight - w_before_step
            step_norm = torch.linalg.norm(weight_diff, ord='fro').item()
            batch_frobenius_sum += step_norm
        
        total_points_processed += batch_size
        
        
        norm_points_history.append(total_points_processed)
        frobenius_norm_history.append(batch_frobenius_sum)
        
        
        if batch_idx % snapshot_interval == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Estimating current accuracy...")
            temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
            snapshot_acc = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps, return_matrix=False)
            print(snapshot_acc)
            
            accuracy_points_history.append(total_points_processed)
            accuracy_history.append(snapshot_acc)
            
            print(f"Accuracy: {snapshot_acc:.2f}% | Batch Frobenius Norm: {batch_frobenius_sum:.4f}")

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} processed.")

print("Training complete")

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Number of Training Points Processed")
ax1.set_ylabel("Accuracy (%) on Validation Subset", color="blue")
ax1.plot(accuracy_points_history, accuracy_history, linestyle='-', color='blue', linewidth=2, label="Accuracy")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()  
ax2.set_ylabel("Batch Weight Frobenius Norm Sum", color="purple")
ax2.plot(norm_points_history, frobenius_norm_history, linestyle='-', color='purple', alpha=0.5, linewidth=1, label="Frobenius Norm")
ax2.tick_params(axis="y", labelcolor="purple")

plt.title(f"Accuracy vs. Per-Batch Weight Update Frobenius Norm ({model.num_neurons} Neurons)")
fig.tight_layout()
plt.show()

neuron_labels = assign_neuron_labels(model, train_loader, device, num_steps)
test_accuracy = evaluate_network(model, test_loader, neuron_labels, device, num_steps)
print(test_accuracy)