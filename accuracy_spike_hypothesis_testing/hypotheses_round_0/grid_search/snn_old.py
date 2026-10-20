import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikegen, surrogate, utils
import math
import matplotlib.pyplot as plt
import numpy as np

# type annotations everywhere because I remember messing up one of my RL projects because of confusing tensors and numpy arrays.
from typing import Tuple
from jaxtyping import Float, Int

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

train_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader: DataLoader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# data: Float[torch.Tensor, "batch channels height width"]
# targets: Int[torch.Tensor, "batch"]
# for data, targets in train_loader:
#     # data.shape = [batch_size=64, 1, 28, 28] (number, channels, width, height)
#     spike_data: torch.Tensor = spikegen.rate(data, num_steps = num_steps) # type: ignore

#     # spike_data.shape = [num_steps, batch_size, 1, 28, 28]
#     # Flatten the channel, width, and height dimensions
#     spike_data: torch.Tensor = spike_data.view(spike_data.shape[0], spike_data.shape[1], 784)

# EXCITATORY NEURONS
# Excitatory neurons are an entire class, but inhibitory neurons can be simulated using simple matrices.
class excitatory_neurons(nn.Module):
    def __init__(self):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = 400  # number of excitatory neurons.
        
        # For the adapting threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = 0.05                 # a penalty added to the minimum fire threshold each time a neuron fires.
        self.theta_decay: float = math.exp(-1.0 / 1e7) # a multiplier to progressively decay the threshold back down to theta_base.
        
        # State variables for leaky integrate-and-fire model. Used as memory for neuron states.
        self.v:     torch.Tensor | None = None     # self.v[i] is the ith neuron's voltage.
        self.g_e:   torch.Tensor | None = None     # self.g_e[i] is the ith neuron's excitatory conductance.
        self.g_i:   torch.Tensor | None = None     # self.g_i[i] is the ith neuron's inhibitory conductance.
        # noise = torch.rand(self.num_neurons) * 10.0 # add up to 2 mV of random noise to prevent pytorch ties
        self.theta: torch.Tensor | None = (torch.ones(self.num_neurons) * self.theta_base) # TODO: + noise? self.theta[i] is the ith neuron's threshold for firing.
        
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
    
    def forward(self, x_exc: torch.Tensor, x_inh: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.g_e is None or self.g_i is None or self.theta is None:
            raise RuntimeError("Call reset() before running the forward pass.")
        
        # Update conductances.
        self.g_e = (self.g_e * self.decay_ge) + x_exc
        self.g_i = (self.g_i * self.decay_gi) + x_inh
        
        # Update voltages.
        dV: torch.Tensor = (self.E_rest - self.v) + (self.g_e * (self.E_exc - self.v)) + (self.g_i * (self.E_inh - self.v))
        dV /= self.tau_m
        self.v += dV
        
        # Check which neurons have fired.
        spikes: torch.Tensor = (self.v >= self.theta).float()
        
        # --- STRICT WINNER-TAKES-ALL (WTA) FILTER ---
        # PyTorch frame-rate tie-breaker: only the single highest voltage is allowed to fire.
        # Find the index of the neuron with the highest voltage for each image in the batch
        max_v_indices = torch.argmax(self.v, dim=1)
        
        # Create a blank mask and put a 1.0 only at the winning index
        wta_mask = torch.zeros_like(spikes)
        wta_mask.scatter_(1, max_v_indices.unsqueeze(1), 1.0)
        
        # Filter the spikes: you must have crossed the threshold AND won the WTA race
        spikes = spikes * wta_mask
        # --------------------------------------------
        
        # Reset the neurons which have spiked, back to E_rest voltage
        self.v[spikes == 1.0] = self.E_rest
        
        # Adaptive thresholding (simulates homeostasis)
        self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
        # TODO: Decide on mean or sum
        self.theta = self.theta + (spikes.sum(dim=0) * self.theta_plus)
        # self.theta.clamp_(max=-20.0)
        
        return spikes

# Architecture of the entire Diehl and Cook network
# STDP is also performed within this class
class DiehlAndCookNetwork(nn.Module):
    def __init__(self):
        super(DiehlAndCookNetwork, self).__init__()
        
        self.num_neurons: int = 400
        
        self.synapses: nn.Linear = nn.Linear(784, self.num_neurons, bias=False) # connect 784 pixels to 100 excitatory neurons.
        self.synapses.weight.requires_grad = False # Using STDP instead
        
        # STDP parameters
        self.w_max:    float = 1.0    # weight upper limit
        self.lr_plus:  float = 0.01  # Learning rate for potentiation
        self.lr_minus: float = 0.0001 # Learning rate for depression
        
        # Trace decay multipliers
        self.decay_x: float = math.exp(-1.0 / 20.0) # Pre-synaptic trace decay
        self.decay_y: float = math.exp(-1.0 / 20.0) # Post-synaptic trace decay
        
        # Initialize weights uniformly between a and b
        nn.init.uniform_(self.synapses.weight, a=0.01, b=0.3) 
        
        self.neurons: excitatory_neurons = excitatory_neurons()
        
        # LATERAL INHIBITION USING A 100 x 100 MATRIX
        # A penalty of 120mV
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * 120.0 # TODO: negative or positive? 120 is too much?
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
        self.inh_matrix = self.inh_matrix.to(device)
        self.prev_spikes = torch.zeros(batch_size, self.num_neurons, device=device)
        self.x_trace = torch.zeros(batch_size, 784, device=device)
        self.y_trace = torch.zeros(batch_size, self.num_neurons, device=device)
    
    def forward(self, input_spikes: torch.Tensor, learning: bool = True) -> torch.Tensor:
        # input spikes shape: (batch_size, 784)
        # Calculate excitatory current
        x_exc: torch.Tensor = self.synapses(input_spikes)
        
        # Calculate inhibitory current (penalties from the previous timestep)
        # Inhibit the neurons that spiked previously
        x_inh: torch.Tensor = torch.matmul(self.prev_spikes, self.inh_matrix)
        
        # Advance time
        current_spikes: torch.Tensor = self.neurons(x_exc, x_inh)
        
        # STDP
        if learning:
            batch_size = input_spikes.shape[0]
            # Update the traces
            self.x_trace = (self.x_trace * self.decay_x) + input_spikes
            self.y_trace = (self.y_trace * self.decay_y) + current_spikes
            
            # When output fired, look back at input trace (potentiation)
            # TODO: divide by batch_size?
            post_interaction: torch.Tensor = torch.matmul(current_spikes.t(), self.x_trace)
            delta_w_plus: torch.Tensor = self.lr_plus * post_interaction * (self.w_max - self.synapses.weight)
            
            # When input fired, look back at output trace (depression)
            pre_interaction = torch.matmul(self.y_trace.t(), input_spikes)
            delta_w_minus = self.lr_minus * pre_interaction * self.synapses.weight
            
            # Apply changes to synapses
            self.synapses.weight.add_(delta_w_plus - delta_w_minus) # delta_w = η(x_pre − x_tar)(w_max − w)^μ
            self.synapses.weight.clamp_(min=0.0, max=self.w_max)
            # BindsNET weight normalization
            weight_sum = self.synapses.weight.sum(dim=1, keepdim=True)
            self.synapses.weight.mul_(78.4 / weight_sum)
            
        self.prev_spikes = current_spikes
        return current_spikes
    


# 1. Setup Device and Model
if torch.backends.mps.is_available():
    device: torch.device = torch.device("mps")
    print("Using Apple Silicon MPS for acceleration.")
elif torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
    print("Using NVIDIA CUDA for acceleration.")
else:
    device: torch.device = torch.device("cpu")
    print("Using CPU.")

model: DiehlAndCookNetwork = DiehlAndCookNetwork().to(device)

num_epochs: int = 5  # In STDP, one pass over the dataset is often enough to see templates form
num_steps: int = 350

print(f"Starting unsupervised training on {device}...")
model.train() # Set to train mode (though it doesn't change much for our custom STDP)

for epoch in range(num_epochs):
    
    data: Float[torch.Tensor, "batch channels height width"]
    targets: Int[torch.Tensor, "batch"]
    for batch_idx, (data, targets) in enumerate(train_loader):
        batch_size: int = data.shape[0]
        data = data.to(device)
        
        # 2. Generate Poisson spike trains (Simulate the Retina)
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        # Flatten spatial dimensions: shape becomes [num_steps, batch_size, 784]
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        # 3. Reset the network memory for the new batch
        model.reset(batch_size, device)
        
        # 4. The Time Loop (Simulate 25 milliseconds of biological time)
        for t in range(num_steps):
            # Extract the spikes for this specific millisecond: shape [batch_size, 784]
            current_input_spikes: torch.Tensor = spike_data[t]
            
            # Forward pass (STDP weight updates happen automatically inside!)
            out_spikes: torch.Tensor = model(current_input_spikes)
            # out_spikes shape is [batch_size, 100]
            # Sum across the batch to see how many times each neuron fired this millisecond
            # spike_counts = out_spikes.sum(dim=0)

            # # Count how many unique neurons fired at least once
            # active_neurons = (spike_counts > 0).sum().item()

            # if active_neurons == 1:
            #     print("Warning: Only 1 neuron is dominating this millisecond!")
            
        # Optional: Print progress
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} processed.")

print("Unsupervised STDP training complete!")



def visualize_learned_templates(model: DiehlAndCookNetwork) -> None:
    print("Extracting learned weights...")
    
    # 1. Pull the weights from the GPU back to the CPU and convert to a NumPy array
    # Shape is (100, 784)
    weights: np.ndarray = model.synapses.weight.detach().cpu().numpy()
    
    # 2. Reshape the flat 784-pixel arrays back into 28x28 images
    # Shape becomes (100, 28, 28)
    weight_images: np.ndarray = weights.reshape(model.num_neurons, 28, 28)
    
    # 3. Set up a 10x10 grid for matplotlib
    fig, axes = plt.subplots(40, 40, figsize=(32, 32))
    fig.suptitle(f"Unsupervised STDP Templates ({model.num_neurons} Neurons)", fontsize=20, y=0.95)
    
    # Remove the spacing between images for a cleaner look
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # 4. Loop through all 100 neurons and plot their weight image
    for i, ax in enumerate(axes.flat):
        # We use the 'hot' colormap to make high weights look like bright glowing embers
        ax.imshow(weight_images[i], cmap='hot', interpolation='nearest')
        # Hide the x and y axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()

# # Run the visualization
# visualize_learned_templates(model)


@torch.no_grad()
def assign_neuron_labels(model: DiehlAndCookNetwork, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    print("Assigning class labels to neurons...")
    model.eval()
    
    # Create a tensor to count spikes: shape [100 neurons, 10 digit classes]
    spike_counts: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    # Loop through the training set one batch at a time
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # Generate the 350ms presentation spikes
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        
        # Run the time loop with learning DISABLED
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False) # shape: [batch_size, 100]
            
            # For each image in the batch, map the spikes to its target label
            # This is a fast vectorized way to accumulate the spikes per digit class
            for digit in range(10):
                mask = (targets == digit)
                if mask.any():
                    # Divide by the number of images in the mask to get the average
                    spike_counts[:, digit] += out_spikes[mask].sum(dim=0) / mask.sum().float()
                    
    # Find which digit caused each neuron to fire the most
    # neuron_assignments[i] will hold the integer digit (0-9) for the i-th neuron
    neuron_assignments: torch.Tensor = torch.argmax(spike_counts, dim=1)
    
    print("Neuron assignments complete!")
    for digit in range(10):
        count = (neuron_assignments == digit).sum().item()
        print(f"Digit {digit} was assigned to {count} neurons.")
        
    return neuron_assignments

@torch.no_grad()
def evaluate_network(model: DiehlAndCookNetwork, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int) -> float:
    print("Evaluating network accuracy on test set...")
    model.eval()
    
    correct_predictions: int = 0
    total_predictions: int = 0
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        model.reset(batch_size, device)
        
        # Track the total spikes fired by each neuron over the 350ms window
        batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
        for t in range(num_steps):
            out_spikes: torch.Tensor = model(spike_data[t], learning=False)
            batch_neuron_spikes += out_spikes
            
        # Classify each image in the batch
        for i in range(batch_size):
            # Create a 10-element array to accumulate votes for digits 0-9
            class_votes: torch.Tensor = torch.zeros(10, device=device)
            
            # Map each neuron's spike total to its assigned digit class
            for neuron_idx in range(model.num_neurons):
                assigned_digit = neuron_assignments[neuron_idx].item()
                class_votes[assigned_digit] += batch_neuron_spikes[i, neuron_idx] # type: ignore
                
            # The network's final guess is the class with the highest total spikes
            prediction = torch.argmax(class_votes).item()
            
            if prediction == targets[i].item():
                correct_predictions += 1
            total_predictions += 1
            
    accuracy: float = (correct_predictions / total_predictions) * 100.0
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 1. Run the assignment phase on the training set
neuron_labels = assign_neuron_labels(model, train_loader, device, num_steps)

# 2. Evaluate the performance on the test set
test_accuracy = evaluate_network(model, test_loader, neuron_labels, device, num_steps)
print(test_accuracy)