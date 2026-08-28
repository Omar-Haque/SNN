import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import math
import matplotlib.pyplot as plt
import numpy as np
# type annotations everywhere because I remember messing up one of my RL projects because of confusing tensors and numpy arrays.
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

# Load the MNIST dataset
train_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader:  DataLoader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Small subsets of MNIST for running average
# Use 2000 images to assign labels and 1000 images to test
assign_subset = torch.utils.data.Subset(train_dataset, range(0, 2000))
eval_subset   = torch.utils.data.Subset(test_dataset, range(0, 1000))
snapshot_assign_loader = DataLoader(assign_subset, batch_size=256, shuffle=False)
snapshot_eval_loader   = DataLoader(eval_subset, batch_size=256, shuffle=False)

# EXCITATORY NEURONS
# Excitatory neurons are an entire class, but inhibitory neurons can be simulated using matrices.
class excitatory_neurons(nn.Module):
    def __init__(self):
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = 400  # number of excitatory neurons.
        
        # For the adaptive threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        self.theta_plus:  float = 0.5                  # prev: 0.1. a penalty added to the minimum fire threshold each time a neuron fires.
        self.theta_decay: float = math.exp(-1.0 / 1e5) # prev: 1e5 106 unassigned
        
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
        wta_mask = torch.zeros_like(spikes)
        wta_mask.scatter_(1, max_v_indices.unsqueeze(1), 1.0)
        
        # Must have crossed the threshold and won the argmax wta.
        spikes = spikes * wta_mask 
        
        # Reset the neurons which have spiked, back to E_rest voltage
        self.v[spikes == 1.0] = self.E_rest
        
        # Adaptive thresholding (simulates homeostasis)
        if learning:
            self.theta = self.theta_base + (self.theta - self.theta_base) * self.theta_decay
            self.theta = self.theta + (spikes.mean(dim=0) * self.theta_plus)
            # For biological accuracy.
            self.theta.clamp_(max=0.0)
            # if (self.theta > -8.0).any():
            #     num_dead = (self.theta > -8.0).sum().item()
            #     max_theta = self.theta.max().item()
            #     print(f"{num_dead} neurons crossed -8.0mV. (Highest: {max_theta:.2f}mV)")
        
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
        self.w_max:    float = 1.0    # weight upper limit
        self.lr_plus:  float = 0.0003   # Learning rate for potentiation 0.01
        self.lr_minus: float = 0.000003 # Learning rate for depression 0.0001
        
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
        # A penalty of 120mV
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * 120.0
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
        
        # current_spikes has shape (batch_size, num_neurons)
        # Row i represents the ith image in our batch
        # Column j represents the jth excitatory neuron
        # current_spikes[i][j] has a value of either 1.0 or 0.0
        # 1.0 indicates neuron j successfully fired at millisecond t (see training and assignment loop) while looking at image i.
        # 0.0 indicates neuron j did not fire when looking at image i.
    
def visualize_learned_templates(model: DiehlAndCookNetwork) -> None:
    print("Extracting weights and thresholds after training...")
    
    # Shape is (400, 784)
    weights: np.ndarray = model.synapses.weight.detach().cpu().numpy()
    
    # Shape becomes (400, 28, 28)
    weight_images: np.ndarray = weights.reshape(model.num_neurons, 28, 28)
    
    # Extract the current adaptive thresholds (theta) for all 400 neurons
    thetas: np.ndarray = model.neurons.theta.detach().cpu().numpy() # type: ignore
    
    # Set up a 20x20 grid for matplotlib
    # Increased figsize and hspace to make room for the text titles
    fig, axes = plt.subplots(20, 20, figsize=(18, 18))
    fig.suptitle(f"Unsupervised STDP Templates & Thresholds ({model.num_neurons} Neurons)", fontsize=20, y=0.95)
    plt.subplots_adjust(wspace=0.05, hspace=0.4) 
    
    # Loop through all neurons and plot their weight image and theta value
    for i, ax in enumerate(axes.flat):
        # We use the 'hot' colormap to make high weights look like bright glowing embers
        ax.imshow(weight_images[i], cmap='hot', interpolation='nearest')
        
        # Add the theta value directly above the neuron
        ax.set_title(f"θ: {thetas[i]:.1f}", fontsize=8, pad=2)
        
        # Hide the x and y axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    
@torch.no_grad()
def assign_neuron_labels(model: DiehlAndCookNetwork, data_loader: DataLoader, device: torch.device, num_steps: int) -> torch.Tensor:
    print("Assigning class labels to neurons")
    model.eval()
    
    # Create a tensor to count spikes: shape [num_neurons neurons, 10 digit classes]
    spike_counts: torch.Tensor = torch.zeros(model.num_neurons, 10, device=device)
    
    for data, targets in data_loader:
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # TODO: Is this helpful?
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
    for digit in range(10):
        count = (neuron_assignments == digit).sum().item()
        print(f"Digit {digit} was assigned to {count} neurons.")
        
    dead_count = (neuron_assignments == -1).sum().item()
    print(f"Unassigned neurons: {dead_count}")
        
    return neuron_assignments

@torch.no_grad()
def evaluate_network(model: DiehlAndCookNetwork, data_loader: DataLoader, neuron_assignments: torch.Tensor, device: torch.device, num_steps: int) -> float:
    print("Evaluating network accuracy on test set...")
    model.eval()
    
    correct_predictions: int = 0
    total_predictions:   int = 0
    
    # Slow loop
    # for data, targets in data_loader:
    #     batch_size: int = data.shape[0]
    #     data, targets = data.to(device), targets.to(device)
        
    #     spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
    #     spike_data = spike_data.view(num_steps, batch_size, 784)
        
    #     model.reset(batch_size, device)
        
    #     # Track the total spikes fired by each neuron over the 350ms window
    #     batch_neuron_spikes: torch.Tensor = torch.zeros(batch_size, model.num_neurons, device=device)
        
    #     for t in range(num_steps):
    #         out_spikes: torch.Tensor = model(spike_data[t], learning=False)
    #         batch_neuron_spikes += out_spikes
            
    #     # Classify each image in the batch
    #     for i in range(batch_size):
    #         # Create a 10-element array to accumulate votes for digits 0-9
    #         class_votes: torch.Tensor = torch.zeros(10, device=device)
            
    #         # Map each neuron's spike total to its assigned digit class
    #         for neuron_idx in range(model.num_neurons):
    #             assigned_digit = neuron_assignments[neuron_idx].item()
    #             if assigned_digit != -1:
    #                 class_votes[assigned_digit] += batch_neuron_spikes[i, neuron_idx] # type: ignore
                
    #         # The network's final guess is the class with the highest total spikes
    #         prediction = torch.argmax(class_votes).item()
            
    #         if prediction == targets[i].item():
    #             correct_predictions += 1
    #         total_predictions += 1
            
    # accuracy: float = (correct_predictions / total_predictions) * 100.0
    # print(f"Final Test Accuracy: {accuracy:.2f}%")
    # return accuracy
    
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
        
        # TODO: Is this helpful?
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
        
        # Instantly count how many predictions match the targets
        correct_predictions += (predictions == targets).sum().item()
        total_predictions += batch_size
        
    return (correct_predictions / total_predictions) * 100.0
    


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

num_epochs: int = 3
num_steps: int = 350

model.train() # This probably doesn't make a difference.

# Tracking variables for running accuracy
total_points_processed = 0
points_history = []
accuracy_history = []
snapshot_interval = 100 # Run an evaluation every 100 batches

# Training loop
for epoch in range(num_epochs):
    data: Float[torch.Tensor, "batch channels height width"]
    targets: Int[torch.Tensor, "batch"]
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        batch_size: int = data.shape[0]
        data, targets = data.to(device), targets.to(device)
        
        # TODO: Is this helpful?
        # Find the total sum of white pixels for each image in the batch
        # image_sums = data.sum(dim=(1, 2, 3), keepdim=True) + 1e-5
        # Force every image to have the exact same total sum of 75.0
        # data = (data / image_sums) * 75.0
        
        # Generate poisson spike trains
        spike_data: torch.Tensor = spikegen.rate(data * 0.1, num_steps=num_steps) # type: ignore
        # Flatten spatial dimensions: shape becomes [num_steps, batch_size, 784]
        spike_data = spike_data.view(num_steps, batch_size, 784)
        
        # Reset the network memory for a new batch
        model.reset(batch_size, device)
        
        # Simulate num_steps milliseconds of biological time
        for t in range(num_steps):
            # Extract the spikes for this specific millisecond: shape [batch_size, 784]
            current_input_spikes: torch.Tensor = spike_data[t]
            
            # Forward pass with those extracted spikes.
            out_spikes: torch.Tensor = model(current_input_spikes)
        
        total_points_processed += batch_size
        
        # # Estimated accuracy evaluation
        # if batch_idx % snapshot_interval == 0:
        #     print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Estimating current accuracy...")

        #     temp_labels = assign_neuron_labels(model, snapshot_assign_loader, device, num_steps)
        #     snapshot_acc = evaluate_network(model, snapshot_eval_loader, temp_labels, device, num_steps)
        #     points_history.append(total_points_processed)
        #     accuracy_history.append(snapshot_acc)
        #     print(snapshot_acc)
        
        # if batch_idx % 100 == 0:
        #     print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} processed.")

print("Training complete")

# --- PLOT THE TRUE TRAINING CURVE ---
plt.figure(figsize=(10, 6))
plt.plot(points_history, accuracy_history, marker='o', linestyle='-', color='blue', linewidth=2)
plt.title(f"True Training Accuracy Curve ({model.num_neurons} Neurons)")
plt.xlabel("Number of Training Points Processed")
plt.ylabel("Accuracy (%) on Validation Subset")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

visualize_learned_templates(model)
neuron_labels = assign_neuron_labels(model, train_loader, device, num_steps)
test_accuracy = evaluate_network(model, test_loader, neuron_labels, device, num_steps)
print(test_accuracy)