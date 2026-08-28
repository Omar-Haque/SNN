import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import collections.abc

# --- MONKEY PATCH FOR BINDSNET COMPATIBILITY ---
# BindsNET expects ancient PyTorch variables that were removed. 
# We inject them back into PyTorch's _six module dynamically so you don't have to edit source code!
import torch._six
torch._six.container_abcs = collections.abc # type: ignore
torch._six.string_classes = (str,)
torch._six.int_classes = (int,) # type: ignore
# -----------------------------------------------

# BindsNET specific imports
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor

def main():
    # 1. Hardware & Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # type: ignore
    print(f"Using device: {device}")

    n_neurons = 1600
    n_epochs = 7
    time_steps = 350
    batch_size = 256  # BindsNET is highly memory-intensive; 64 prevents OOM crashes
    
    # BindsNET's standard input scaling (flat intensity multiplier)
    intensity = 128.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * intensity)
    ])

    # 3. Load Datasets (BindsNET handles Poisson encoding inside the dataset wrapper)
    print("Loading Datasets...")
    train_dataset = MNIST(
        PoissonEncoder(time=time_steps, dt=1.0),
        None,
        root=os.path.join("data", "MNIST"),
        download=True,
        train=True,
        transform=transform
    )

    test_dataset = MNIST(
        PoissonEncoder(time=time_steps, dt=1.0),
        None,
        root=os.path.join("data", "MNIST"),
        download=True,
        train=False,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Build the BindsNET Network
    print("Initializing BindsNET DiehlAndCook2015 Model...")
    network = DiehlAndCook2015(
        n_inpt=784,
        n_neurons=n_neurons,
        nu=(1e-4, 1e-2)  # Required: STDP learning rates
    ).to(device)

    # --- NEW: ADD MONITOR TO RECORD SPIKES ---
    exc_monitor = Monitor(network.layers["Ae"], state_vars=["s"], time=time_steps)
    network.add_monitor(exc_monitor, name="Ae")

    # --- NEW: SAVE/LOAD TOGGLE ---
    save_path = "bindsnet_1600_trained.pt"
    load_model = False # Set to True to skip training and load the saved model

    if load_model and os.path.exists(save_path):
        print(f"Loading pre-trained model from {save_path}...")
        network.load_state_dict(torch.load(save_path, map_location=device))
    else:
        # 5. Training Phase
        print("\n--- Starting Training Phase ---")
        for epoch in range(n_epochs):
            network.train(mode=True)
            for step, batch in enumerate(train_loader):
                # BindsNET expects inputs in shape: [time, batch, neurons]
                inpts = {"X": batch["encoded_image"].view(batch["encoded_image"].shape[0], time_steps, 784).permute(1, 0, 2).to(device)}
                
                network.run(inputs=inpts, time=time_steps)
                network.reset_state_variables()
                
                if step % 100 == 0:
                    print(f"Epoch {epoch} | Batch {step}/{len(train_loader)} processed.")

    # 6. Assignment Phase
    print("\n--- Starting Assignment Phase ---")
    network.train(mode=False) # Freeze STDP and homeostasis
    spike_counts = torch.zeros(n_neurons, 10, device=device)
    
    with torch.no_grad():
        for step, batch in enumerate(train_loader):
            inpts = {"X": batch["encoded_image"].view(batch["encoded_image"].shape[0], time_steps, 784).permute(1, 0, 2).to(device)}
            labels = batch["label"].to(device)
            
            network.run(inputs=inpts, time=time_steps)
            
            # Extract spikes from the excitatory layer monitor
            spikes = network.monitors["Ae"].get("s") # shape: [time, batch, n_neurons]
            spike_sum = spikes.sum(dim=0)            # shape: [batch, n_neurons]
            
            for digit in range(10):
                mask = (labels == digit)
                if mask.any():
                    spike_counts[:, digit] += spike_sum[mask].sum(dim=0) / mask.sum().float()
                    
            network.reset_state_variables()
            
            if step % 100 == 0:
                print(f"Assignment | Batch {step}/{len(train_loader)} processed.")

    neuron_assignments = torch.argmax(spike_counts, dim=1)
    dead_neurons = (spike_counts.sum(dim=1) == 0)
    neuron_assignments[dead_neurons] = -1
    
    print("\nNeuron assignments complete:")
    for digit in range(10):
        count = (neuron_assignments == digit).sum().item()
        print(f"Digit {digit} was assigned to {count} neurons.")
    print(f"Unassigned neurons: {dead_neurons.sum().item()}")

    # 7. Evaluation Phase
    print("\n--- Starting Evaluation Phase ---")
    correct = 0
    total = 0
    
    assignment_matrix = torch.zeros(n_neurons, 10, device=device)
    valid_mask = (neuron_assignments != -1)
    safe_assignments = torch.where(valid_mask, neuron_assignments, torch.zeros_like(neuron_assignments))
    assignment_matrix.scatter_(1, safe_assignments.unsqueeze(1), 1.0)
    assignment_matrix[~valid_mask] = 0.0
    
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            inpts = {"X": batch["encoded_image"].view(batch["encoded_image"].shape[0], time_steps, 784).permute(1, 0, 2).to(device)}
            labels = batch["label"].to(device)
            
            network.run(inputs=inpts, time=time_steps)
            spikes = network.monitors["Ae"].get("s")
            spike_sum = spikes.sum(dim=0).float()
            
            class_votes = torch.matmul(spike_sum, assignment_matrix)
            predictions = torch.argmax(class_votes, dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            network.reset_state_variables()

    print(f"\nFinal Test Accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
    # Ensure bindsnet is installed: pip install bindsnet
    main()