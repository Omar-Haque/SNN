"""
dc_model.py -- the Diehl & Cook network used by the STDP demo: the LIF cell (excitatory_neurons) and the layer
(DiehlAndCookNetwork), extracted VERBATIM from new_architecture_testing/test4/test4.py (which inherited them unchanged
from hypo_1.py / the replicate file), minus the imprint and reserve initialisation branches. Nothing about the
dynamics is changed: 200 mV lateral inhibition, hard WTA off, theta_plus applied as sum(batch)/16, STDP with batch-mean
reduction, clamp to [0, 1], row normalisation to 78.4.
"""
import math
import torch, torch.nn as nn
import numpy as np


class excitatory_neurons(nn.Module):
    def __init__(self, theta_plus: float = 0.05, theta_decay_tau_steps: float = 1e7, theta_max: float | None = None,
                 num_neurons: int = 400):   # test4: network size
        # sweep: theta_plus is a constructor argument (baseline 0.05).
        # test2: theta_decay_tau_steps (baseline 1e7 batch-timesteps, i.e. effectively no decay) and theta_max (baseline
        #        None = no clamp) are constructor arguments too.
        super(excitatory_neurons, self).__init__()
        self.num_neurons: int = num_neurons  # number of excitatory neurons. (test4: constructor argument, baseline 400)
        
        # For the adaptive threshold (in mV)
        self.theta_base:  float = -52.0                # The biological firing threshold
        # CURRENT: Changed theta_plus changed from 0.5 to 0.05
        self.theta_plus:  float = theta_plus            # prev: 0.1. a penalty added to the minimum fire threshold each time a neuron fires.
        # sweep: baseline value 0.05; set from the command line (--theta-plus).
        self.theta_decay: float = math.exp(-1.0 / theta_decay_tau_steps)   # test2: baseline 1e7; --theta-tau sets it in examples
        self.theta_max: float | None = theta_max                            # test2: --theta-max; None = the baseline's no-clamp
        
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
            if self.theta_max is not None:          # test2: the base file's commented-out clamp, enabled by --theta-max
                self.theta.clamp_(max=self.theta_max)

        return spikes


class DiehlAndCookNetwork(nn.Module):
    def __init__(self, theta_plus: float = 0.05, inhibition_mV: float = 200.0, init: str = "uniform",
                 imprint_dataset=None, rng: np.random.Generator | None = None, theta_init: float | None = None,
                 theta_decay_tau_steps: float = 1e7, theta_max: float | None = None,   # test2: decay constant and clamp
                 reserve_idx: list[int] | None = None,                                    # test3: neurons that start flat
                 num_neurons: int = 400):                                                 # test4: network size
        # sweep: the three architecture knobs (theta_plus, inhibition_mV, init) and the optional theta_init are
        # constructor arguments; the defaults reproduce hypo_1.py's baseline exactly.
        super(DiehlAndCookNetwork, self).__init__()
        
        self.num_neurons: int = num_neurons   # test4: constructor argument (baseline 400)
        
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
            # sweep: "imprint at zero". Every row becomes a random training image (784 pixel intensities) scaled to sum
            # 78.4, exactly what hypo_1's imprint intervention did to the reservoir at 100k, but for ALL 400 neurons at
            # t = 0 and with theta left at theta_base. Labels are not used. No flat template exists afterwards, so the
            # reservoir is empty from the start and there is nothing to hijack. Images are drawn without replacement
            # with the run's numpy generator (same --seed => same templates).
            if init != "uniform":
                raise ValueError("the STDP demo trains from the uniform init only")
        
        self.neurons: excitatory_neurons = excitatory_neurons(theta_plus, theta_decay_tau_steps, theta_max, num_neurons)
        if theta_init is not None:
            # sweep: start every adaptive threshold at theta_init mV instead of theta_base ("sabotage at zero").
            self.neurons.theta = torch.ones(self.num_neurons) * float(theta_init)
        
        # LATERAL INHIBITION USING A 100 x 100 MATRIX
        # A penalty of 200mV
        self.inh_matrix: torch.Tensor = torch.ones(self.num_neurons, self.num_neurons) * inhibition_mV   # sweep: baseline 200.0 mV, set by --inhibition
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
