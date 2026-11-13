"""
snn_theta_plus_decay.py -- thin wrapper around snn_exp.py that exposes the two threshold knobs directly.
snn_exp.py itself is NOT modified; put this file next to it.

    theta_plus   : increment of the adaptive threshold per output spike  (theta_div is fixed to 1 here, so this is
                   the REAL per-spike increment; the original script's 0.05/16 corresponds to theta_plus=0.003125)
    theta_decay  : the multiplier in   theta <- theta_base + (theta - theta_base) * theta_decay   (every step, learning on)
                   original = exp(-1/1e7) ~= 0.9999999; smaller value = faster decay.

Steady state (rate r spikes/step/neuron, batch B, lam = 1 - theta_decay):
        theta - theta_base  ~=  r * B * theta_plus / lam
so the hypothesis is that behaviour depends on  K = theta_plus / (1 - theta_decay)  (the plus : decay ratio) and not
on the two numbers separately.
"""
import argparse, json
import numpy as np
import torch
import snn_exp as snn

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--results_dir", default="results_theta_plus_decay")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_neurons", type=int, default=100)
p.add_argument("--theta_plus", type=float, default=0.05, help="per-spike threshold increment (theta_div = 1)")
p.add_argument("--theta_decay", type=float, default=0.9999999, help="per-step multiplier of (theta - theta_base)")
p.add_argument("--inh", type=float, default=200.0)
p.add_argument("--num_steps", type=int, default=350)
p.add_argument("--intensity", type=float, default=0.1)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--max_train", type=int, default=None)
p.add_argument("--snapshot_interval", type=int, default=50)
p.add_argument("--assign_n", type=int, default=2000)
p.add_argument("--eval_n", type=int, default=1000)
a = p.parse_args()

lam = 1.0 - a.theta_decay
cfg = vars(a) | {"theta_div": 1.0, "lam": lam, "K": a.theta_plus / lam if lam > 0 else float("inf")}

snn.output_dir = snn.script_dir / a.results_dir / a.name
snn.output_dir.mkdir(parents=True, exist_ok=True)
(snn.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))

snn.ENC.update(encoding="rate", intensity=a.intensity)
torch.manual_seed(a.seed); np.random.seed(a.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
loaders = snn.build_loaders(a.batch_size, a.assign_n, a.eval_n)
model = snn.DiehlAndCookNetwork(num_neurons=a.n_neurons, theta_plus=a.theta_plus, theta_div=1.0, inh=a.inh).to(device)
model.neurons.theta_decay = a.theta_decay   # <-- the variable used in: theta = base + (theta - base) * self.theta_decay
model.train()
snn.train(model, loaders[0], loaders[2], loaders[3], loaders[1], device,
          num_epochs=a.epochs, num_steps=a.num_steps, track_accuracy_curve=True, track_dominant_neuron=False,
          track_cooper=True, snapshot_interval=a.snapshot_interval, max_train_samples=a.max_train, config=cfg)
