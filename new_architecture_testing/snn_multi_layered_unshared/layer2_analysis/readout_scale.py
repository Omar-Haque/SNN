"""readout_scale.py RUN -- the trained network with layer-2's input weights scaled down at test time (thresholds unchanged):
a smaller scale means a neuron needs more input spikes before it can fire, i.e. it decides on more of the code."""
import sys, json, numpy as np
import l2lib  # noqa: F401  (puts ../layer1_analysis, where l1lib.py lives, on the path)
from l1lib import Run, load_mnist, assign_labels, evaluate
from l2lib import Net
run = Run(sys.argv[1]); Xa, Ya = load_mnist("train", 2000); Xt, Yt = load_mnist("test", 1000)
scales = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 0.5, 0.3, 0.2]
rows = []
for f in scales:
    net = Net(run, W2=run.W2 * f); net.theta = run.th2.copy()
    ca = net.counts(Xa, 1); lab = assign_labels(ca, Ya)
    rng = np.random.default_rng(2); res = [net.present(Xt[i:i + 125], learn=False, rng=rng, probe=True) for i in range(0, 1000, 125)]
    ct = np.concatenate([r[0] for r in res]); c1 = np.concatenate([r[1] for r in res]); tf = np.concatenate([r[2] for r in res]); cf = np.concatenate([r[3] for r in res])
    ok = tf >= 0; acc = evaluate(ct, Yt, lab)[0]
    r = dict(scale=f, acc=acc, silent=int((~ok).sum()), first_spike_ms=float(np.median(tf[ok])), input_spikes_before=float(np.median(cf.sum(1)[ok])),
             frac_channels=float(np.median(((cf > 0).sum(1) / np.maximum(1, (c1 > 0).sum(1)))[ok])), l2_spikes=float(ct.sum(1).mean()))
    rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(f"cache/readout_scale_{run.name[:24]}.json", "w"))
