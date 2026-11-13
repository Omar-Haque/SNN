"""probe_first.py RUN -- on how much of its input has layer 2 decided? For each test image: when the first layer-2 spike
happens, how many input spikes have arrived, and from what fraction of the channels that are active over the presentation."""
import sys, json, numpy as np
import l2lib  # noqa: F401  (puts ../layer1_analysis, where l1lib.py lives, on the path)
from l1lib import Run, load_mnist
from l2lib import Net
run = Run(sys.argv[1]); net = Net(run, W2=run.W2); net.theta = run.th2.copy()
if "--weights" in sys.argv:
    f = np.load(sys.argv[sys.argv.index("--weights") + 1]); net.W = f["W"].astype(np.float32); net.theta = f["theta"].astype(np.float32)
X, Y = load_mnist("test", 512); rng = np.random.default_rng(5)
res = [net.present(X[i:i + 128], learn=False, rng=rng, probe=True) for i in range(0, 512, 128)]
c2 = np.concatenate([r[0] for r in res]); c1 = np.concatenate([r[1] for r in res]); tf = np.concatenate([r[2] for r in res]); cf = np.concatenate([r[3] for r in res])
ok = tf >= 0
act_all = (c1 > 0).sum(1); act_first = (cf > 0).sum(1); spikes_first = cf.sum(1)
winner = c2.argmax(1); W = net.W
# input spikes the winner needed: weighted by its own weights (mV received) vs its threshold distance
mv = (cf * W[winner]).sum(1)
out = {"run": run.name[:46], "first_l2_spike_ms_median": float(np.median(tf[ok])),
  "input_spikes_before_decision_median": float(np.median(spikes_first[ok])), "input_spikes_whole_presentation_median": float(np.median(c1.sum(1))),
  "channels_reported_before_decision_median": float(np.median(act_first[ok])), "channels_active_whole_presentation_median": float(np.median(act_all)),
  "fraction_of_active_channels_reported": float(np.median(act_first[ok] / act_all[ok])),
  "mV_delivered_to_winner_before_its_first_spike_median": float(np.median(mv[ok])), "winner_threshold_distance_median_mV": float(np.median((net.theta + 65)[winner][ok])),
  "largest_single_input_weight_median_mV": float(np.median(W.max(1))), "weights_above_half_wmax_per_neuron_median": float(np.median((W > 0.5 * net.w_max).sum(1)))}
print(json.dumps(out, indent=1)); open(f"cache/probe_{'single' if run.no_l1 else 'f' + str(run.F)}.json", "w").write(json.dumps(out, indent=1))
