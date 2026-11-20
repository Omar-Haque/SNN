"""drive_data.py -- data for the drive figure: for each trained network, per test image, the input spikes the winning
layer-2 neuron had received when it first fired; and one image's membrane trace of the winner."""
import json, numpy as np
import l2lib  # noqa: F401
from l1lib import Run, load_mnist
from l2lib import Net

RUNS = {"single": "singlelayer_n400_tp0.5_tau30k_ep6_seed0_l2wta_20261111-063717",
        "drv1.2": "unshared_p7s3f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_20261111-063618",
        "drv0.6_lr6": "unshared_p7s3f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_drv0.6_lr6_20261113-165810",
        "drv0.36_lr10": "unshared_p7s3f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_drv0.36_lr10_20261113-165859"}
X, Y = load_mnist("test", 500); out = {}
for key, name in RUNS.items():
    run = Run(name); net = Net(run, W2=run.W2); net.theta = run.th2.copy()
    rng = np.random.default_rng(2); res = [net.present(X[i:i + 125], learn=False, rng=rng, probe=True) for i in range(0, 500, 125)]
    c2 = np.concatenate([r[0] for r in res]); c1 = np.concatenate([r[1] for r in res]); tf = np.concatenate([r[2] for r in res]); cf = np.concatenate([r[3] for r in res])
    ok = tf >= 0; win = c2.argmax(1)
    # mV delivered to the winner per input spike it had received before firing
    mv_per = ((cf * net.W[win]).sum(1) / np.maximum(1, cf.sum(1)))
    # trace of the same image (test image 3, a '0') with a fixed seed
    _, v, zin, s, th = net.present(X[3:4], learn=False, rng=np.random.default_rng(11), trace=True)
    w = int(s.any(0).argmax()) if s.any() else int(v.max(0).argmax())
    first = int(np.argmax(s[:, w])) if s[:, w].any() else -1
    out[key] = dict(spikes_before=cf.sum(1)[ok].tolist(), frac=((cf > 0).sum(1) / np.maximum(1, (c1 > 0).sum(1)))[ok].tolist(),
                    t_first=tf[ok].tolist(), mv_per_spike=mv_per[ok].tolist(), trace_v=v[:, w].tolist(), trace_in=zin.tolist(),
                    trace_spikes=np.where(s[:, w])[0].tolist(), trace_theta=float(th[w]), trace_first=first, mean_weight=float(net.W.mean()))
    print(key, f"median spikes before {np.median(cf.sum(1)[ok]):.0f}, mV per spike {np.median(mv_per[ok]):.1f}, first spike {np.median(tf[ok]):.0f} ms; trace first spike {first} ms", flush=True)
json.dump(out, open("cache/drive_data.json", "w"))
