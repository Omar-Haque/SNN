"""
train_l2.py -- train layer 2 (numpy replica of stage B) on top of a finished run's frozen layer 1, with checkpoints.

  python train_l2.py --run <run folder> --tag base                       # replica of the run's own stage B
  python train_l2.py --run <run folder> --tag lr0 --lr-plus 0 --lr-minus 0     # no learning: random weights + conscience
  python train_l2.py --run <run folder> --tag jit --init jitter:0.05           # near-uniform initial weights
Works for a single-layer run too (--run singlelayer_...): the "layer 1" is then the pixels themselves.

At each checkpoint: held-out accuracy with the run's estimator (labels from train images 0-1999, test images 0-999), rows
that have not moved from their initial weights (cosine > 0.95), theta, spikes per presentation, and -- over the interval
since the last checkpoint -- how the wins were shared (per neuron, and per neuron x class: 'purity' = share of a neuron's
won presentations that belong to its most frequent class, averaged over neurons weighted by wins).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import l2lib  # noqa: F401  (puts ../layer1_analysis, where l1lib.py lives, on the path)
from l1lib import Run, load_mnist, assign_labels, evaluate, shift_images
from l2lib import Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--examples", type=int, default=60000); ap.add_argument("--eval-at", default="3200,6400,12800,32000,60000")
    ap.add_argument("--lr-plus", type=float, default=1e-2); ap.add_argument("--lr-minus", type=float, default=1e-4)
    ap.add_argument("--theta-plus", type=float, default=0.5); ap.add_argument("--theta-tau", type=float, default=30000.0)
    ap.add_argument("--init", default="run", help="run = the run's own initial weights | jitter:A = mean x uniform(1-A, 1+A) | const")
    ap.add_argument("--drive-mult", type=float, default=1.0, help="scale layer-2's input weights (row sum and w_max) by this factor: < 1 = a neuron needs more input spikes to fire")
    ap.add_argument("--wmax-mult", type=float, default=None, help="w_max as a multiple of the mean weight (run: 10)")
    ap.add_argument("--l1-inh", type=float, default=None); ap.add_argument("--inh", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--shifted", action="store_true")
    a = ap.parse_args()
    run = Run(a.run); rng = np.random.default_rng(a.seed + 77)
    n_in = 784 if run.no_l1 else run.P * run.F; w_norm = float(run.cfg["l2_norm"]) * a.drive_mult; mean_w = w_norm / n_in
    W0 = run.d["W2_init"].astype(np.float32) * a.drive_mult if "W2_init" in run.d else None
    if a.init.startswith("jitter:"):
        A = float(a.init.split(":")[1]); W0 = rng.uniform(1 - A, 1 + A, (400, n_in)).astype(np.float32); W0 *= w_norm / W0.sum(1, keepdims=True)
    elif a.init == "const":
        W0 = np.full((400, n_in), mean_w, np.float32)
    elif W0 is None:
        W0 = rng.uniform(0.01, 0.3, (400, n_in)).astype(np.float32); W0 *= w_norm / W0.sum(1, keepdims=True)
    w_max = (float(run.cfg["l2_wmax"]) * a.drive_mult) if a.wmax_mult is None else a.wmax_mult * mean_w
    net = Net(run, W2=W0, w_norm=w_norm, w_max=w_max, theta_plus=a.theta_plus, theta_tau=a.theta_tau, lr_plus=a.lr_plus, lr_minus=a.lr_minus,
              l2_inh=a.inh, l1_inh=a.l1_inh, seed=a.seed)
    X, Y = load_mnist("train"); Xa, Ya = X[:2000], Y[:2000]; Xt, Yt = load_mnist("test", 1000); Xs = shift_images(Xt) if a.shifted else None
    order = np.random.default_rng(a.seed).permutation(len(X))
    eval_at = sorted(int(e) for e in a.eval_at.split(",")); hist = []
    wins = np.zeros(400); wins_cls = np.zeros((400, 10)); spk = 0.0; npres = 0; t0 = time.time(); seen = 0
    print(f"[{a.tag}] {run.name[:50]} | init {a.init} lr+ {a.lr_plus} θ+ {a.theta_plus} τ {a.theta_tau} w_max {net.w_max:.2f} (mean {mean_w:.3f}) l1_inh {None if run.no_l1 else net.l1_inh}", flush=True)
    while seen < a.examples:
        sel = order[(seen + np.arange(32)) % len(X)]
        c = net.present(X[sel]); seen += 32
        w = c > 0; wins += w.sum(0); np.add.at(wins_cls.T, Y[sel], w.astype(float)); spk += c.sum(); npres += 32
        if eval_at and seen >= eval_at[0]:
            eval_at.pop(0)
            lab = assign_labels(net.counts(Xa, 1), Ya); ct = net.counts(Xt, 2); acc, _ = evaluate(ct, Yt, lab)
            acc_s = evaluate(net.counts(Xs, 3), Yt, lab)[0] if Xs is not None else float("nan")
            ci = net.cos_init(); pur = float((wins_cls.max(1).sum()) / max(1.0, wins_cls.sum()))
            h = dict(examples=seen, acc=acc, acc_shifted=acc_s, unmoved=int((ci > 0.95).sum()), cos_init_med=float(np.median(ci)), theta_med=float(np.median(net.theta)),
                     theta_p90=float(np.percentile(net.theta, 90)), spikes_per_pres=spk / npres, purity=pur, wins_top20_share=float(np.sort(wins)[-20:].sum() / max(1, wins.sum())),
                     neurons_winning=int((wins > 0).sum()), test_spikes=float(ct.sum(1).mean()), minutes=(time.time() - t0) / 60)
            hist.append(h)
            print(f"[{a.tag}] {seen:>6d} ex | acc {acc:5.1f}" + (f" shifted {acc_s:5.1f}" if Xs is not None else "") + f" | unmoved rows {h['unmoved']:3d} cos-init {h['cos_init_med']:.2f} | "
                  f"θ med {h['theta_med']:6.1f} p90 {h['theta_p90']:6.1f} | spikes/pres {h['spikes_per_pres']:5.1f} | win purity {pur:.2f} | {h['minutes']:5.1f} min", flush=True)
            wins[:] = 0; wins_cls[:] = 0; spk = 0.0; npres = 0
    Path("cache").mkdir(exist_ok=True)
    np.savez_compressed(f"cache/l2_{a.tag}.npz", W=net.W.astype(np.float16), W_init=net.W_init.astype(np.float16), theta=net.theta, hist=json.dumps(hist, default=float), args=json.dumps(vars(a), default=float), run=run.name)
    print(f"[{a.tag}] saved cache/l2_{a.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
