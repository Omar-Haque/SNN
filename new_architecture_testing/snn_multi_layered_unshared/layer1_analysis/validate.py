import sys, time, numpy as np
from l1lib import *
name = sys.argv[1]
run = Run(f"runs/{name}")
Xtr, Ytr = load_mnist("train", 2000); Xte, Yte = load_mnist("test", 1000)
t0 = time.time()
c_tr, _ = run_network(run, Xtr, seed=1); lab = assign_labels(c_tr, Ytr)
c_te, l1 = run_network(run, Xte, seed=2); acc, _ = evaluate(c_te, Yte, lab)
c_sh, _ = run_network(run, shift_images(Xte), seed=3); acc_s, _ = evaluate(c_sh, Yte, lab)
print(f"{name[:60]:60s} numpy replay: {acc:.1f}% (shifted {acc_s:.1f}%) | run's last snapshot: {run.d['accuracy'][-1]:.1f}% (shifted {run.d['accuracy_shifted'][-1]:.1f}%) | "
      f"L2 spikes/img {c_te.sum(1).mean():.1f} | {time.time() - t0:.0f}s")
