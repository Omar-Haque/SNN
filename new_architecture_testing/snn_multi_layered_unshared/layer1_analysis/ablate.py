import numpy as np, time
from l1lib import *
run = Run("runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133")
c = np.load("cache/census.npz"); st = c["st"].ravel(); ring = np.repeat(c["ring"], run.F)
Xtr, Ytr = load_mnist("train", 2000); Xte, Yte = load_mnist("test", 1000)
rng = np.random.default_rng(5)
unstr = st < 0.4
rand_struct = np.zeros(len(st), bool); rand_struct[rng.choice(np.where(~unstr)[0], unstr.sum(), replace=False)] = True
conds = {"intact": np.ones(len(st)),
         "hide the 347 unstructured channels": (~unstr).astype(float),
         "hide 347 random STRUCTURED channels": (~rand_struct).astype(float),
         "hide the outer ring (448 channels)": (ring > 0).astype(float),
         "show ONLY the unstructured channels": unstr.astype(float)}
res = {}
for name, m in conds.items():
    t0 = time.time()
    ctr, l1tr = run_network(run, Xtr, seed=1, mask=m); lab = assign_labels(ctr, Ytr)
    cte, l1te = run_network(run, Xte, seed=2, mask=m); acc, _ = evaluate(cte, Yte, lab)
    vis = (l1te * m).sum(1).mean()
    res[name] = (acc, cte.sum(1).mean(), vis)
    print(f"{name:40s} acc {acc:5.1f}% | L2 spikes/img {cte.sum(1).mean():5.1f} | L1 spikes/img visible to L2 {vis:6.1f} of {l1te.sum(1).mean():.0f} | {time.time() - t0:.0f}s", flush=True)
np.save("cache/ablate.npy", res, allow_pickle=True)
