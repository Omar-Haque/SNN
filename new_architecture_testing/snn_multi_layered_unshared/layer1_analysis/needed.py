import numpy as np
from l1lib import *
run = Run("runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133")
c = np.load("cache/census.npz"); st, wa = c["st"], c["win_active"]
X, _ = load_mnist("test", 400)
D = np.einsum("npk,pfk->npf", 0.1 * X[:, run.idx], run.W1); gap = run.th1[None] - E_REST
with np.errstate(divide="ignore", invalid="ignore"):
    tc = np.where(100 * D > gap, 100 * np.log(100 * D / (100 * D - gap)), np.inf)
dead = (st < 0.4) & (wa[:, None] >= 0.2)
extra = []; impossible = 0; total = 0
for p in np.where(dead.any(1))[0]:
    for n in range(len(X)):
        if np.isinf(tc[n, p]).all(): continue                         # the window does not fire on this image
        for f in np.where(dead[p])[0]:
            total += 1
            if np.isinf(tc[n, p, f]): impossible += 1; continue        # the dead unit's drive cannot reach its threshold at all
            faster = np.where(tc[n, p] < tc[n, p, f])[0]
            if len(faster) == 0: extra.append(0.0); continue
            t = tc[n, p, f]                                            # the siblings must not cross before t
            need = E_REST + 100 * D[n, p, faster] * (1 - np.exp(-t / 100)) - run.th1[p, faster]
            extra.append(float(need.max()))
extra = np.array(extra)
print(f"(dead unit, image) pairs where the window fires: {total}; dead unit cannot reach threshold at all: {impossible / total * 100:.0f}%")
print(f"where it can: extra threshold the faster siblings would need: median {np.median(extra):.1f} mV (p25 {np.percentile(extra, 25):.1f}, p75 {np.percentile(extra, 75):.1f}); "
      f"current offset of structured siblings median {np.median((run.th1 + 52)[(st >= 0.4) & (wa[:, None] >= 0.2)]):.1f} mV")
np.savez("cache/needed.npz", extra=extra, impossible=impossible / total)
