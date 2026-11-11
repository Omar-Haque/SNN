import numpy as np, json, time
from l1lib import *
run = Run("runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133")
Xtr, Ytr = load_mnist("train", 5000)
t0 = time.time()
cnt, first = layer1_codes(run, Xtr, seed=11, record_first=True)        # (N,P,F)
print(f"replay {time.time()-t0:.0f}s")
P, F = run.P, run.F
fired = cnt > 0
win_frac = fired.mean(0)                                  # (P,F) fraction of images each neuron fired in
win_active = fired.any(2).mean(0)                         # (P,) fraction of images the window fired at all
st = structure(run.W1.reshape(-1, 49)).reshape(P, F)
ring = ring_of(run.side)
# ink: expected input spikes per window per presentation (0.1/ms x 350 ms x ink mass)
ink_mass = Xtr[:, run.idx].sum(2)                         # (N,P)
ink_any = (Xtr[:, run.idx] > 0.25).sum(2) >= 3            # >= 3 inked pixels
np.savez("cache/census.npz", win_frac=win_frac, win_active=win_active, st=st, ring=ring, ink_any=ink_any.mean(0), ink_mass=ink_mass.mean(0),
         first_med=np.nanmedian(np.where(np.isinf(first), np.nan, first).reshape(len(Xtr), P, F).min(2) if False else np.where(np.isinf(first.min(2)), np.nan, first.min(2)), axis=0))
frozen = run.cfg["l1"]["frozen_at"]
est_wins = win_frac * frozen
print("structured (>=0.4) per ring:", {r: f"{(st[ring == r] >= 0.4).sum()}/{(ring == r).sum() * F}" for r in range(4)})
print("window fires on (fraction of images) per ring: ", {r: f"{np.median(win_active[ring == r]):.2f} (min {win_active[ring == r].min():.2f})" for r in range(4)})
print("windows with >=3 inked px per ring:", {r: f"{np.median(ink_any.mean(0)[ring == r]):.2f}" for r in range(4)})
for thr in [0, 10, 30, 100, 300, 1000]:
    m = est_wins.ravel() <= thr if thr == 0 else (est_wins.ravel() > thr / 3) & (est_wins.ravel() <= thr)
for lo, hi in [(0, 0.5), (0.5, 10), (10, 30), (30, 100), (100, 300), (300, 1000), (1000, 1e9)]:
    m = (est_wins.ravel() >= lo) & (est_wins.ravel() < hi)
    if m.any(): print(f"  est. lifetime wins [{lo:>6g},{hi:>6g}): {m.sum():4d} neurons, structured {np.mean(st.ravel()[m] >= 0.4) * 100:5.1f}%, median structure {np.median(st.ravel()[m]):.2f}")
# dead units: unstructured neurons in windows that are active often
dead = (st < 0.4) & (win_active[:, None] > 0.2)
print("unstructured neurons in windows active on >20% of images:", dead.sum(), "| of which never fired in 5000 images:", (dead & (win_frac == 0)).sum())
print("neurons that never fired in 5000 images:", (win_frac == 0).sum(), "| of those unstructured:", ((win_frac == 0) & (st < 0.4)).sum())
