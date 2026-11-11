import numpy as np, time
from l1lib import *
run = Run("runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133")
c = np.load("cache/census.npz"); st = c["st"]; win_active = c["win_active"]
X, Y = load_mnist("test", 400); N, P, F = len(X), run.P, run.F; R = 8
reps = np.stack([layer1_codes(run, X, seed=300 + r) for r in range(R)])             # (R,N,P,F)
# deterministic race: expected drive D (mV/ms) of every feature, expected time to cross its own threshold
D = np.einsum("npk,pfk->npf", 0.1 * X[:, run.idx], run.W1)                          # (N,P,F)
gap = run.th1[None] - E_REST                                                        # mV from rest to threshold
with np.errstate(divide="ignore", invalid="ignore"):
    tcross = np.where(100 * D > gap, 100 * np.log(100 * D / (100 * D - gap)), np.inf)
order = np.sort(tcross, axis=2); t1, t2 = order[..., 0], order[..., 1]; margin = t2 - t1
fast = tcross.argmin(2)
active = reps.sum(3) > 0; win = reps.argmax(3)                                      # (R,N,P)
rows = []
for n in range(N):
    for p in range(P):
        a = active[:, n, p]
        if a.sum() >= 4 and np.isfinite(t1[n, p]):
            w = win[a, n, p]; rows.append((margin[n, p], np.mean(w == fast[n, p]), np.bincount(w, minlength=F).max() / len(w), t1[n, p], p))
rows = np.array(rows)
np.save("cache/mechanism_rows.npy", rows)
for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    print(f"race margin {lo:>3g}-{hi:<4g} ms: {m.mean() * 100:5.1f}% of active windows | the expected-fastest feature wins {rows[m, 1].mean() * 100:5.1f}% of repeats | modal agreement {rows[m, 2].mean() * 100:5.1f}%")
print(f"median expected first-crossing time {np.median(rows[:, 3]):.0f} ms; median margin {np.median(rows[:, 0]):.1f} ms")
# dead units: how far behind are they?
dead = (st < 0.4) & (win_active[:, None] > 0.2)
behind = []; needed = []
th_off = run.th1 + 52.0
for p in np.where(dead.any(1))[0]:
    for n in range(N):
        if not np.isfinite(t1[n, p]) or D[n, p].max() < 0.05: continue
        best = tcross[n, p].min()
        for f in np.where(dead[p])[0]:
            behind.append(tcross[n, p, f] - best)
            # threshold of the dead unit that would make it cross as early as the current fastest: solve for theta at time `best`
            v_at = E_REST + 100 * D[n, p, f] * (1 - np.exp(-best / 100))      # where the dead unit's v is when the winner crosses
            needed.append(run.th1[p, f] - v_at)                              # how many mV it is still short
behind = np.array(behind); needed = np.array(needed)
print(f"dead units in active windows: behind the winner by median {np.median(behind[np.isfinite(behind)]):.0f} ms (never cross at all on {np.mean(~np.isfinite(behind)) * 100:.0f}% of images); "
      f"short of threshold by median {np.median(needed):.1f} mV when the winner fires")
print(f"theta offsets above -52: structured neurons in those windows median {np.median(th_off[dead.any(1)][st[dead.any(1)] >= 0.4]):.2f} mV, dead units median {np.median(th_off[dead]):.2f} mV")
sim = []
for p in range(P):
    if win_active[p] < 0.5: continue
    T = run.W1[p] - run.W1[p].mean(1, keepdims=True); T /= np.linalg.norm(T, axis=1, keepdims=True); C = T @ T.T
    s = st[p] >= 0.4; Cs = C[np.ix_(s, s)]; np.fill_diagonal(Cs, -1); sim.append(Cs.max(1))
sim = np.concatenate(sim)
print(f"structured features: cosine to their most similar sibling (centred templates) median {np.median(sim):.2f}, >0.5 for {np.mean(sim > 0.5) * 100:.0f}%")
np.savez("cache/mechanism.npz", behind=behind, needed=needed, sim=sim, th_off=th_off, D=D[:50], tcross=tcross[:50])
