import numpy as np, time
from l1lib import *
run = Run("runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133")
X, Y = load_mnist("train", 5000)
C = layer1_codes(run, X, seed=21).reshape(len(X), -1); np.save("cache/codes_f16_train5000.npy", C.astype(np.float16))
def mi_binary(on, Y):
    """I(channel fires; class) in bits for each channel; on: (N, C) bool"""
    N = len(Y); py = np.bincount(Y, minlength=10) / N; out = np.zeros(on.shape[1])
    for v in (0, 1):
        m = on if v else ~on
        pv = m.mean(0)                                                       # (C,)
        pvy = np.stack([m[Y == c].mean(0) for c in range(10)], 1) * py[None]  # (C,10) joint
        with np.errstate(divide="ignore", invalid="ignore"):
            t = pvy * np.log2(pvy / (pv[:, None] * py[None]))
        out += np.nansum(t, 1)
    return out
mi = mi_binary(C > 0, Y)
c = np.load("cache/census.npz"); st = c["st"].ravel(); ring = np.repeat(c["ring"], run.F)
print(f"total over channels {mi.sum():.2f} bits (not additive -- an index of how much is spread around) | structured channels carry {mi[st >= 0.4].sum() / mi.sum() * 100:.1f}% | unstructured {mi[st < 0.4].sum() / mi.sum() * 100:.1f}%")
print("per ring share of channel information:", {r: f"{mi[ring == r].sum() / mi.sum() * 100:.0f}%" for r in range(4)})
fr = (C > 0).mean(0)
print(f"best channel {mi.max():.3f} bits (fires on {fr[mi.argmax()] * 100:.0f}% of images); median structured channel {np.median(mi[st >= 0.4]):.3f} bits; class entropy {np.log2(10):.2f} bits")
np.savez("cache/info.npz", mi=mi, fire_frac=fr)
