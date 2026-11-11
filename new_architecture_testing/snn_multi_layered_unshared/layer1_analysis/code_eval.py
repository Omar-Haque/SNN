"""
code_eval.py -- evaluate the code a layer 1 hands to layer 2, independent of layer-2 training.

  python code_eval.py run:<run folder>  [--inh 17.5 --nowta]  --tag NAME
  python code_eval.py npz:<train_l1 output> --tag NAME
  python code_eval.py pixels --tag pixels

Metrics (same seeds for every candidate):
  capacity   neurons that never fire on 2,000 training images; templates without structure; effective features per active
             window (exp of the entropy of how its wins are shared)
  stability  rate-cosine of the code between two Poisson draws of the same image, the image shifted by 1 px, another image
             of the same class, an image of another class (400 test images)
  readers    nearest class centroid (averaging reader, 2,000 train -> 1,000 test); 1-NN from k stored exemplars per class
             (what single-instance imprints give, k = 1, 4, 10, 40; 5 draws); spherical k-means with 400 prototypes on
             10,000 training codes, prototypes labelled by majority (an idealised competitive learner, i.e. what D&C
             learning approximates)
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from l1lib import Run, load_mnist, shift_images, layer1_codes, structure, window_index, ring_of


class L1:
    def __init__(self, spec, inh=None, wta=None):
        kind, path = spec.split(":", 1)
        if kind == "run":
            r = Run(path); self.W1, self.th1, self.idx, self.side, self.P, self.F = r.W1, r.th1, r.idx, r.side, r.P, r.F
            self.l1_inh, self.l1_wta = r.l1_inh, r.l1_wta
        else:
            d = np.load(path); self.W1 = d["W"].astype(np.float32); self.th1 = d["theta"].astype(np.float32)
            self.P, self.F, _ = self.W1.shape; self.idx, self.side = window_index(int(d["patch"]), int(d["stride"])); self.l1_inh, self.l1_wta = 200.0, True
        if inh is not None: self.l1_inh = inh
        if wta is not None: self.l1_wta = wta


def codes(src, X, seed):
    if src == "pixels":
        r = np.random.default_rng(seed); return sum((r.random(X.shape, dtype=np.float32) < 0.1 * X) for _ in range(350)).astype(np.float32)
    return layer1_codes(src, X, seed=seed).reshape(len(X), -1)


def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("--tag", required=True)
    ap.add_argument("--inh", type=float, default=None); ap.add_argument("--nowta", action="store_true")
    a = ap.parse_args(); t0 = time.time(); out = {"tag": a.tag}
    src = "pixels" if a.spec == "pixels" else L1(a.spec, a.inh, False if a.nowta else None)
    Xtr, Ytr = load_mnist("train", 10000); Xte, Yte = load_mnist("test", 1000)
    Ctr = codes(src, Xtr, 1); Cte = codes(src, Xte, 2)
    # ---- capacity
    if src != "pixels":
        cw = Ctr[:2000].reshape(2000, src.P, src.F) > 0
        st = structure(src.W1.reshape(-1, src.W1.shape[-1])).reshape(src.P, src.F)
        out["never_fire"] = int((cw.sum(0) == 0).sum()); out["unstructured"] = int((st < 0.4).sum()); out["neurons"] = src.P * src.F
        eff = []
        for p in range(src.P):
            w = cw[:, p].sum(0)
            if cw[:, p].any(1).mean() > 0.2:
                q = w / w.sum(); q = q[q > 0]; eff.append(np.exp(-(q * np.log(q)).sum()))
        out["eff_features"] = float(np.median(eff)); out["active_ch_per_img"] = float((Cte > 0).sum(1).mean()); out["spikes_per_img"] = float(Cte.sum(1).mean())
    # ---- stability (400 test images)
    X4, Y4 = Xte[:400], Yte[:400]; A = Cte[:400]; B = codes(src, X4, 3); S = codes(src, shift_images(X4, 1, seed=7), 4)
    cos = lambda u, v: (unit(u) * unit(v)).sum(1)
    I, J = np.triu_indices(400, 1); sel = np.random.default_rng(0).choice(len(I), 20000, replace=False); I, J = I[sel], J[sel]
    cc = cos(A[I], B[J])
    out.update(same_image=float(cos(A, B).mean()), shift_1px=float(cos(A, S).mean()), same_class=float(cc[Y4[I] == Y4[J]].mean()), diff_class=float(cc[Y4[I] != Y4[J]].mean()))
    # ---- readers
    Utr, Ute = unit(Ctr), unit(Cte)
    cents = unit(np.stack([Ctr[:2000][Ytr[:2000] == c].mean(0) for c in range(10)]))
    out["centroid"] = float(((Ute @ cents.T).argmax(1) == Yte).mean() * 100)
    g = np.random.default_rng(1); nn = {}
    for k in (1, 4, 10, 40):
        accs = []
        for _ in range(5):
            ex = np.concatenate([g.choice(np.where(Ytr == c)[0], k, replace=False) for c in range(10)])
            accs.append(((Ute @ Utr[ex].T).argmax(1) == Yte).mean() * 100 if True else 0)
            accs[-1] = float((Ytr[ex][(Ute @ Utr[ex].T).argmax(1)] == Yte).mean() * 100)
        nn[k] = float(np.mean(accs))
    out["nn"] = nn
    km = KMeans(400, n_init=1, max_iter=60, random_state=0).fit(Utr)
    lab = np.array([np.bincount(Ytr[km.labels_ == j], minlength=10).argmax() if np.any(km.labels_ == j) else 0 for j in range(400)])
    out["kmeans400"] = float((lab[(Ute @ unit(km.cluster_centers_).T).argmax(1)] == Yte).mean() * 100)
    out["seconds"] = time.time() - t0
    print(json.dumps(out))
    Path("cache/code_eval").mkdir(parents=True, exist_ok=True); Path(f"cache/code_eval/{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
