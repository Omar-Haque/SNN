"""
replay_layer1.py -- torch-free diagnostic: replay a finished run's frozen layer 1 on real MNIST test digits (numpy, inference
only, the same LIF equations as FeatureLayer) and measure what kind of code it hands to layer 2.

    python replay_layer1.py results/<run>                 # trained readout (WTA, the run's inhibition)
    python replay_layer1.py results/<run> --inhibition 17.5 --no-wta
    python replay_layer1.py results/<run> --pooling       # also the pooled versions of the code

Reports, for the first 1,000 test images: spikes and active channels per image; the within-class vs between-class
similarity of the code (Jaccard of the active sets and cosine of the rates) against the same numbers for raw pixels; and
nearest-class-centroid accuracy (cosine, 700 train / 300 test) on the code and on pixels. Saves the per-image spike counts
as l1_code_<tag>.npy in the run folder.

Why this exists (2026-11-06): the layer-1 code of the 7x7/3 x16 WTA run is MORE class-informative than pixels (centroid
87.0 % vs 82.0 %, class contrast 3.6x vs 1.5x) yet layer 2 learns it slowly, because two instances of a class share only
~15 % of their active channels, so a D&C neuron's single-instance imprint rarely wins the next instance of its class.
"""
import argparse, json, numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--inhibition", type=float, default=None, help="within-window inhibition (mV); default: the run's readout value")
    ap.add_argument("--no-wta", action="store_true", help="no within-step tie-break (the run's layer 1 used one if l1_wta is set)")
    ap.add_argument("--images", type=int, default=1000); ap.add_argument("--pooling", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run); d = np.load(run / "run.npz"); cfg = json.loads((run / "run.json").read_text())
    if cfg.get("no_l1") or "W1" not in d: raise SystemExit("this run has no layer 1")
    patch, stride, F = cfg["patch"], cfg["stride"], cfg["features"]; side = (28 - patch) // stride + 1; P, K = side * side, patch * patch
    W = d["W1"].astype(np.float32).reshape(P, F, K); TH = d["l1_theta"].astype(np.float32).reshape(P, F)
    idx = np.array([[((py * stride) + ky) * 28 + px * stride + kx for ky in range(patch) for kx in range(patch)] for py in range(side) for px in range(side)])
    inhib = a.inhibition if a.inhibition is not None else (cfg.get("l1_readout_inhibition") if cfg.get("l1_readout_inhibition") is not None else cfg["l1_inhibition"])
    wta = bool(cfg.get("l1_wta")) and not a.no_wta
    tag = f"inh{inhib:g}" + ("_wta" if wta else "")
    raw = Path(__file__).resolve().parents[2] / "data" / "MNIST" / "raw"
    N = a.images
    X = np.frombuffer((raw / "t10k-images-idx3-ubyte").read_bytes(), dtype=np.uint8, offset=16)[: N * 784].reshape(N, 784).astype(np.float32) / 255.0
    Y = np.frombuffer((raw / "t10k-labels-idx1-ubyte").read_bytes(), dtype=np.uint8, offset=8)[:N]
    rng = np.random.default_rng(a.seed); decay = np.exp(-1 / 100); E = -65.0; T = cfg["num_steps"]

    def run_layer1(x):
        B = x.shape[0]; v = np.full((B, P, F), E, np.float32); prev = np.zeros((B, P, F), np.float32); cnt = np.zeros((B, P, F), np.float32)
        for t in range(T):
            s_in = (rng.random((B, 784)) < 0.1 * x).astype(np.float32)          # spikegen.rate(data * 0.1): Bernoulli per ms
            patches = s_in[:, idx]
            x_exc = np.einsum("bpk,pfk->bpf", patches, W); x_inh = inhib * (prev.sum(2, keepdims=True) - prev)
            v = decay * (v - E) + E + x_exc - x_inh
            s = (v >= TH[None]).astype(np.float32)
            if wta:
                comp = np.where(s > 0, v, -np.inf); top = comp.argmax(2); m = np.zeros_like(s); np.put_along_axis(m, top[..., None], 1.0, axis=2); s = s * m
            v[s > 0] = E; cnt += s; prev = s
        return cnt
    counts = np.concatenate([run_layer1(X[i:i + 50]) for i in range(0, N, 50)])        # (N, P, F)
    np.save(run / f"l1_code_{tag}.npy", counts.astype(np.float16))

    def pair_stats(M):
        js, jd, cs, cd = [], [], [], []
        for i in range(0, N, 7):
            for j in range(i + 1, N, 13):
                a_, b_ = M[i] > 0, M[j] > 0; jac = (a_ & b_).sum() / max(1, (a_ | b_).sum())
                cos = M[i] @ M[j] / (np.linalg.norm(M[i]) * np.linalg.norm(M[j]) + 1e-9)
                (js if Y[i] == Y[j] else jd).append(jac); (cs if Y[i] == Y[j] else cd).append(cos)
        return np.mean(js), np.mean(jd), np.mean(cs), np.mean(cd)

    def ncc(M):
        ntr = int(0.7 * N); tr, te = slice(0, ntr), slice(ntr, N); cents = np.stack([M[tr][Y[tr] == c].mean(0) for c in range(10)])
        Mn = M[te] / (np.linalg.norm(M[te], axis=1, keepdims=True) + 1e-9); Cn = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        return (np.argmax(Mn @ Cn.T, 1) == Y[te]).mean() * 100

    def report(lab, M):
        js, jd, cs, cd = pair_stats(M)
        print(f"  {lab:34s} channels {M.shape[1]:5d} | active/image {np.mean((M > 0).sum(1)):6.1f} | Jaccard within {js:.3f} between {jd:.3f} (x{js / jd:.2f}) | "
              f"rate cosine within {cs:.3f} between {cd:.3f} (x{cs / cd:.2f}) | centroid acc {ncc(M):.1f}%")
    code = counts.reshape(N, -1)
    print(f"{run.name}\n[{tag}] layer-1 spikes per image {code.sum(1).mean():.0f} | active windows {(counts.sum(2) > 0).sum(1).mean():.1f} | "
          f"features per active window {((counts > 0).sum(2)[counts.sum(2) > 0]).mean():.2f} | first-spike latency n/a (counts only)")
    report("layer-1 code", code); report("raw pixels (for comparison)", X)
    if a.pooling:
        def pool(c, k, st):
            Np, S, _, F_ = c.shape; out = []
            for y in range(0, S - k + 1, st):
                for x in range(0, S - k + 1, st): out.append(c[:, y:y + k, x:x + k].max(axis=(1, 2)))
            return np.stack(out, 1).reshape(Np, -1)
        c = counts.reshape(N, side, side, F)
        report("pool 2x2 stride 2 (= --pool2)", pool(c, 2, 2)); report("pool 2x2 stride 1", pool(c, 2, 1)); report("pool 3x3 stride 1", pool(c, 3, 1))


if __name__ == "__main__":
    main()
