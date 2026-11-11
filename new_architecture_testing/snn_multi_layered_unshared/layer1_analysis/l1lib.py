"""
l1lib.py -- torch-free replica of multi_layer_unshared.py's inference path (layer 1, optional pool2, layer 2, label
assignment, evaluation) plus a layer-1 STDP trainer, for analysing finished runs. Same equations, same constants:

  LIF  v <- e^(-1/100) (v - E_rest) + E_rest + x_exc - x_inh;  spike = v >= theta;  WTA tie-break;  v[spike] = E_rest
  input  Bernoulli(0.1 x intensity) per ms, 350 ms
  layer 1  x_exc = patch . W[p,f];  x_inh = inh x (spikes of the OTHER features of the window at t-1)
  layer 2  x_exc = s1 . W2^T;       x_inh = 200 x (spikes of the OTHER layer-2 neurons at t-1)
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
# MNIST: <repo>/data/MNIST/raw (the folder torchvision downloads into); a local ./data also works
DATA = ROOT / "data" if (ROOT / "data").exists() else ROOT.parents[2] / "data" / "MNIST" / "raw"
# finished runs: ../results/<run name> (a local ./runs/<run name> also works)
RESULTS = ROOT / "runs" if (ROOT / "runs").exists() else ROOT.parent / "results"
E_REST, DECAY = -65.0, float(np.exp(-1 / 100))


# ------------------------------------------------------------------------------------------------ data
def load_mnist(split="train", n=None, data_dir=None):
    d = Path(data_dir) if data_dir else DATA
    pre = "train" if split == "train" else "t10k"
    X = np.frombuffer((d / f"{pre}-images-idx3-ubyte").read_bytes(), dtype=np.uint8, offset=16).reshape(-1, 784)
    Y = np.frombuffer((d / f"{pre}-labels-idx1-ubyte").read_bytes(), dtype=np.uint8, offset=8)
    if n is not None: X, Y = X[:n], Y[:n]
    return X.astype(np.float32) / 255.0, Y.astype(np.int64)


def shift_images(X, shift=2, seed=1234):
    """multi_layer_unshared.shifted_test_set: each image translated by a fixed random (dx, dy) in [-shift, shift]^2."""
    g = np.random.default_rng(seed); out = np.zeros_like(X); imgs = X.reshape(-1, 28, 28)
    for i in range(len(X)):
        dx, dy = int(g.integers(-shift, shift + 1)), int(g.integers(-shift, shift + 1))
        ys, ye = max(0, dy), min(28, 28 + dy); xs, xe = max(0, dx), min(28, 28 + dx)
        out.reshape(-1, 28, 28)[i, ys:ye, xs:xe] = imgs[i, ys - dy:ye - dy, xs - dx:xe - dx]
    return out


# ------------------------------------------------------------------------------------------------ geometry
def window_index(patch=7, stride=3, side_img=28):
    side = (side_img - patch) // stride + 1
    idx = np.array([[(py * stride + ky) * side_img + px * stride + kx for ky in range(patch) for kx in range(patch)]
                    for py in range(side) for px in range(side)])
    return idx, side


def ring_of(side):
    """distance of each window from the border: 0 = outermost ring."""
    yy, xx = np.mgrid[0:side, 0:side]
    return np.minimum(np.minimum(yy, xx), np.minimum(side - 1 - yy, side - 1 - xx)).ravel()


# ------------------------------------------------------------------------------------------------ template metrics
def structure(W, patch=7):
    """1-px spatial autocorrelation of the mean-centred template (random ~0, p95 0.16; stroke bands 0.5-0.9)."""
    T = W.reshape(-1, patch, patch).astype(np.float64); T = T - T.mean(axis=(1, 2), keepdims=True)
    def cos(a, b): return (a * b).sum(axis=(1, 2)) / (np.sqrt((a * a).sum(axis=(1, 2)) * (b * b).sum(axis=(1, 2))) + 1e-12)
    return 0.5 * (cos(T[:, 1:, :], T[:, :-1, :]) + cos(T[:, :, 1:], T[:, :, :-1]))


# ------------------------------------------------------------------------------------------------ runs
class Run:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "run.json").exists(): self.path = RESULTS / self.path.name      # "runs/<name>" -> ../results/<name>
        self.name = self.path.name
        self.cfg = json.loads((self.path / "run.json").read_text()); self.d = dict(np.load(self.path / "run.npz"))
        c = self.cfg; self.no_l1 = bool(c.get("no_l1"))
        if not self.no_l1:
            self.patch, self.stride, self.F = c["patch"], c["stride"], c["features"]
            self.idx, self.side = window_index(self.patch, self.stride); self.P = self.side ** 2; self.K = self.patch ** 2
            self.W1 = self.d["W1"].astype(np.float32).reshape(self.P, self.F, self.K)
            self.th1 = self.d["l1_theta"].astype(np.float32).reshape(self.P, self.F)
            self.l1_inh = c.get("l1_readout_inhibition") if c.get("l1_readout_inhibition") is not None else c["l1_inhibition"]
            self.l1_wta = bool(c.get("l1_wta")); self.pool2 = bool(c.get("pool2"))
        self.W2 = self.d["W2"].astype(np.float32); self.th2 = self.d["theta_snap"][-1].astype(np.float32)
        self.l2_wta = bool(c.get("l2_wta")); self.l2_inh = float(c.get("inhibition_mV", 200.0))


# ------------------------------------------------------------------------------------------------ simulation
def _wta(s, v, axis):
    comp = np.where(s > 0, v, -np.inf); top = comp.argmax(axis)
    m = np.zeros_like(s); np.put_along_axis(m, np.expand_dims(top, axis), 1.0, axis=axis)
    return s * m


def run_layer1_batch(x, W1, th1, idx, inh=200.0, wta=True, T=350, rng=None, record_first=False, record_t=False):
    """x (B,784) intensities -> spike counts (B,P,F) [, first-spike time (B,P,F), per-step spikes (T,B,P,F) bool]."""
    rng = rng or np.random.default_rng(0)
    B = x.shape[0]; P, F, K = W1.shape
    v = np.full((B, P, F), E_REST, np.float32); prev = np.zeros((B, P, F), np.float32); cnt = np.zeros((B, P, F), np.float32)
    first = np.full((B, P, F), np.inf, np.float32) if record_first else None
    st = np.zeros((T, B, P, F), bool) if record_t else None
    for t in range(T):
        s_in = (rng.random((B, 784), dtype=np.float32) < 0.1 * x).astype(np.float32)
        x_exc = np.einsum("bpk,pfk->bpf", s_in[:, idx], W1, optimize=True)
        x_inh = inh * (prev.sum(2, keepdims=True) - prev)
        v = DECAY * (v - E_REST) + E_REST + x_exc - x_inh
        s = (v >= th1[None]).astype(np.float32)
        if wta: s = _wta(s, v, 2)
        v[s > 0] = E_REST; cnt += s; prev = s
        if record_first: first = np.where((s > 0) & np.isinf(first), t, first)
        if record_t: st[t] = s > 0
    out = (cnt,)
    if record_first: out += (first,)
    if record_t: out += (st,)
    return out if len(out) > 1 else cnt


def layer1_codes(run, X, inh=None, wta=None, seed=0, B=100, record_first=False, W1=None, th1=None):
    rng = np.random.default_rng(seed); inh = run.l1_inh if inh is None else inh; wta = run.l1_wta if wta is None else wta
    W1 = run.W1 if W1 is None else W1; th1 = run.th1 if th1 is None else th1
    outs = [run_layer1_batch(X[i:i + B], W1, th1, run.idx, inh, wta, rng=rng, record_first=record_first) for i in range(0, len(X), B)]
    if record_first: return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])
    return np.concatenate(outs)


def pool2x2(s, side, F):
    B = s.shape[0]; s = s.reshape(B, side, side, F)[:, : side // 2 * 2, : side // 2 * 2, :]
    return s.reshape(B, side // 2, 2, side // 2, 2, F).max(axis=(2, 4)).reshape(B, -1)


def run_network(run, X, seed=0, B=100, mask=None, l1_inh=None, T=350):
    """Full network, learning off. Returns layer-2 spike counts (N, 400) [and layer-1 counts (N, P*F)].
    mask: (P*F,) 0/1 -- layer-1 channels hidden from layer 2 (layer-1 dynamics untouched)."""
    rng = np.random.default_rng(seed); n2 = run.W2.shape[0]; out2 = []; out1 = []
    W2t = run.W2.T.copy(); inh1 = run.l1_inh if (l1_inh is None and not run.no_l1) else l1_inh
    for i in range(0, len(X), B):
        x = X[i:i + B]; b = x.shape[0]
        if not run.no_l1:
            v1 = np.full((b, run.P, run.F), E_REST, np.float32); p1 = np.zeros_like(v1); c1 = np.zeros_like(v1)
        v2 = np.full((b, n2), E_REST, np.float32); p2 = np.zeros((b, n2), np.float32); c2 = np.zeros((b, n2), np.float32)
        for t in range(T):
            s_in = (rng.random((b, 784), dtype=np.float32) < 0.1 * x).astype(np.float32)
            if run.no_l1: z = s_in
            else:
                x_exc = np.einsum("bpk,pfk->bpf", s_in[:, run.idx], run.W1, optimize=True)
                v1 = DECAY * (v1 - E_REST) + E_REST + x_exc - inh1 * (p1.sum(2, keepdims=True) - p1)
                s1 = (v1 >= run.th1[None]).astype(np.float32)
                if run.l1_wta: s1 = _wta(s1, v1, 2)
                v1[s1 > 0] = E_REST; p1 = s1; c1 += s1
                z = s1.reshape(b, -1)
                if mask is not None: z = z * mask
                if run.pool2: z = pool2x2(z, run.side, run.F)
            v2 = DECAY * (v2 - E_REST) + E_REST + z @ W2t - run.l2_inh * (p2.sum(1, keepdims=True) - p2)
            s2 = (v2 >= run.th2[None]).astype(np.float32)
            if run.l2_wta: s2 = _wta(s2, v2, 1)
            v2[s2 > 0] = E_REST; p2 = s2; c2 += s2
        out2.append(c2)
        if not run.no_l1: out1.append(c1.reshape(b, -1))
    return (np.concatenate(out2), np.concatenate(out1) if out1 else None)


def assign_labels(counts, Y, n_classes=10):
    """assign_neuron_labels: per-class average spike count; argmax; -1 if the neuron never fired."""
    avg = np.stack([counts[Y == c].mean(0) if np.any(Y == c) else np.zeros(counts.shape[1]) for c in range(n_classes)], 1)
    lab = avg.argmax(1); lab[counts.sum(0) == 0] = -1
    return lab


def evaluate(counts, Y, labels, n_classes=10):
    A = np.zeros((counts.shape[1], n_classes)); ok = labels >= 0; A[np.where(ok)[0], labels[ok]] = 1.0
    pred = (counts @ A).argmax(1)
    return float((pred == Y).mean() * 100), pred
