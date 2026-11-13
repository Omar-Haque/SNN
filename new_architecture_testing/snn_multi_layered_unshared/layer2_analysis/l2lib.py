"""
l2lib.py -- torch-free replica of the two-layer network WITH layer-2 learning (stage B of multi_layer_unshared.py), written
for speed on a CPU: layer 1 is frozen (taken from a finished run), every product uses the sparsity of the spikes.

Same equations and order of operations as DiehlAndCookNetwork.forward / excitatory_neurons.forward:
    v <- e^(-1/100)(v - E_rest) + E_rest ; theta decays (learning only) ; v += x_exc - x_inh ; spike = v >= theta ;
    tie-break (one spike per image per step, highest v) ; v[spike] = E_rest ; theta += spikes.sum(batch) * theta_plus / 16
    x_trace = x_trace e^(-1/20) + in ; y_trace = y_trace e^(-1/20) + out
    W += lr+ (out^T x_trace / B)(w_max - W) - lr- (y_trace^T in / B) W ; clamp [0, w_max] ; rows rescaled to w_norm
Only rows whose neuron has fired in the current batch can change, so only those rows are updated (identical result).
"""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer1_analysis"))   # l1lib.py lives there
from l1lib import Run, E_REST, DECAY, window_index

DX = float(np.exp(-1 / 20))


class Net:
    def __init__(self, run, W2=None, w_norm=None, w_max=None, theta_plus=0.5, theta_tau=30000.0, lr_plus=1e-2, lr_minus=1e-4,
                 l2_inh=200.0, l2_wta=True, l1_inh=None, B=32, T=350, seed=0, theta_rule="add"):
        self.run = run; self.no_l1 = run.no_l1; self.B, self.T = B, T
        self.rng = np.random.default_rng(seed)
        if not self.no_l1:
            P, F, K = run.W1.shape; self.P, self.F = P, F
            M = np.zeros((784, P * F), np.float32)                       # pixel -> layer-1 neuron weights
            for p in range(P):
                for k in range(K): M[run.idx[p, k], p * F:(p + 1) * F] = run.W1[p, :, k]
            self.M = M; self.th1 = run.th1.reshape(1, P, F).copy(); self.l1_inh = run.l1_inh if l1_inh is None else l1_inh; self.l1_wta = run.l1_wta
            self.n_in = P * F
        else:
            self.n_in = 784
        self.W = (run.d["W2_init"] if W2 is None else W2).astype(np.float32).copy(); self.n = self.W.shape[0]
        self.W_init = self.W.copy()
        self.w_norm = float(run.cfg["l2_norm"] if w_norm is None else w_norm); self.w_max = float(run.cfg["l2_wmax"] if w_max is None else w_max)
        self.theta = np.full(self.n, -52.0, np.float32); self.theta_plus = theta_plus
        self.th_decay = 1.0 if not theta_tau else float(np.exp(-1.0 / (theta_tau * T / B)))
        self.lr_plus, self.lr_minus, self.l2_inh, self.l2_wta = lr_plus, lr_minus, l2_inh, l2_wta
        self.seen = 0

    # -------------------------------------------------------------------------------------------------- one batch
    def present(self, x, learn=True, return_l1=False, rng=None, probe=False):
        rng = rng or self.rng; b = x.shape[0]; n = self.n; T = self.T
        if probe: return_l1 = True; t_first = np.full(b, -1); c1_first = np.zeros((b, self.n_in), np.float32)
        if not self.no_l1:
            v1 = np.full((b, self.P, self.F), E_REST, np.float32); p1 = np.zeros((b, self.P, self.F), np.float32)
            c1 = np.zeros((b, self.n_in), np.float32) if return_l1 else None
        else:
            c1 = np.zeros((b, 784), np.float32) if return_l1 else None            # single layer: the 'code' is the pixel spike counts
        v2 = np.full((b, n), E_REST, np.float32); p2 = np.zeros((b, n), np.float32); c2 = np.zeros((b, n), np.float32)
        if learn:
            xt = np.zeros((b, self.n_in), np.float32); yt = np.zeros((b, n), np.float32); fired_rows = np.zeros(n, bool)
        px = 0.1 * x; WT = np.ascontiguousarray(self.W.T)                 # (n_in, n) copy kept in step with W
        for t in range(T):
            s_in = rng.random((b, 784), dtype=np.float32) < px
            if self.no_l1:
                z = sparse.csr_matrix(s_in, dtype=np.float32)
                if return_l1: c1 += s_in
            else:
                x_exc = (sparse.csr_matrix(s_in, dtype=np.float32) @ self.M).reshape(b, self.P, self.F)
                v1 = DECAY * (v1 - E_REST) + E_REST + x_exc - self.l1_inh * (p1.sum(2, keepdims=True) - p1)
                s1 = v1 >= self.th1
                if s1.any():
                    if self.l1_wta:
                        comp = np.where(s1, v1, -np.inf); top = comp.argmax(2); m = np.zeros_like(s1); np.put_along_axis(m, top[..., None], True, axis=2); s1 &= m
                    v1[s1] = E_REST; p1 = s1.astype(np.float32)
                else:
                    p1 = np.zeros_like(v1)
                z = sparse.csr_matrix(p1.reshape(b, -1))
                if return_l1: c1 += p1.reshape(b, -1)
            v2 = DECAY * (v2 - E_REST) + E_REST
            if learn: self.theta = -52.0 + (self.theta + 52.0) * self.th_decay
            v2 += (z @ WT) - self.l2_inh * (p2.sum(1, keepdims=True) - p2)
            s2 = v2 >= self.theta[None]
            any2 = s2.any()
            if any2:
                if self.l2_wta:
                    comp = np.where(s2, v2, -np.inf); top = comp.argmax(1); m = np.zeros_like(s2); m[np.arange(b), top] = True; s2 &= m
                v2[s2] = E_REST; p2 = s2.astype(np.float32); c2 += p2
                if probe:
                    new = s2.any(1) & (t_first < 0); t_first[new] = t; c1_first[new] = c1[new]
            else:
                p2 = np.zeros_like(v2)
            if learn:
                if any2: self.theta += p2.sum(0) * (self.theta_plus / 16)
                xt *= DX
                if z.nnz: xt += z.toarray() if self.no_l1 else p1.reshape(b, -1)
                yt *= DX
                if any2: yt += p2; fired_rows |= s2.any(0)
                rows = np.where(fired_rows)[0]
                if len(rows) and (any2 or z.nnz):
                    W = self.W
                    if any2:                                                          # potentiation: only the neurons that spiked this step
                        rn = np.where(s2.any(0))[0]
                        dplus = self.lr_plus * ((p2[:, rn].T @ xt) / b) * (self.w_max - W[rn])
                    if z.nnz:                                                         # depression: only the inputs that spiked this step
                        cols = np.unique(z.indices); ix = np.ix_(rows, cols)
                        zc = z[:, cols].toarray()
                        sub = W[ix]; W[ix] = sub - self.lr_minus * ((yt[:, rows].T @ zc) / b) * sub
                    if any2: W[rn] += dplus
                    Wr = np.clip(W[rows], 0.0, self.w_max); Wr *= self.w_norm / Wr.sum(1, keepdims=True)
                    W[rows] = Wr; WT[:, rows] = Wr.T
        if learn: self.seen += b
        if probe: return c2, c1, t_first, c1_first
        return (c2, c1) if return_l1 else c2

    # -------------------------------------------------------------------------------------------------- evaluation
    def counts(self, X, seed, B=128, return_l1=False):
        rng = np.random.default_rng(seed); out = [self.present(X[i:i + B], learn=False, return_l1=return_l1, rng=rng) for i in range(0, len(X), B)]
        if return_l1: return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])
        return np.concatenate(out)

    def cos_init(self):
        return (self.W * self.W_init).sum(1) / (np.linalg.norm(self.W, axis=1) * np.linalg.norm(self.W_init, axis=1) + 1e-12)
