"""
sta_replay.py -- spike-triggered averages for three_layer.py runs made before the network computed them itself (batch 1),
in numpy (no torch). Runs each trained network at its stage-B readout settings on the first --n test digits, averages for
every layer-2 neuron the pixel patch of its block area and for every layer-3 neuron the whole image, each weighted by how
many times the neuron fired on it, writes sta.npz into the run folder and redraws the run's figures (F6, F7 added).

    python sta_replay.py <run dir> [<run dir> ...] [--n 1000]

Same equations as three_layer.py with learning off: LIF v <- e^(-1/100)(v - E_rest) + E_rest + x_exc - x_inh, spike when
v >= theta, tie-break, reset to E_rest; Bernoulli(0.1 x intensity) pixel spikes per ms for 350 ms; layer 1 at its readout
inhibition, layer 2 at its readout inhibition (or its training value), layer 3 at 200 mV. The Poisson draws differ from
torch's, so counts differ run to run by noise only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "new_architecture_testing" / "snn_multi_layered_unshared" / "layer1_analysis"))
sys.path.insert(0, str(HERE))
from l1lib import load_mnist, window_index, E_REST, DECAY, _wta      # noqa: E402  (the two-layer study's numpy replica)
from three_layer_plots import make_figures                             # noqa: E402


def lif_step(v, x_exc, prev, inh, theta, wta, axis):
    """One ms of a competing group (within a window / block: axis 2; all of layer 3: axis 1)."""
    v = DECAY * (v - E_REST) + E_REST + x_exc - inh * (prev.sum(axis, keepdims=True) - prev)
    s = (v >= theta).astype(np.float32)
    if wta: s = _wta(s, v, axis)
    v[s > 0] = E_REST
    return v, s


def replay(run_dir: Path, n: int, B: int = 100, seed: int = 0) -> None:
    c = json.loads((run_dir / "run.json").read_text()); z = np.load(run_dir / "run.npz")
    p1, s1, F1 = c["l1_patch"], c["l1_stride"], c["l1_features"]
    idx1, side1 = window_index(p1, s1); P1 = side1 * side1
    W1 = z["W1"].astype(np.float32).reshape(P1, F1, p1 * p1); th1 = z["th1"].astype(np.float32).reshape(P1, F1)
    b, s2, F2, side2 = c["l2_block"], c["l2_stride"], c["l2_features"], c["l2_side"]; P2 = side2 * side2
    idx2 = np.array([[((qy * s2 + dy) * side1 + qx * s2 + dx) * F1 + f for dy in range(b) for dx in range(b) for f in range(F1)]
                     for qy in range(side2) for qx in range(side2)])           # the same table as BlockLayer.idx
    W2 = z["W2"].astype(np.float32).reshape(P2, F2, -1); th2 = z["th2"].astype(np.float32).reshape(P2, F2)
    W3t = z["W3"].astype(np.float32).T.copy(); th3 = z["th3"].astype(np.float32); N3 = len(th3)
    inh1 = c["l1_readout_inhibition"]
    inh2 = c["l2_readout_inhibition"] if c.get("l2_readout_inhibition") is not None else c["l2_inhibition"]
    inh3 = c["l3_inhibition"]; skip = bool(c.get("l3_skip"))
    wta1, wta2, wta3 = c.get("l1_wta", True), c.get("l2_wta", True), c.get("l3_wta", True)
    T = c.get("steps", 350); cc = p1 + (b - 1) * s1; o = s2 * s1
    X, Y = load_mnist("test", n); rng = np.random.default_rng(seed); c3_all = []
    s2sum = np.zeros((P2, F2, cc, cc), np.float64); n2 = np.zeros((P2, F2)); s3sum = np.zeros((N3, 28, 28), np.float64); n3 = np.zeros(N3)
    for i in range(0, n, B):
        x = X[i:i + B]; bb = len(x); img = x.reshape(bb, 28, 28)
        v1 = np.full((bb, P1, F1), E_REST, np.float32); a1 = np.zeros_like(v1)
        v2 = np.full((bb, P2, F2), E_REST, np.float32); a2 = np.zeros_like(v2); c2 = np.zeros_like(v2)
        v3 = np.full((bb, N3), E_REST, np.float32); a3 = np.zeros_like(v3); c3 = np.zeros_like(v3)
        for t in range(T):
            s_in = (rng.random((bb, 784), dtype=np.float32) < 0.1 * x).astype(np.float32)
            v1, a1 = lif_step(v1, np.einsum("bpk,pfk->bpf", s_in[:, idx1], W1, optimize=True), a1, inh1, th1[None], wta1, 2)
            z1 = a1.reshape(bb, -1)
            v2, a2 = lif_step(v2, np.einsum("bpk,pfk->bpf", z1[:, idx2], W2, optimize=True), a2, inh2, th2[None], wta2, 2)
            c2 += a2
            z3 = np.concatenate([z1, a2.reshape(bb, -1)], 1) if skip else a2.reshape(bb, -1)
            v3, a3 = lif_step(v3, z3 @ W3t, a3, inh3, th3[None], wta3, 1)
            c3 += a3
        patches = np.stack([img[:, (q // side2) * o:(q // side2) * o + cc, (q % side2) * o:(q % side2) * o + cc] for q in range(P2)], 1)
        s2sum += np.einsum("bqf,bqij->qfij", c2, patches); n2 += c2.sum(0)
        s3sum += np.einsum("bn,bij->nij", c3, img); n3 += c3.sum(0)
        c3_all.append(c3)
        print(f"  {run_dir.name[:60]}: {i + bb}/{n} images", flush=True)
    sta2 = (s2sum / np.maximum(n2, 1)[..., None, None]).reshape(P2 * F2, cc, cc).astype(np.float32)
    sta3 = (s3sum / np.maximum(n3, 1)[:, None, None]).astype(np.float32)
    np.savez(run_dir / "sta.npz", sta2=sta2, sta2_count=n2.reshape(-1), sta3=sta3, sta3_count=n3, n_images=n)
    # check against the run: classify these test digits with the run's own final labels (the estimator's vote: summed spikes per class)
    lab = z["labels"].astype(int); C3 = np.concatenate(c3_all); onehot = np.zeros((N3, 10)); ok = lab >= 0; onehot[np.where(ok)[0], lab[ok]] = 1
    acc = 100 * float(((C3 @ onehot).argmax(1) == Y).mean())
    print(f"{run_dir.name}: replay accuracy with the run's labels {acc:.1f}% (run's final held-out accuracy {c['final_accuracy']:.1f}%)", flush=True)
    print(f"{run_dir.name}: layer 2 never fired: {int((n2 == 0).sum())}/{P2 * F2} | layer 3 never fired: {int((n3 == 0).sum())}/{N3}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=1000); a = ap.parse_args()
    for r in a.runs:
        rd = Path(r); replay(rd, a.n); make_figures(rd); print("figures written:", rd, flush=True)


if __name__ == "__main__":
    main()
