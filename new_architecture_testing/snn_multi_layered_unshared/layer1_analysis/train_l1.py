"""
train_l1.py -- numpy replica of multi_layer_unshared.py stage A (FeatureLayer, unshared, learning on), with the logging
the torch code does not have: per-neuron lifetime wins (presentations in which it fired) and spikes, and the structure
of every template at every snapshot. Same equations and constants as FeatureLayer.forward.

  python train_l1.py --out cache/l1_base.npz                       # replica of the default stage A (30k images)
  python train_l1.py --examples 90000 --out cache/l1_long.npz      # longer
"""
import argparse, time
import numpy as np
from l1lib import load_mnist, window_index, structure, E_REST, DECAY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=7); ap.add_argument("--stride", type=int, default=3); ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--norm", type=float, default=12.0); ap.add_argument("--wmax", type=float, default=1.5)
    ap.add_argument("--theta-plus", type=float, default=0.1); ap.add_argument("--theta-tau", type=float, default=2000.0, help="examples; <=0 = no decay")
    ap.add_argument("--inhibition", type=float, default=200.0); ap.add_argument("--no-wta", action="store_true")
    ap.add_argument("--lr-plus", type=float, default=1e-2); ap.add_argument("--lr-minus", type=float, default=1e-4)
    ap.add_argument("--examples", type=int, default=30000); ap.add_argument("--snapshot", type=int, default=20, help="batches")
    ap.add_argument("--steps", type=int, default=350); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--frames", default="0,320,1280,3200,9600,30000", help="examples at which to keep all templates")
    ap.add_argument("--shift-aug", type=int, default=0, help="translate every training image by a random (dx, dy) in [-k, k]^2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    X, Y = load_mnist("train")
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(X))
    idx, side = window_index(a.patch, a.stride); P, K, F, B, T = side * side, a.patch ** 2, a.features, a.batch, a.steps
    W = rng.uniform(0.01, 0.3, (P, F, K)).astype(np.float32); W *= a.norm / W.sum(-1, keepdims=True)
    theta = np.full((P, F), -52.0, np.float32)
    th_decay = 1.0 if a.theta_tau <= 0 else float(np.exp(-1.0 / (a.theta_tau * T / B)))
    dx = dy = float(np.exp(-1 / 20))
    wins = np.zeros((P, F), np.int64); spikes = np.zeros((P, F), np.int64); win_active = np.zeros(P, np.int64)
    frames_at = sorted(int(f) for f in a.frames.split(",") if f.strip()); frames = {}
    snaps = {"examples": [], "struct": [], "wins": [], "theta": [], "W_norm_dW": []}
    first_formed = np.full((P, F), -1, np.int64); wins_at_formed = np.full((P, F), -1, np.int64)
    seen = 0; t0 = time.time(); W_prev = W.copy(); bi = 0
    if 0 in frames_at: frames[0] = W.copy()
    while seen < a.examples:
        sel = order[(seen + np.arange(B)) % len(X)]; x = X[sel]
        if a.shift_aug > 0:
            xs = np.zeros_like(x).reshape(B, 28, 28); src = x.reshape(B, 28, 28)
            for i in range(B):
                ddx, ddy = rng.integers(-a.shift_aug, a.shift_aug + 1, 2)
                ys, ye = max(0, ddy), min(28, 28 + ddy); xs0, xe = max(0, ddx), min(28, 28 + ddx)
                xs[i, ys:ye, xs0:xe] = src[i, ys - ddy:ye - ddy, xs0 - ddx:xe - ddx]
            x = xs.reshape(B, 784)
        v = np.full((B, P, F), E_REST, np.float32); prev = np.zeros((B, P, F), np.float32)
        xt = np.zeros((B, P, K), np.float32); yt = np.zeros((B, P, F), np.float32); cnt = np.zeros((B, P, F), np.float32)
        for t in range(T):
            s_in = (rng.random((B, 784), dtype=np.float32) < 0.1 * x).astype(np.float32)
            patches = s_in[:, idx]
            x_exc = np.einsum("bpk,pfk->bpf", patches, W, optimize=True)
            x_inh = a.inhibition * (prev.sum(2, keepdims=True) - prev)
            v = DECAY * (v - E_REST) + E_REST
            theta = -52.0 + (theta + 52.0) * th_decay
            v += x_exc - x_inh
            s = (v >= theta[None]).astype(np.float32)
            if not a.no_wta and s.any():
                comp = np.where(s > 0, v, -np.inf); top = comp.argmax(2)
                m = np.zeros_like(s); np.put_along_axis(m, top[..., None], 1.0, axis=2); s *= m
            v[s > 0] = E_REST
            theta += s.sum(0) * (a.theta_plus / 16)
            xt = xt * dx + patches; yt = yt * dy + s
            if s.any():
                post = np.einsum("bpf,bpk->pfk", s, xt, optimize=True) / B
            else:
                post = 0.0
            pre = np.einsum("bpf,bpk->pfk", yt, patches, optimize=True) / B
            W += a.lr_plus * post * (a.wmax - W) - a.lr_minus * pre * W
            np.clip(W, 0.0, a.wmax, out=W); W *= a.norm / W.sum(-1, keepdims=True)
            prev = s; cnt += s
        wins += (cnt > 0).sum(0); spikes += cnt.sum(0).astype(np.int64); win_active += (cnt.sum(2) > 0).sum(0)
        seen += B; bi += 1
        for f in frames_at:
            if f not in frames and seen >= f: frames[f] = W.copy()
        if bi % a.snapshot == 0 or seen >= a.examples:
            st = structure(W.reshape(-1, K), a.patch).reshape(P, F)
            newly = (st >= 0.4) & (first_formed < 0); first_formed[newly] = seen; wins_at_formed[newly] = wins[newly]
            snaps["examples"].append(seen); snaps["struct"].append(st.astype(np.float32)); snaps["wins"].append(wins.copy()); snaps["theta"].append(theta.copy())
            snaps["W_norm_dW"].append(np.linalg.norm((W - W_prev).reshape(P * F, K), axis=1).astype(np.float32)); W_prev = W.copy()
            print(f"{seen:>6d} ex | structured {(st >= 0.4).sum():4d}/{P * F} | wins/neuron med {np.median(wins):6.0f} | θ med {np.median(theta):6.2f} | "
                  f"{(time.time() - t0) / 60:5.1f} min", flush=True)
    np.savez_compressed(a.out, W=W, theta=theta, wins=wins, spikes=spikes, win_active=win_active, first_formed=first_formed, wins_at_formed=wins_at_formed,
                        snap_examples=np.array(snaps["examples"]), snap_struct=np.array(snaps["struct"]), snap_wins=np.array(snaps["wins"]),
                        snap_theta=np.array(snaps["theta"]), snap_dW=np.array(snaps["W_norm_dW"]),
                        frame_examples=np.array(sorted(frames)), frames=np.array([frames[k] for k in sorted(frames)]),
                        config=np.array(str(vars(a))), patch=a.patch, stride=a.stride, features=F, side=side)
    print("saved", a.out, f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
