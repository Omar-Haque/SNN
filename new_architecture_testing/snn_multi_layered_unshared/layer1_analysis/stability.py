import numpy as np, time, sys
from l1lib import *
run = Run(sys.argv[1] if len(sys.argv) > 1 else "runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133")
inh = float(sys.argv[2]) if len(sys.argv) > 2 else None; wta = None if len(sys.argv) <= 3 else sys.argv[3] == "wta"
tag = sys.argv[4] if len(sys.argv) > 4 else "main"
Xte, Yte = load_mnist("test", 400); R = 6
t0 = time.time()
reps = np.stack([layer1_codes(run, Xte, inh=inh, wta=wta, seed=100 + r).reshape(len(Xte), -1) for r in range(R)])   # (R,N,C)
shifted = layer1_codes(run, shift_images(Xte, 1, seed=7), inh=inh, wta=wta, seed=200).reshape(len(Xte), -1)
def jac(a, b): a, b = a > 0, b > 0; return (a & b).sum(-1) / np.maximum(1, (a | b).sum(-1))
def cos(a, b): return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
same_img = np.mean([jac(reps[i], reps[j]).mean() for i in range(R) for j in range(i + 1, R)])
same_img_cos = np.mean([cos(reps[i], reps[j]).mean() for i in range(R) for j in range(i + 1, R)])
shift1 = jac(reps[0], shifted).mean()
N = len(Xte); I, J = np.triu_indices(N, 1); sel = np.random.default_rng(0).choice(len(I), 20000, replace=False); I, J = I[sel], J[sel]
jj = jac(reps[0][I], reps[1][J]); same_cls = jj[Yte[I] == Yte[J]].mean(); diff_cls = jj[Yte[I] != Yte[J]].mean()
cc = cos(reps[0][I], reps[1][J]); same_cls_c = cc[Yte[I] == Yte[J]].mean(); diff_cls_c = cc[Yte[I] != Yte[J]].mean()
# per active window: does the same image give the same winner? (modal-winner agreement over repeats)
P, F = run.P, run.F; cw = reps.reshape(R, N, P, F)
active = cw.sum(3) > 0                              # (R,N,P)
winner = cw.argmax(3)                               # (R,N,P) (WTA: one feature per window)
agree = []
for n in range(N):
    for p in range(P):
        a = active[:, n, p]
        if a.sum() >= 3:
            w = winner[a, n, p]; agree.append(np.bincount(w).max() / len(w))
agree = np.array(agree)
print(f"[{tag}] same image, new Poisson draw: Jaccard {same_img:.3f} (rate cos {same_img_cos:.3f}) | same image shifted 1 px: {shift1:.3f} | "
      f"same class: {same_cls:.3f} (cos {same_cls_c:.3f}) | different class: {diff_cls:.3f} (cos {diff_cls_c:.3f}) | "
      f"window winner agrees with its modal winner: {agree.mean() * 100:.0f}% of repeats (windows with a unique winner in all repeats: {np.mean(agree == 1) * 100:.0f}%) | "
      f"active channels/img {np.mean((reps[0] > 0).sum(1)):.0f} | {time.time() - t0:.0f}s", flush=True)
np.savez(f"cache/stability_{tag}.npz", same_img=same_img, shift1=shift1, same_cls=same_cls, diff_cls=diff_cls, agree=agree,
         same_img_cos=same_img_cos, same_cls_c=same_cls_c, diff_cls_c=diff_cls_c)
