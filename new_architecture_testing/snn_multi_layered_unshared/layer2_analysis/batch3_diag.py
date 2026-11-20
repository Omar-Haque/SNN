"""batch3_diag.py RUN [RUN ...] -- for each trained run: the four readers of static_l2.py plus the decision probe of
probe_first.py, in one row (so that runs trained with different --l2-drive / --l2-lr-mult can be compared)."""
import sys, json, numpy as np
import l2lib  # noqa: F401
from l1lib import Run, load_mnist, assign_labels, evaluate
from l2lib import Net

Xa, Ya = load_mnist("train", 2000); Xt, Yt = load_mnist("test", 1000)
un = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
for path in sys.argv[1:]:
    run = Run(path); net = Net(run, W2=run.W2); net.theta = run.th2.copy(); n = net.n; W = net.W
    ca, la = net.counts(Xa, 1, return_l1=True)
    rng = np.random.default_rng(2); res = [net.present(Xt[i:i + 125], learn=False, rng=rng, probe=True) for i in range(0, 1000, 125)]
    ct = np.concatenate([r[0] for r in res]); lt = np.concatenate([r[1] for r in res]); tf = np.concatenate([r[2] for r in res]); cf = np.concatenate([r[3] for r in res])
    ok = tf >= 0; lab = assign_labels(ca, Ya)
    def onehot(win): m = np.zeros((len(win), n), np.float32); m[np.arange(len(win)), win] = 1; return m
    def reader(sa, st_, largest=True):
        wa = sa.argmax(1) if largest else sa.argmin(1); wt = st_.argmax(1) if largest else st_.argmin(1)
        return evaluate(onehot(wt), Yt, assign_labels(onehot(wa), Ya))[0]
    def race(D):
        r = D / 350.0; gap = (net.theta + 65.0)[None]
        with np.errstate(divide="ignore", invalid="ignore"): return np.where(100 * r > gap, 100 * np.log(100 * r / (100 * r - gap)), np.inf)
    wc = np.stack([(ca[Ya == c] > 0).sum(0) for c in range(10)], 1).astype(float); winner = ct.argmax(1)
    k = max(1, W.shape[1] // 20)
    out = dict(run=run.name, drive=run.cfg.get("l2_drive_arg", 1.2), lr_mult=run.cfg.get("l2_lr_mult", 1.0), readout=run.cfg.get("l1_readout_inhibition"),
               unity_final=run.cfg["final_accuracy"], spiking=evaluate(ct, Yt, lab)[0], race=reader(race(la @ W.T), race(lt @ W.T), False),
               best_cosine=reader(un(la) @ un(W).T, un(lt) @ un(W).T), silent=int((~ok).sum()), win_purity=float(wc.max(1).sum() / wc.sum()),
               neurons_winning=int((wc.sum(1) > 0).sum()), first_spike_ms=float(np.median(tf[ok])), input_spikes_before=float(np.median(cf.sum(1)[ok])),
               channels_before=float(np.median((cf > 0).sum(1)[ok])), channels_total=float(np.median((lt > 0).sum(1))),
               frac_channels=float(np.median(((cf > 0).sum(1) / np.maximum(1, (lt > 0).sum(1)))[ok])), l2_spikes=float(ct.sum(1).mean()),
               mean_weight=float(W.mean()), largest_weight_median=float(np.median(W.max(1))), threshold_distance=float(np.median((net.theta + 65)[winner][ok])),
               weight_share_top5pct=float(np.median(np.sort(W, 1)[:, -k:].sum(1) / W.sum(1))), theta_p10_p50_p90=[float(np.percentile(net.theta, q)) for q in (10, 50, 90)])
    print(json.dumps(out), flush=True)
    key = f"drv{out['drive']:g}_lr{out['lr_mult']:g}" + (f"_rd{out['readout']:g}" if out["readout"] else "")
    open(f"cache/batch3_{key}.json", "w").write(json.dumps(out, indent=1))
