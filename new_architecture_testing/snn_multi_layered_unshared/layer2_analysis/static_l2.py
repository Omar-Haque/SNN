"""static_l2.py RUN [--weights cache/l2_x.npz] -- what the trained layer-2 templates can do, read out in different ways.

Replays the trained network on 2,000 training images (label assignment) and 1,000 test images, keeping the code that
layer 2 receives (layer-1 spike counts; pixel spike counts for a single-layer run). Then, with the SAME templates W2:
  spiking      the network's own answer (first neuron to threshold takes the presentation; thresholds theta included)
  best drive   winner = argmax_j code . W2[j]              (the template that matches best, no thresholds)
  best cosine  winner = argmax_j cos(code, W2[j])
  race         winner = argmin_j time-to-threshold from the mean drive and theta_j (the race without Poisson timing)
Labels are assigned to neurons by the run's rule in every case (per-class average of wins on the training images).
"""
import sys, json, numpy as np
import l2lib  # noqa: F401  (puts ../layer1_analysis, where l1lib.py lives, on the path)
from l1lib import Run, load_mnist, assign_labels, evaluate
from l2lib import Net

def main():
    run = Run(sys.argv[1]); W = None; th = None; tag = run.name[:44]
    if "--weights" in sys.argv:
        f = np.load(sys.argv[sys.argv.index("--weights") + 1]); W = f["W"].astype(np.float32); th = f["theta"].astype(np.float32); tag = sys.argv[sys.argv.index("--weights") + 1]
    net = Net(run, W2=W if W is not None else run.W2); net.theta = (th if th is not None else run.th2).copy()
    Xa, Ya = load_mnist("train", 2000); Xt, Yt = load_mnist("test", 1000)
    ca, la = net.counts(Xa, 1, return_l1=True); ct, lt = net.counts(Xt, 2, return_l1=True)
    out = {"tag": tag}
    lab = assign_labels(ca, Ya); out["spiking"] = evaluate(ct, Yt, lab)[0]
    Wn = net.W; n = Wn.shape[0]
    def onehot(win): m = np.zeros((len(win), n), np.float32); m[np.arange(len(win)), win] = 1; return m
    def reader(score_a, score_t, name, largest=True):
        wa = score_a.argmax(1) if largest else score_a.argmin(1); wt = score_t.argmax(1) if largest else score_t.argmin(1)
        l = assign_labels(onehot(wa), Ya); out[name] = evaluate(onehot(wt), Yt, l)[0]; return wa, wt
    Da, Dt = la @ Wn.T, lt @ Wn.T                                                    # total drive over the presentation (mV)
    wa_d, wt_d = reader(Da, Dt, "best_drive")
    un = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    reader(un(la) @ un(Wn).T, un(lt) @ un(Wn).T, "best_cosine")
    def race(D):                                                                     # time to reach theta_j at the mean drive rate
        r = D / 350.0; gap = (net.theta + 65.0)[None]
        with np.errstate(divide="ignore", invalid="ignore"): t = np.where(100 * r > gap, 100 * np.log(100 * r / (100 * r - gap)), np.inf)
        return t
    wa_r, wt_r = reader(race(Da), race(Dt), "race", largest=False)
    spk_w = ct.argmax(1); fired = ct.sum(1) > 0
    out["spiking_winner_is_best_drive"] = float((spk_w == wt_d)[fired].mean() * 100); out["spiking_winner_is_race_winner"] = float((spk_w == wt_r)[fired].mean() * 100)
    # where does the spiking winner rank by drive?
    rank = (Dt > Dt[np.arange(len(Dt)), spk_w][:, None]).sum(1); out["spiking_winner_drive_rank_median"] = float(np.median(rank[fired]))
    # purity of the neurons' wins on the training images
    wc = np.stack([(ca[Ya == c] > 0).sum(0) for c in range(10)], 1).astype(float)   # (n, 10) presentations won per class
    out["win_purity"] = float(wc.max(1).sum() / wc.sum()); out["neurons_winning"] = int((wc.sum(1) > 0).sum())
    # drive contrast: best template vs the median template, and theta spread expressed in the same units
    best = np.sort(Dt, 1)[:, -1]; second = np.sort(Dt, 1)[:, -2]; med = np.median(Dt, 1)
    out["drive_best_over_median"] = float(np.median(best / med)); out["drive_best_over_second"] = float(np.median(best / second))
    out["theta_p10_p50_p90"] = [float(np.percentile(net.theta, q)) for q in (10, 50, 90)]
    out["code_spikes_per_image"] = float(lt.sum(1).mean()); out["l2_spikes_per_image"] = float(ct.sum(1).mean()); out["silent_test_images"] = int((~fired).sum())
    # templates: weight concentration and how class-like they are (cosine with the class-mean code)
    cent = un(np.stack([la[Ya == c].mean(0) for c in range(10)])); cs = un(Wn) @ cent.T; srt = np.sort(cs, 1)
    out["template_cos_to_best_class_centroid"] = float(np.median(srt[:, -1])); out["template_margin_over_second_class"] = float(np.median(srt[:, -1] - srt[:, -2]))
    k = max(1, Wn.shape[1] // 20); out["weight_share_top5pct_inputs"] = float(np.median(np.sort(Wn, 1)[:, -k:].sum(1) / Wn.sum(1)))
    print(json.dumps(out, indent=1, default=float))
    key = "single" if run.no_l1 else f"f{run.F}" + (f"_rd{run.l1_inh:g}" if run.l1_inh != 200 else "")
    open(f"cache/static_{key}.json", "w").write(json.dumps(out, indent=1, default=float))

main()
