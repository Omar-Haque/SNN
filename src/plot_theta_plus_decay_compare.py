#!/usr/bin/env python3
"""
plot_theta_plus_decay_compare.py -- does accuracy depend only on K = theta_plus / (1 - theta_decay) (the plus:decay ratio)?

Outputs in <results_dir>/figures/:
  heatmap.png        final held-out accuracy over (theta_plus, 1 - theta_decay)
  collapse.png       final accuracy (held-out and Cooper-proportion) vs log10 K, one line per theta_plus.
                     If the lines fall on ONE curve, only the ratio matters.
  theta_vs_K.png     final mean theta, silent-sample fraction, active neurons vs K
  curves.png         learning curves grouped by K (half-decade bins)
  summary.csv        per-run numbers; the console prints how much variance K / theta_plus / theta_decay explain
Partial runs are used (marked finished=False in summary.csv).
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TAIL = 5000


def tail(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(y[x >= x[-1] - TAIL].mean()) if len(y) else float("nan")


def load(rd):
    runs = []
    for d in sorted(p for p in rd.iterdir() if p.is_dir()):
        f = d / "curves.npz"
        if not f.exists():
            continue
        try:
            z = np.load(f)
            cfg = json.loads(str(z["config"]))
            dg = {k[5:]: z[k] for k in z.files if k.startswith("diag_")}
            has = "points" in dg and len(dg["points"]) > 0
            runs.append(dict(
                name=d.name, done=(d / "DONE.json").exists(), plus=cfg["theta_plus"], decay=cfg["theta_decay"],
                lam=cfg["lam"], K=cfg["theta_plus"] / cfg["lam"],
                hx=z["heldout_points"], hy=z["heldout_acc"], cx=z["cooper_points"], cp=z["cooper_proportion"],
                theta=tail(dg["points"], dg["theta_mean"]) if has else float("nan"),
                silent=100 * tail(dg["points"], dg["silent_frac"]) if has else float("nan"),
                active=tail(dg["points"], dg["active_neurons"]) if has else float("nan")))
        except Exception as e:
            print("skip", d.name, e)
    for r in runs:
        r["ho"] = tail(r["hx"], r["hy"])
        r["co"] = tail(r["cx"], r["cp"])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_theta_plus_decay")
    a = ap.parse_args()
    rd = HERE / a.results_dir
    runs = load(rd) if rd.exists() else []
    if not runs:
        print("no results in", rd)
        return
    fd = rd / "figures"
    fd.mkdir(exist_ok=True)
    pluss = sorted({r["plus"] for r in runs})
    lams = sorted({round(r["lam"], 12) for r in runs}, reverse=True)   # fast decay first

    # heatmap
    M = np.full((len(pluss), len(lams)), np.nan)
    for r in runs:
        M[pluss.index(r["plus"]), lams.index(round(r["lam"], 12))] = r["ho"]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(lams))); ax.set_xticklabels([f"{l:.0e}" for l in lams])
    ax.set_yticks(range(len(pluss))); ax.set_yticklabels([f"{p:g}" for p in pluss])
    ax.set_xlabel("1 - theta_decay  (left = fast decay)"); ax.set_ylabel("theta_plus (per spike)")
    for i in range(len(pluss)):
        for j in range(len(lams)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, label="final held-out accuracy (%)")
    ax.set_title("Final held-out accuracy (100 neurons)")
    fig.tight_layout(); fig.savefig(fd / "heatmap.png", dpi=130); plt.close(fig)

    cm = plt.get_cmap("viridis")
    col = lambda i: cm(i / max(1, len(pluss) - 1))

    # collapse
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, key, ttl in ((axes[0], "ho", "held-out"), (axes[1], "co", "Cooper proportion (training windows)")):
        for i, p in enumerate(pluss):
            rs = sorted([r for r in runs if r["plus"] == p], key=lambda r: r["K"])
            ax.plot([np.log10(r["K"]) for r in rs], [r[key] for r in rs], "-o", color=col(i), label=f"theta_plus={p:g}")
        ax.set_xlabel("log10 K,  K = theta_plus / (1 - theta_decay)"); ax.set_ylabel("final accuracy (%)")
        ax.set_title(ttl); ax.grid(True, ls="--", alpha=.5); ax.legend(fontsize=8)
    fig.suptitle("Collapse test: one curve => only the plus:decay ratio matters")
    fig.tight_layout(); fig.savefig(fd / "collapse.png", dpi=130); plt.close(fig)

    # theta / silent / active vs K
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, yl in ((axes[0], "theta", "final mean theta (mV)"), (axes[1], "silent", "silent samples (%)"),
                        (axes[2], "active", "neurons that fired in window")):
        for i, p in enumerate(pluss):
            rs = sorted([r for r in runs if r["plus"] == p], key=lambda r: r["K"])
            ax.plot([np.log10(r["K"]) for r in rs], [r[key] for r in rs], "-o", color=col(i), label=f"plus={p:g}")
        ax.set_xlabel("log10 K"); ax.set_ylabel(yl); ax.grid(True, ls="--", alpha=.5)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "theta_vs_K.png", dpi=130); plt.close(fig)

    # curves grouped by K
    bins = sorted({round(2 * np.log10(r["K"])) / 2 for r in runs})
    nrow = int(np.ceil(len(bins) / 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(16, 3.6 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, b in zip(axes.flat, bins):
        for r in runs:
            if round(2 * np.log10(r["K"])) / 2 == b:
                ax.plot(r["hx"], r["hy"], lw=1.4, label=f"plus={r['plus']:g}, 1-decay={r['lam']:.0e}")
        ax.set_title(f"log10 K ~ {b:g}", fontsize=10); ax.grid(True, ls="--", alpha=.5); ax.legend(fontsize=6, loc="lower right")
    fig.supxlabel("training samples"); fig.supylabel("held-out accuracy (%)")
    fig.tight_layout(); fig.savefig(fd / "curves.png", dpi=120); plt.close(fig)

    # summary
    with open(rd / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "finished", "theta_plus", "theta_decay", "1-theta_decay", "K", "heldout_final", "coop_prop_final",
                    "theta_mean", "silent_%", "active"])
        for r in sorted(runs, key=lambda r: r["K"]):
            w.writerow([r["name"], r["done"], r["plus"], r["decay"], f"{r['lam']:.3g}", f"{r['K']:g}", round(r["ho"], 2),
                        round(r["co"], 2), round(r["theta"], 2), round(r["silent"], 2), round(r["active"], 1)])

    def explained(groupkey):
        g = {}
        for r in runs:
            g.setdefault(groupkey(r), []).append(r["ho"])
        y = np.array([r["ho"] for r in runs])
        tot = ((y - y.mean()) ** 2).sum()
        ss = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in g.values())
        return 1 - ss / tot if tot > 0 else float("nan")
    print("\nvariance of final held-out accuracy explained by:")
    print(f"  K bin (half decade) : {explained(lambda r: round(2*np.log10(r['K']))/2):.3f}")
    print(f"  theta_plus alone    : {explained(lambda r: r['plus']):.3f}")
    print(f"  theta_decay alone   : {explained(lambda r: round(r['lam'], 12)):.3f}")
    print("(K explaining clearly more than plus or decay alone => a ratio-like relationship)")
    print("figures + summary.csv in", rd)


if __name__ == "__main__":
    main()
