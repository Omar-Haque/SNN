#!/usr/bin/env python3
"""
plot_theta_compare.py -- figures + summary for H2 (theta_plus / theta_div, C = theta_plus / theta_div).

Runs are discovered from <results_dir>/*/curves.npz; (theta_plus, theta_div, seed) are read from the saved config,
so it works for any subset of runs (partial runs are drawn and marked "(partial)").

Outputs in <results_dir>/figures/:
  same_C.png      one panel per C: held-out accuracy of the different (theta_plus, theta_div) pairs with that C
                  -> H2 predicts the lines coincide (compare with seed_noise.png)
  across_C.png    held-out / Cooper-proportion accuracy for each C (seed-0 runs; line style = theta_div)
  seed_noise.png  baseline C, different seeds and pairs: how far apart are runs that are "the same" by H2?
  theta_diag.png  mean theta, neurons that fired, silent samples, spikes/sample vs training samples (colour = C)
  scatter.png     final accuracy and final mean theta vs C (log axis); marker = theta_div, black = extra seeds
  summary.csv     per run: final / max / auc accuracy, final theta, ...; plus per-C spread across pairs
  (printed)       "pair spread" (max-min of final accuracy over pairs with equal C, seed 0) vs "seed spread"
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TAIL = 5000
DIV_STYLE = {1: "-", 16: "--", 64: ":"}
DIV_MARK = {1: "o", 16: "s", 64: "^"}


def smooth(x, y, k):
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    if k <= 1 or len(y) < k:
        return x, y
    return x[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid")


def tail_mean(x, y, tail=TAIL):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(y[x >= x[-1] - tail].mean()) if len(y) else float("nan")


def auc(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) < 2:
        return float("nan")
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0) / (x[-1] - x[0]))


def load_runs(rd: Path) -> dict:
    runs: dict = {}
    if not rd.exists():
        return runs
    for d in sorted(p for p in rd.iterdir() if p.is_dir()):
        f = d / "curves.npz"
        if not f.exists():
            continue
        try:
            z = np.load(f)
            cfg = json.loads(str(z["config"]))
            plus, div = float(cfg["theta_plus"]), float(cfg["theta_div"])
            runs[d.name] = dict(
                done=(d / "DONE.json").exists(), plus=plus, div=int(div), C=plus / div, seed=int(cfg.get("seed", 0)),
                ho_x=z["heldout_points"], ho_y=z["heldout_acc"],
                co_x=z["cooper_points"], co_prop=z["cooper_proportion"], co_all=z["cooper_all"],
                diag={k[5:]: z[k] for k in z.files if k.startswith("diag_")},
                minutes=float(z["elapsed_s"]) / 60 if "elapsed_s" in z.files else float("nan"))
        except Exception as e:
            print(f"skip {d.name}: {e}")
    return runs


def lab(n, r):
    return f"{n} (theta_plus={r['plus']:g}, div={r['div']})" + ("" if r["done"] else " (partial)")


def c_colors(runs):
    cs = sorted({round(r["C"], 12) for r in runs.values()})
    cmap = plt.get_cmap("viridis")
    return {c: cmap(i / max(1, len(cs) - 1)) for i, c in enumerate(cs)}


# ------------------------------------------------------------------------------------------------ figures
def fig_same_C(runs, out):
    cs = sorted({round(r["C"], 12) for r in runs.values()})
    n = len(cs)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.6 * n), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], cs):
        for name, r in runs.items():
            if round(r["C"], 12) != c or r["seed"] != 0:
                continue
            ax.plot(r["ho_x"], r["ho_y"], DIV_STYLE.get(r["div"], "-"), lw=1.8, label=lab(name, r))
        ax.set_title(f"C = theta_plus/theta_div = {c:g}   (H2: lines should coincide)")
        ax.set_ylabel("held-out acc (%)")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)
    axes[-1, 0].set_xlabel("training samples processed")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_across_C(runs, out):
    col = c_colors(runs)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for name, r in runs.items():
        if r["seed"] != 0:
            continue
        c, ls = col[round(r["C"], 12)], DIV_STYLE.get(r["div"], "-")
        axes[0].plot(r["ho_x"], r["ho_y"], ls, color=c, lw=1.6, label=f"C={r['C']:g}, div={r['div']}" + ("" if r["done"] else " (partial)"))
        xs, ys = smooth(r["co_x"], r["co_prop"], 10)
        axes[1].plot(xs, ys, ls, color=c, lw=1.6)
    axes[0].set_title("Held-out accuracy (colour = C, style = theta_div)")
    axes[1].set_title("Cooper / BindsNET 'proportion' accuracy (10-window mean)")
    for ax in axes:
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)
    axes[1].set_xlabel("training samples processed")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_seed_noise(runs, out):
    base_c = min((r["C"] for r in runs.values()), default=None)
    sel = {n: r for n, r in runs.items() if base_c is not None and abs(r["C"] - base_c) < 1e-12}
    if len(sel) < 2:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for n, r in sel.items():
        ax.plot(r["ho_x"], r["ho_y"], DIV_STYLE.get(r["div"], "-"), lw=1.3 if r["seed"] else 2.0,
                alpha=0.8, label=f"{lab(n, r)}, seed {r['seed']}")
    ax.set_title(f"C = {base_c:g}: all pairs and seeds. If pair-to-pair differences are no larger than seed-to-seed ones, H2 holds.", fontsize=10)
    ax.set_xlabel("training samples processed")
    ax.set_ylabel("held-out accuracy (%)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_diag(runs, out):
    panels = [("theta_mean", "mean theta (mV)"), ("active_neurons", "neurons that fired in window"),
              ("silent_frac", "samples with NO output spike (fraction)"), ("mean_spikes", "output spikes per sample")]
    have = {n: r for n, r in runs.items() if r["seed"] == 0 and r["diag"] and len(r["diag"].get("points", [])) > 2}
    if not have:
        return
    col = c_colors(runs)
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), sharex=True)
    for n, r in have.items():
        for ax, (k, _) in zip(axes, panels):
            xs, ys = smooth(r["diag"]["points"], r["diag"][k], 10)
            ax.plot(xs, ys, DIV_STYLE.get(r["div"], "-"), color=col[round(r["C"], 12)], lw=1.4,
                    label=f"C={r['C']:g}, div={r['div']}")
    for ax, (_, yl) in zip(axes, panels):
        ax.set_ylabel(yl, fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    axes[-1].set_xlabel("training samples processed")
    fig.suptitle("Diagnostics (10-window mean; colour = C, style = theta_div)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_scatter(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in runs.values():
        fin = tail_mean(r["ho_x"], r["ho_y"])
        th = tail_mean(r["diag"]["points"], r["diag"]["theta_mean"]) if r["diag"] else float("nan")
        kw = dict(marker=DIV_MARK.get(r["div"], "o"), s=60, alpha=0.85,
                  color="black" if r["seed"] else "tab:blue", edgecolors="w")
        axes[0].scatter(r["C"], fin, **kw)
        axes[1].scatter(r["C"], th, **kw)
    for ax, yl in zip(axes, ("final held-out accuracy (%)", "final mean theta (mV)")):
        ax.set_xscale("log")
        ax.set_xlabel("C = theta_plus / theta_div")
        ax.set_ylabel(yl)
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].set_title("blue = seed 0, black = extra seeds; marker = theta_div (o=1, s=16, ^=64)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ summary
def write_summary(runs, out):
    rows = []
    for n, r in runs.items():
        d = r["diag"]
        rows.append(dict(
            run=n, finished=r["done"], C=f"{r['C']:g}", theta_plus=f"{r['plus']:g}", theta_div=r["div"], seed=r["seed"],
            samples=int(r["ho_x"][-1]) if len(r["ho_x"]) else "",
            heldout_final=round(tail_mean(r["ho_x"], r["ho_y"]), 2) if len(r["ho_y"]) else "",
            heldout_max=round(float(np.max(r["ho_y"])), 2) if len(r["ho_y"]) else "",
            heldout_auc=round(auc(r["ho_x"], r["ho_y"]), 2) if len(r["ho_y"]) else "",
            coop_prop_final=round(tail_mean(r["co_x"], r["co_prop"]), 2) if len(r["co_prop"]) else "",
            theta_mean_final=round(tail_mean(d["points"], d["theta_mean"]), 3) if d else "",
            active_neurons_final=round(tail_mean(d["points"], d["active_neurons"]), 1) if d else "",
            silent_frac_final_pct=round(100 * tail_mean(d["points"], d["silent_frac"]), 2) if d else ""))
    if not rows:
        return
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["run", "C", "theta_plus", "theta_div", "seed", "heldout_final", "heldout_auc", "coop_prop_final",
            "theta_mean_final", "active_neurons_final", "silent_frac_final_pct"]
    print("\n" + "  ".join(f"{c[:14]:>14s}" for c in cols))
    for row in rows:
        print("  ".join(f"{str(row.get(c, ''))[:14]:>14s}" for c in cols))

    # H2 test: spread over pairs with equal C (seed 0) vs spread over seeds of the same pair
    print("\nH2 check: spread of final held-out accuracy (max - min, in %-points)")
    for c in sorted({round(r["C"], 12) for r in runs.values()}):
        vals = [tail_mean(r["ho_x"], r["ho_y"]) for r in runs.values()
                if round(r["C"], 12) == c and r["seed"] == 0 and len(r["ho_y"])]
        if len(vals) >= 2:
            print(f"  C={c:<10g} across {len(vals)} (theta_plus, theta_div) pairs: {max(vals) - min(vals):6.2f}")
    by_pair: dict = {}
    for r in runs.values():
        if len(r["ho_y"]):
            by_pair.setdefault((round(r["C"], 12), r["div"]), []).append(tail_mean(r["ho_x"], r["ho_y"]))
    for (c, dv), v in sorted(by_pair.items()):
        if len(v) >= 2:
            print(f"  C={c:<10g} div={dv:<3d} across {len(v)} seeds:                  {max(v) - min(v):6.2f}")
    print("  -> H2 is supported if the pair spread is not clearly larger than the seed spread.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_theta")
    a = ap.parse_args()
    rd = HERE / a.results_dir
    runs = load_runs(rd)
    if not runs:
        print("no results found in", rd)
        return
    fd = rd / "figures"
    fd.mkdir(exist_ok=True)
    print("runs found:", ", ".join(f"{n}{'' if r['done'] else '*'}" for n, r in runs.items()), "(* = partial)")
    fig_same_C(runs, fd / "same_C.png")
    fig_across_C(runs, fd / "across_C.png")
    fig_seed_noise(runs, fd / "seed_noise.png")
    fig_diag(runs, fd / "theta_diag.png")
    fig_scatter(runs, fd / "scatter.png")
    write_summary(runs, rd / "summary.csv")
    print("\nfigures + summary.csv written to", rd)


if __name__ == "__main__":
    main()