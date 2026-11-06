#!/usr/bin/env python3
"""
plot_wta_compare.py -- comparison figures + summary table for the inhibition 17.5 / hard-WTA / presentation-time batch.

Reads <results_dir>/W{0,1}_t{150,200,250,350}/curves.npz (partial runs are fine: they are drawn and marked "(partial)").

Outputs in <results_dir>/figures/:
  all_runs.png          ALL 8 configurations on one figure: held-out accuracy, Cooper "proportion", Cooper "all"
                        (colour = presentation time, solid = WTA off, dashed = WTA on)
  by_duration.png       one panel per presentation time: WTA off vs on (held-out + Cooper proportion)
  by_wta.png            one panel per WTA setting: the four presentation times against each other
  diagnostics.png       silent samples, spikes per sample, active neurons, mean theta, weight std for all runs
  summary.csv           final / max accuracy, samples needed to reach 95% of the run's own final accuracy, mean accuracy

Metrics (held-out curve; the Cooper curves are summarised the same way after a 10-window moving average):
  final        mean accuracy over the last 5000 training samples
  max          best single checkpoint (noisy: prefer `final`)
  conv_95      first training-sample count at which a 3-point moving average reaches 95% of `final`
               (smaller = faster convergence)
  auc          mean accuracy over the whole run (area under the learning curve / run length: rewards learning early)
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
STEPS = (150, 200, 250, 350)
WTA = (0, 1)
STEP_COLOR = {150: "#1f77b4", 200: "#2ca02c", 250: "#ff7f0e", 350: "#d62728"}
TAIL = 5000


def name_of(w: int, s: int) -> str:
    return f"W{w}_t{s}"


def pretty(w: int, s: int) -> str:
    return f"{s} ms, WTA {'on' if w else 'off'}"


def style(w: int) -> str:
    return "--" if w else "-"


def load_runs(rd: Path) -> dict:
    runs: dict = {}
    for w in WTA:
        for s in STEPS:
            d = rd / name_of(w, s)
            f = d / "curves.npz"
            if not f.exists():
                continue
            try:
                z = np.load(f)
                runs[(w, s)] = dict(
                    done=(d / "DONE.json").exists(),
                    ho_x=z["heldout_points"], ho_y=z["heldout_acc"],
                    co_x=z["cooper_points"], co_all=z["cooper_all"], co_prop=z["cooper_proportion"],
                    diag={k[5:]: z[k] for k in z.files if k.startswith("diag_")},
                    minutes=float(z["elapsed_s"]) / 60 if "elapsed_s" in z.files else float("nan"),
                )
            except Exception as e:   # file is being rewritten right now
                print(f"skip {d.name}: {e}")
    return runs


def smooth(x, y, k):
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    if k <= 1 or len(y) < k:
        return x, y
    return x[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid")


def tail_mean(x, y, tail=TAIL):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(y[x >= x[-1] - tail].mean())


def conv_point(x, y, frac=0.95, k=3):
    """First x at which the k-point moving average reaches frac * final accuracy."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) < k + 1:
        return float("nan")
    target = frac * tail_mean(x, y)
    xs, ys = smooth(x, y, k)
    hit = np.nonzero(ys >= target)[0]
    return float(xs[hit[0]]) if len(hit) else float("nan")


def auc(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) < 2:
        return float("nan")
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))   # trapezoid rule (np.trapz was removed in NumPy 2)
    return area / float(x[-1] - x[0])


def lab(w, s, r):
    return pretty(w, s) + ("" if r["done"] else " (partial)")


# --------------------------------------------------------------------------------------------- figures
def fig_all(runs, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for (w, s), r in runs.items():
        c, ls = STEP_COLOR[s], style(w)
        axes[0].plot(r["ho_x"], r["ho_y"], ls, color=c, lw=1.8, label=lab(w, s, r))
        for ax, key in ((axes[1], "co_prop"), (axes[2], "co_all")):
            x, y = r["co_x"], r[key]
            if len(y) == 0:
                continue
            ax.plot(x, y, color=c, alpha=0.12, lw=0.7)
            xs, ys = smooth(x, y, 10)
            ax.plot(xs, ys, ls, color=c, lw=1.8, label=lab(w, s, r))
    axes[0].set_title("Held-out accuracy (labels assigned on 2000 train images, scored on 1000 test images)")
    axes[1].set_title("BindsNET-style 'proportion' accuracy on training windows (faint = raw, bold = 10-window mean)")
    axes[2].set_title("BindsNET-style 'all activity' accuracy on training windows")
    for ax in axes:
        ax.set_ylabel("accuracy (%)")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    axes[2].set_xlabel("training samples processed")
    fig.suptitle("Inhibition 17.5 mV: presentation time x hard WTA   (colour = duration, solid = WTA off, dashed = WTA on)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_by_duration(runs, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, STEPS):
        for w in WTA:
            r = runs.get((w, s))
            if r is None:
                continue
            c = "tab:blue" if not w else "tab:red"
            ax.plot(r["ho_x"], r["ho_y"], "-", color=c, lw=1.8, label=f"held-out, WTA {'on' if w else 'off'}")
            xs, ys = smooth(r["co_x"], r["co_prop"], 10)
            ax.plot(xs, ys, ":", color=c, lw=1.6, label=f"Cooper proportion (10-win), WTA {'on' if w else 'off'}")
        ax.set_title(f"{s} ms presentation")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("training samples processed")
    fig.suptitle("Effect of hard WTA at each presentation time (inhibition 17.5 mV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_by_wta(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, w in zip(axes, WTA):
        for s in STEPS:
            r = runs.get((w, s))
            if r is None:
                continue
            ax.plot(r["ho_x"], r["ho_y"], "-", color=STEP_COLOR[s], lw=1.8, label=lab(w, s, r))
        ax.set_title(f"hard WTA {'ON' if w else 'OFF'}")
        ax.set_xlabel("training samples processed")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("held-out accuracy (%)")
    fig.suptitle("Effect of presentation time (inhibition 17.5 mV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_diag(runs, out):
    panels = [("silent_frac", "samples with NO output spike (fraction)", 1.0),
              ("mean_spikes", "output spikes per sample", 1.0),
              ("active_neurons", "neurons that fired in window", 1.0),
              ("theta_mean", "mean theta (mV)", 1.0),
              ("w_std", "weight std", 1.0)]
    have = [r for r in runs.values() if r["diag"] and len(r["diag"].get("points", [])) > 2]
    if not have:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3 * len(panels)), sharex=True)
    for (w, s), r in runs.items():
        d = r["diag"]
        if not d or len(d.get("points", [])) < 3:
            continue
        for ax, (k, _, _) in zip(axes, panels):
            xs, ys = smooth(d["points"], d[k], 10)
            ax.plot(xs, ys, style(w), color=STEP_COLOR[s], lw=1.5, label=lab(w, s, r))
    for ax, (_, yl, _) in zip(axes, panels):
        ax.set_ylabel(yl, fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("training samples processed")
    fig.suptitle("Diagnostics (10-window moving mean): why do the curves differ?", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- summary
def write_summary(runs, out):
    rows = []
    for (w, s), r in runs.items():
        row = dict(run=name_of(w, s), wta="on" if w else "off", steps=s, finished=r["done"],
                   minutes=round(r["minutes"], 1) if r["minutes"] == r["minutes"] else "",
                   samples=int(r["ho_x"][-1]) if len(r["ho_x"]) else "")
        for tag, x, y in (("heldout", r["ho_x"], r["ho_y"]),
                          ("coop_prop", *smooth(r["co_x"], r["co_prop"], 10)),
                          ("coop_all", *smooth(r["co_x"], r["co_all"], 10))):
            if len(y) == 0:
                row.update({f"{tag}_final": "", f"{tag}_max": "", f"{tag}_conv95": "", f"{tag}_auc": ""})
                continue
            cp = conv_point(x, y, k=3 if tag == "heldout" else 1)
            row.update({f"{tag}_final": round(tail_mean(x, y), 2), f"{tag}_max": round(float(np.max(y)), 2),
                        f"{tag}_conv95": int(cp) if cp == cp else "", f"{tag}_auc": round(auc(x, y), 2)})
        d = r["diag"]
        if d and len(d.get("points", [])):
            row["silent_frac_final_%"] = round(100 * tail_mean(d["points"], d["silent_frac"]), 2)
            row["active_neurons_final"] = round(tail_mean(d["points"], d["active_neurons"]), 1)
            row["spikes_per_sample_final"] = round(tail_mean(d["points"], d["mean_spikes"]), 2)
        rows.append(row)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    cols = ["run", "finished", "samples", "minutes", "heldout_final", "heldout_max", "heldout_conv95", "heldout_auc",
            "coop_prop_final", "coop_prop_conv95", "silent_frac_final_%", "active_neurons_final"]
    print("\n" + "  ".join(f"{c[:15]:>15s}" for c in cols))
    for row in rows:
        print("  ".join(f"{str(row.get(c, ''))[:15]:>15s}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_wta")
    a = ap.parse_args()
    rd = HERE / a.results_dir
    runs = load_runs(rd)
    if not runs:
        print("no results found in", rd)
        return
    fd = rd / "figures"
    fd.mkdir(exist_ok=True)
    print("runs found:", ", ".join(f"{name_of(*k)}{'' if r['done'] else '*'}" for k, r in runs.items()),
          "(* = still running / partial)")
    fig_all(runs, fd / "all_runs.png")
    fig_by_duration(runs, fd / "by_duration.png")
    fig_by_wta(runs, fd / "by_wta.png")
    fig_diag(runs, fd / "diagnostics.png")
    write_summary(runs, rd / "summary.csv")
    print("\nfigures + summary.csv written to", rd)


if __name__ == "__main__":
    main()
