"""figure_drive.py -- figs/B3_drive.png: a high layer-2 drive makes layer 2 decide on a few large layer-1 spikes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#bdbbb5", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
                     "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e4df"})
D = json.load(open("cache/drive_data.json"))
B3 = {k: json.load(open(f"cache/batch3_{k}.json")) for k in ("drv1.2_lr1", "drv0.6_lr6", "drv0.36_lr10")}
S = [("drv1.2", "two-layer, drive 1.2 (batch 2 control)", ORANGE), ("drv0.6_lr6", "two-layer, drive 0.6, lr ×6", BLUE),
     ("drv0.36_lr10", "two-layer, drive 0.36, lr ×10", AQUA), ("single", "single layer", GRAY)]

fig = plt.figure(figsize=(19, 9.4)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.28)

# ---- row 1: membrane traces of the winner on one image
for i, (k, lab, col) in enumerate([S[0], S[2], S[3]]):
    ax = fig.add_subplot(gs[0, i]); d = D[k]; v = np.array(d["trace_v"]); T = 80
    t = np.arange(T); ax.plot(t, v[:T], color=col, linewidth=1.8)
    ax.axhline(d["trace_theta"], color=INK2, linestyle="--", linewidth=1); ax.text(T - 1, d["trace_theta"] + 1.5, "threshold θ", ha="right", fontsize=8.5, color=INK2)
    ax.axhline(-65, color=GRAY, linestyle=":", linewidth=1)
    jumps = np.diff(np.r_[-65, v[:T]]); f = d["trace_first"]
    if k != "single":
        for tt in np.where(jumps > 0.8)[0]:
            if f < 0 or tt <= f: ax.annotate("", xy=(tt, v[tt]), xytext=(tt, v[tt] - jumps[tt]), arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.6))
    if f >= 0 and f < T: ax.axvline(f, color=INK, linewidth=0.8, alpha=0.5); ax.text(f + 1, ax.get_ylim()[0] + 3 if False else -63, f"first spike\n{f} ms", fontsize=8.5, va="bottom")
    ax.set_ylim(-68, 25); ax.set_xlim(0, T); ax.set_xlabel("time in the presentation (ms)"); ax.set_ylabel("membrane potential (mV)"); ax.grid(True, alpha=0.6)
    n_in = int(np.sum(np.array(d["trace_in"])[:max(f, 0) + 1])) if k == "single" else None
    ax.set_title(f"({'abc'[i]}) {lab}\nthe winning neuron on one test digit (same y-axis in a-c)" + (f";\n{n_in} pixel spikes by its first spike" if n_in else ""))

# ---- row 2, left: input spikes received before the winner's first spike
ax = fig.add_subplot(gs[1, 0]); bins = np.logspace(0, 3, 40)
for k, lab, col in S:
    x = np.array(D[k]["spikes_before"]); ax.hist(np.maximum(x, 1), bins=bins, histtype="step", linewidth=2, color=col, label=f"{lab}: median {np.median(x):.0f}")
ax.set_xscale("log"); ax.set_xlabel("input spikes the winner had received when it first fired"); ax.set_ylabel("test images (of 500)")
ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(True, alpha=0.6); ax.set_title("(d) How much input the decision is based on")

# ---- row 2, middle: mV per input spike vs spikes needed
ax = fig.add_subplot(gs[1, 1])
for k, lab, col in S:
    m = np.median(D[k]["mv_per_spike"]); n = np.median(D[k]["spikes_before"])
    ax.scatter(m, n, s=120, color=col, zorder=3, edgecolors="white", linewidth=1.2)
    off = {"single": (10, -4), "drv1.2": (10, -14), "drv0.6_lr6": (-150, -22), "drv0.36_lr10": (-150, 6)}[k]
    ax.annotate(f"{lab.split(' (')[0]}\n{m:.1f} mV × {n:.0f} spikes", (m, n), textcoords="offset points", xytext=off, fontsize=8.5)
mm = np.linspace(0.3, 9, 100); ax.plot(mm, 65 / mm, color=GRAY, linestyle=":", linewidth=1); ax.text(0.33, 65 / 0.33 * 1.15, "spikes × mV ≈ 65 mV to threshold", fontsize=8, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.25, 15); ax.set_ylim(5, 400)
ax.set_xlabel("mV the winner gets per input spike (median, before its first spike)"); ax.set_ylabel("input spikes needed (median)"); ax.grid(True, alpha=0.6, which="both")
ax.set_title("(e) Bigger weights per spike → fewer spikes to threshold")

# ---- row 2, right: readout gap
ax = fig.add_subplot(gs[1, 2]); rows = [("drv1.2_lr1", "drive 1.2", ORANGE), ("drv0.6_lr6", "drive 0.6, lr ×6", BLUE), ("drv0.36_lr10", "drive 0.36, lr ×10", AQUA)]
x = np.arange(len(rows)); w = 0.36
best = [B3[k]["best_cosine"] for k, _, _ in rows]; spk = [B3[k]["spiking"] for k, _, _ in rows]
ax.bar(x - w / 2, best, width=w * 0.92, color="#cfcdc7", label="same templates, best match wins")
ax.bar(x + w / 2, spk, width=w * 0.92, color=[c for _, _, c in rows]); ax.bar([0], [0], color=INK2, label="the spiking network (colour = run)")
for xi, b, s_ in zip(x, best, spk):
    ax.text(xi - w / 2, b + 0.8, f"{b:.0f}", ha="center", fontsize=9); ax.text(xi + w / 2, s_ + 0.8, f"{s_:.0f}", ha="center", fontsize=9)
    ax.text(xi, 52, f"gap {b - s_:.0f}", ha="center", fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in rows]); ax.set_ylim(50, 92); ax.set_ylabel("test accuracy (%)"); ax.grid(True, axis="y", alpha=0.6)
ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.set_title("(f) Deciding early costs accuracy: the readout gap closes\nas the drive goes down")

fig.suptitle("High layer-2 drive: each layer-1 spike is worth several mV, so the winning layer-2 neuron reaches threshold on ~12 spikes, "
             "and its answer falls 10 points short of its own templates", fontsize=12, fontweight="bold", x=0.01, ha="left", y=0.995)
fig.text(0.995, 0.005, "numpy replay of the trained Unity weights (batches 2-3, seed 0); 500 test images for (d, e), 1,000 for (f); traces: test image 3 (a 0), one Poisson draw",
         ha="right", fontsize=7.5, color=INK2)
Path("figs").mkdir(exist_ok=True); fig.savefig("figs/B3_drive.png", dpi=150, bbox_inches="tight"); print("ok")
