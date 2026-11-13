"""figures_batch2.py -- one summary figure for batch 2 (the six tau30k_ep6 Unity runs) and the layer-2 readout analysis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer1_analysis"))   # l1lib.py lives there
from l1lib import RESULTS

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#bdbbb5", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
                     "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e4df"})
RUNS = [("singlelayer_n400_tp0.5_tau30k_ep6_seed0_l2wta_20261111-063717", "single layer", GRAY),
        ("unshared_p7s3f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_rd17.5_l2wta_20261111-070123", "16 features, 17.5 mV readout", AQUA),
        ("unshared_p7s3f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_20261111-063618", "16 features (control)", BLUE),
        ("unshared_p7s3f8_n400_tp0.5_tau30k_ep6_seed0_l1wta_rd17.5_l2wta_20261111-070156", "8 features, 17.5 mV readout", VIOLET),
        ("unshared_p7s3f8_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_20261111-063907", "8 features", ORANGE),
        ("unshared_p7s3f4_n400_tp0.5_tau30k_ep6_seed0_l1wta_l2wta_20261111-065128", "4 features", YELLOW)]

def smooth(y, k=3):
    y = np.asarray(y, float); out = y.copy()
    for i in range(len(y)): out[i] = y[max(0, i - k + 1): i + 1].mean()
    return out

fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
ax = axes[0]; ends = []
for name, lab, col in RUNS:
    d = np.load(RESULTS / name / "run.npz"); x = d["points"] / 1000; y = d["accuracy"]
    ax.plot(x, y, color=col, linewidth=0.8, alpha=0.35); ax.plot(x, smooth(y), color=col, linewidth=2)
    ends.append([float(y[-3:].mean()), lab, col])
ends.sort(key=lambda e: -e[0]); ypos = [e[0] for e in ends]
for i in range(1, len(ypos)): ypos[i] = min(ypos[i], ypos[i - 1] - 5.0)
for (v, lab, col), yy in zip(ends, ypos): ax.text(372, yy, f"{v:.1f}  {lab}", color=INK, fontsize=8.5, va="center"); ax.plot([362, 370], [v, yy], color=GRAY, linewidth=0.7)
ax.set_xlim(0, 545); ax.set_xticks(range(0, 361, 60)); ax.set_ylim(15, 92); ax.set_xlabel("layer-2 training examples (thousands; 6 epochs)"); ax.set_ylabel("held-out accuracy (%)"); ax.grid(True, alpha=0.7)
ax.set_title("(a) Batch 2 on Unity: thin = every snapshot, thick = 3-snapshot running mean;\nnumber = final accuracy (mean of the last 3 snapshots)")

ax = axes[1]
st = {k: json.load(open(f"cache/static_{k}.json")) for k in ("single", "f16", "f16_rd17.5")}
km = {"single": 90.9, "f16": 82.0, "f16_rd17.5": 84.1}                      # 400-prototype k-means on the same code (layer1_analysis/cache/code_eval)
stages = ["ideal prototypes\n(k-means on the\nsame input)", "the learned\ntemplates,\nbest match wins", "+ thresholds,\nno spike-timing\nnoise", "the spiking\nnetwork"]
for k, lab, col in [("single", "single layer", GRAY), ("f16_rd17.5", "16 features, 17.5 mV readout", AQUA), ("f16", "16 features (control)", BLUE)]:
    v = [km[k], st[k]["best_cosine"], st[k]["race"], st[k]["spiking"]]
    ax.plot(range(4), v, "o-", color=col, linewidth=2, markersize=7, label=lab)
    for i, vi in enumerate(v): ax.text(i, vi + (1.6 if k != "f16" else -3.6), f"{vi:.0f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=8.5); ax.set_ylim(58, 96); ax.set_xlim(-0.4, 3.4); ax.set_ylabel("test accuracy (%)"); ax.grid(True, axis="y", alpha=0.7)
ax.legend(fontsize=8.5, frameon=False, loc="lower left")
ax.set_title("(b) Where the accuracy goes: the same trained weights,\nread out four ways (numpy replay of the Unity weights)")

ax = axes[2]; rs = json.load(open(sorted(Path("cache").glob("readout_scale_unshared*.json"))[0]))
sc = [r["scale"] for r in rs]; acc = [r["acc"] for r in rs]
xs = np.arange(len(rs)); ax.plot(xs, acc, "o-", color=BLUE, linewidth=2, markersize=7)
for x_, r in zip(xs, rs):
    ax.text(x_, r["acc"] + 3, f"{r['acc']:.0f}%" + (f"\n{r['silent']} of 1,000\nimages silent" if r["silent"] > 5 else ""), ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(xs); ax.set_xticklabels([f"×{r['scale']:g}\n{r['input_spikes_before']:.0f} spikes\n{100 * r['frac_channels']:.0f}% of windows" for r in rs], fontsize=8)
ax.set_xlim(-0.5, len(rs) - 0.5); ax.set_ylim(25, 92); ax.set_ylabel("held-out accuracy (%)")
ax.set_xlabel("layer-2 input weights at test time (fraction of the trained scale),\nwith how much input the winner has received when it first fires")
ax.grid(True, alpha=0.7); ax.set_title("(c) Making layer 2 wait: the trained control network,\nweights scaled at test time only (no retraining)")
fig.suptitle("Batch 2 - the layer-1 changes rank as predicted, but every two-layer run stays 16+ points under the single layer. Layer 2's templates are good for 76-81%;\n"
             "the spiking race reads them out at 66-68% because it is decided on the first ~12 input spikes, when a third of the windows have reported", fontsize=12, fontweight="bold", x=0.01, ha="left")
fig.text(0.995, 0.005, "6 epochs, layer-2 θ decay τ = 30k, tie-break in both layers, seed 0; replay: labels from train images 0-1999, test images 0-999", ha="right", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.9)); fig.subplots_adjust(wspace=0.26); Path("figs").mkdir(exist_ok=True); fig.savefig("figs/B2_summary.png", dpi=150); print("ok")
