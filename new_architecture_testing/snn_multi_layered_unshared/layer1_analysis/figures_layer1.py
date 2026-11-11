"""figures_layer1.py -- figures for the layer-1 analysis (reads cache/ written by the test scripts)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from l1lib import Run, load_mnist, structure, ring_of

OUT = Path("figs"); OUT.mkdir(exist_ok=True)
C = Path("cache")
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, INK2 = "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#bdbbb5", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlelocation": "left", "figure.dpi": 100,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e4df", "grid.linestyle": "-"})
RUN = "runs/unshared_p7s3f16_n400_tp0.5_nodecay_ep2_seed0_l1wta_l2wta_20261106-152133"
FOOT = "layer 1 of unshared 7x7/3 x16, tie-break in both layers, seed 0 (Unity run 20261106-152133)"


def foot(fig, text=FOOT):
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7.5, color=INK2)


def heat(ax, grid, title, cmap="Blues", vmin=0, vmax=None, fmt="{:.0f}", hatch=None):
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    vmax = vmax if vmax is not None else grid.max()
    for (y, x), val in np.ndenumerate(grid):
        ax.text(x, y, fmt.format(val), ha="center", va="center", fontsize=7.5, color="white" if val > 0.6 * vmax else INK)
        if hatch is not None and hatch[y, x]:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, hatch="////", edgecolor=GRAY, linewidth=0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    for s in ax.spines.values(): s.set_visible(False)
    return im


# ------------------------------------------------------------------------------------------------------------- A1
def fig_A1():
    run = Run(RUN); c = np.load(C / "census.npz"); st, wa, ring = c["st"], c["win_active"], c["ring"]
    X, _ = load_mnist("train", 5000); side, F = run.side, run.F
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    ax = axes[0]; ax.imshow(X.mean(0).reshape(28, 28), cmap="Greys"); ax.set_xticks([]); ax.set_yticks([])
    for p, r in enumerate(ring):
        py, px = divmod(p, side); ax.plot(px * 3 + 3, py * 3 + 3, "o", ms=5, color=[ORANGE, YELLOW, AQUA, BLUE][r], mec="white", mew=0.6)
    for (py, px), col in [((0, 0), ORANGE), ((1, 3), YELLOW), ((3, 3), BLUE)]:
        ax.add_patch(plt.Rectangle((px * 3 - 0.5, py * 3 - 0.5), 7, 7, fill=False, edgecolor=col, linewidth=1.6))
    ax.set_title("(a) mean training digit; dots = window centres\n(orange outer ring, yellow ring 1, aqua/blue centre)")
    heat(axes[1], wa.reshape(side, side) * 100, "(b) % of images in which the window fires", vmax=100)
    heat(axes[2], (st >= 0.4).sum(1).reshape(side, side), "(c) structured templates per window (of 16)", vmax=16)
    unstr = (st < 0.4).sum(1).reshape(side, side); starved = (wa < 0.2).reshape(side, side)
    heat(axes[3], unstr, "(d) noise templates per window\n(hatched = starved: fires on < 20% of images)", cmap="Oranges", vmax=16, hatch=starved)
    tot_dead = int(((st < 0.4) & (wa[:, None] >= 0.2)).sum()); tot_starved = int(((st < 0.4) & (wa[:, None] < 0.2)).sum())
    fig.suptitle(f"A1 - Where layer 1's noise templates are: {int((st < 0.4).sum())} of 1,024. {tot_starved} sit in starved windows (rarely any ink); "
                 f"{tot_dead} are DEAD UNITS in windows that fire often", fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig); fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(OUT / "A1_noise_map.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A2
def fig_A2():
    p = C / "l1_base120k.npz"
    if not p.exists(): print("A2: waiting for l1_base120k"); return
    d = np.load(p); ex = d["snap_examples"]; k30 = int(np.argmin(np.abs(ex - 30000)))
    wins30 = d["snap_wins"][k30].ravel(); st30 = d["snap_struct"][k30].ravel(); side = int(d["side"]); F = int(d["features"])
    ring = np.repeat(ring_of(side), F); wf = d["wins_at_formed"].ravel()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.15, 0.85, 1.6]})
    ax = axes[0]; x = np.maximum(wins30, 0.7)
    for r, col, lab in [(0, ORANGE, "outer ring"), (1, YELLOW, "ring 1"), (2, AQUA, "ring 2"), (3, BLUE, "centre")]:
        m = ring == r; ax.scatter(x[m], st30[m], s=9, color=col, alpha=0.7, label=lab, edgecolors="none")
    ax.set_xscale("log"); ax.axhline(0.4, color=GRAY, linewidth=1, linestyle="--"); ax.text(0.8, 0.43, "structured", fontsize=8, color=INK2)
    ax.set_xlabel("presentations the neuron has won in 30,000 training images (log; 0 drawn at 0.7)"); ax.set_ylabel("template structure (1-px autocorrelation)")
    ax.set_title("(a) A template forms if, and only if, its neuron wins"); ax.legend(fontsize=8, loc="upper left", frameon=False); ax.grid(True, alpha=0.6)
    ax = axes[1]; v = wf[wf > 0]
    ax.hist(v, bins=np.logspace(0, 3.5, 30), color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5); ax.set_xscale("log")
    ax.axvline(np.median(v), color=INK, linewidth=1); ax.text(np.median(v) * 1.08, ax.get_ylim()[1] * 0.92, f"median {np.median(v):.0f} wins", fontsize=8.5)
    ax.set_xlabel("wins accumulated when the template first became structured"); ax.set_ylabel("templates")
    ax.set_title("(b) How many wins it takes to form a template"); ax.grid(True, alpha=0.6)
    ax = axes[2]; frames = d["frames"]; fex = d["frame_examples"]
    run = Run(RUN); cen = np.load(C / "census.npz"); wa = cen["win_active"]
    p_show = [0 * side + 0, 1 * side + 3, 3 * side + 3]
    cols = [int(np.argmin(np.abs(fex - e))) for e in (0, 1280, 9600, 30000, 120000)]
    W = np.zeros((len(p_show) * 8, len(cols) * 4 * 8)) * np.nan
    big = []
    for i, pp in enumerate(p_show):
        order = np.argsort(-d["snap_wins"][-1][pp])[:4]
        row = []
        for j in cols:
            for f in order:
                t = frames[j][pp, f].reshape(7, 7); t = (t - t.min()) / (t.max() - t.min() + 1e-9); row.append(np.pad(t, ((0, 1), (0, 1)), constant_values=np.nan))
            row.append(np.full((8, 3), np.nan))
        big.append(np.hstack(row))
    big = np.vstack([np.pad(b, ((0, 2), (0, 0)), constant_values=np.nan) for b in big])
    ax.imshow(big, cmap="gray", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    w_each = 4 * 8 + 3
    for j, e in enumerate((0, 1280, 9600, 30000, 120000)): ax.text(j * w_each + 15, -1.5, f"{e // 1000 if e >= 1000 else e}{'k' if e >= 1000 else ''}", ha="center", fontsize=8.5)
    for i, (pp, lab) in enumerate(zip(p_show, ["corner", "ring 1", "centre"])): ax.text(-1.5, i * 10 + 3.5, lab, ha="right", va="center", fontsize=8.5)
    ax.set_title("(c) The 4 most-winning templates of three windows\nafter 0, 1.3k, 9.6k, 30k, 120k training images", pad=14)
    fig.suptitle(f"A2 - Templates are imprinted by winning: about {np.median(v):.0f} wins make a template; neurons below ~30 wins in 30k images stay random "
                 f"(numpy replica of stage A)", fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig, "numpy replica of stage A (same equations and constants), 120,000 training images, seed 0"); fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "A2_wins_make_templates.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A3
def fig_A3():
    run = Run(RUN); d = run.d; ring = np.repeat(ring_of(run.side), run.F)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    ax = axes[0]; ex = d["l1_points"]; S = d["l1_struct"] >= 0.4
    for r, col, lab in [(0, ORANGE, "outer ring (448)"), (1, YELLOW, "ring 1 (320)"), (2, AQUA, "ring 2 (192)"), (3, BLUE, "centre (64)")]:
        ax.plot(ex, S[:, ring == r].mean(1) * 100, color=col, linewidth=2, label=lab)
    ax.set_xlabel("stage-A training images"); ax.set_ylabel("% of templates structured"); ax.set_ylim(0, 102); ax.grid(True, alpha=0.6)
    ax.set_title("(a) Torch run: formation by ring (stage A ran to its 30k cap)"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[1]; p = C / "l1_base120k.npz"
    if p.exists():
        b = np.load(p); bx = b["snap_examples"]; bS = b["snap_struct"] >= 0.4; bw = b["snap_wins"]; rb = np.repeat(ring_of(int(b["side"])), int(b["features"]))
        for r, col in [(0, ORANGE), (1, YELLOW), (2, AQUA), (3, BLUE)]:
            ax.plot(bx, bS.reshape(len(bx), -1)[:, rb == r].mean(1) * 100, color=col, linewidth=2)
        never = (bw.reshape(len(bx), -1) == 0).sum(1)
        ax.plot(bx, never / 1024 * 100, color=GRAY, linewidth=1.5, linestyle="--", label="neurons that have never won (% of 1,024)")
        ax.axvline(30000, color=INK2, linewidth=1, linestyle=":"); ax.text(31000, 52, "torch stage A stops (30k)", fontsize=8, color=INK2)
        ax.set_title("(b) Numpy replica run 4x longer: the noise does not go away"); ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_xlabel("stage-A training images"); ax.set_ylim(0, 102); ax.grid(True, alpha=0.6)
    fig.suptitle("A3 - More stage-A training does not fix it: the centre forms in a few thousand images, then everything plateaus",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig); fig.tight_layout(rect=(0, 0.02, 1, 0.92)); fig.savefig(OUT / "A3_formation_timeline.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A4
def fig_A4():
    run = Run(RUN); c = np.load(C / "census.npz"); st, wa, wf = c["st"], c["win_active"], c["win_frac"]; m = np.load(C / "mechanism.npz"); nd = np.load(C / "needed.npz")
    dead = (st < 0.4) & (wa[:, None] >= 0.2)
    cand = [p for p in range(run.P) if 0.5 < wa[p] < 0.9]; p = max(cand, key=lambda q: dead[q].sum())
    fig = plt.figure(figsize=(17, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax = fig.add_subplot(gs[0]); order = np.argsort(-wf[p]); th = run.th1[p] + 52
    big = np.full((2 * 11, 8 * 9), np.nan)
    for j, f in enumerate(order):
        r, q = divmod(j, 8); t = run.W1[p, f].reshape(7, 7); big[r * 11:r * 11 + 7, q * 9:q * 9 + 7] = (t - t.min()) / (t.max() - t.min() + 1e-9)
    ax.imshow(big, cmap="gray", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    for j, f in enumerate(order):
        r, q = divmod(j, 8); col = INK if st[p, f] >= 0.4 else ORANGE
        ax.text(q * 9 + 3, r * 11 + 7.6, f"{wf[p, f] * 100:.1f}%", ha="center", va="top", fontsize=8, color=col)
        ax.text(q * 9 + 3, r * 11 + 9.1, f"+{th[f]:.1f} mV", ha="center", va="top", fontsize=7, color=INK2)
    py, px = divmod(p, run.side)
    ax.set_title(f"(a) Window ({py},{px}) fires on {wa[p] * 100:.0f}% of images. Its 16 templates, ordered by\n% of images won (orange = dead) and threshold above −52 mV")
    ax = fig.add_subplot(gs[1]); b = m["behind"]; fin = b[np.isfinite(b)]
    ax.hist(np.clip(fin, 0, 200), bins=np.arange(0, 205, 8), color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.text(198, ax.get_ylim()[1] * 0.62, "≥ 200", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("ms after the winner that a dead unit would cross threshold"); ax.set_ylabel("(dead unit, image) pairs")
    ax.set_title(f"(b) Dead units are far behind, not narrowly beaten:\nmedian {np.median(fin):.0f} ms; on {np.mean(~np.isfinite(b)) * 100:.0f}% of images they cannot reach threshold at all"); ax.grid(True, alpha=0.6)
    ax = fig.add_subplot(gs[2]); ex = nd["extra"]; cur = (run.th1 + 52)[(st >= 0.4) & (wa[:, None] >= 0.2)]
    bins = np.arange(0, 41, 1)
    ax.hist(np.clip(cur, 0, 40), bins=bins, density=True, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5, label="what the conscience gives: structured\nneurons' threshold above −52 mV")
    ax.hist(np.clip(ex + np.median(cur), 0, 40), bins=bins, density=True, color=ORANGE, alpha=0.6, edgecolor="white", linewidth=0.5,
            label="what a dead unit needs: the threshold the faster\nsiblings would need for it to cross first")
    ax.set_xlabel("threshold above −52 mV (40 = 40 or more)"); ax.set_ylabel("density"); ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(True, alpha=0.6)
    ax.set_title(f"(c) The conscience gives the winners +{np.median(cur):.0f} mV; a dead unit\nneeds them at ~+{np.median(cur) + np.median(ex):.0f} mV, i.e. 3-4x stronger")
    fig.suptitle("A4 - Why dead units never win: a random template's drive is far below a learned sibling's, and the threshold conscience is 3-4x too weak to close the gap",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig); fig.tight_layout(rect=(0, 0.02, 1, 0.92)); fig.savefig(OUT / "A4_dead_units.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A5
def fig_A5():
    res = np.load(C / "ablate.npy", allow_pickle=True).item()
    names = list(res); acc = [res[n][0] for n in names]; vis = [res[n][2] for n in names]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    cols = [BLUE, GRAY, ORANGE, AQUA, GRAY]
    y = np.arange(len(names))[::-1]
    ax.barh(y, acc, color=cols, height=0.62)
    for yi, a, v in zip(y, acc, vis):
        ax.text(a + 0.8, yi, f"{a:.1f}%   (layer 2 sees {v:.0f} of 314 layer-1 spikes per image)", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 115); ax.set_xlabel("held-out accuracy of the TRAINED network (numpy replay, labels re-assigned)")
    ax.grid(True, axis="x", alpha=0.6); ax.axvline(res["intact"][0], color=INK2, linewidth=0.8, linestyle=":")
    ax.set_title("A5 - The noise templates are silent: hiding all 347 from layer 2 changes nothing; hiding as many learned ones costs 18 points", fontsize=11.5)
    foot(fig); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "A5_ablation.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A6
def fig_A6():
    srcs = [("pixels", "raw pixels (Poisson counts)", GRAY), ("torch_f16", "layer 1, 16 features (as trained)", BLUE),
            ("torch_f32", "layer 1, 32 features", ORANGE), ("torch_f16_read17.5", "layer 1, 16 features, read out at 17.5 mV", AQUA)]
    keys = [("same_image", "same image,\nnew Poisson draw"), ("shift_1px", "same image\nshifted 1 px"), ("same_class", "another image\nof the same class"), ("diff_class", "an image of a\ndifferent class")]
    E = {s: json.loads((C / "code_eval" / f"{s}.json").read_text()) for s, _, _ in srcs}
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]; w = 0.2; x = np.arange(len(keys))
    for i, (s, lab, col) in enumerate(srcs):
        vals = [E[s][k] for k, _ in keys]
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=col, label=lab)
        for xi, v in zip(x, vals): ax.text(xi + (i - 1.5) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel("cosine similarity of the two codes"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("(a) How similar is the code of a digit to the code of ...")
    ax = axes[1]
    for s, lab, col in srcs:
        e = E[s]; ax.scatter(e["same_class"] / e["diff_class"], e["shift_1px"], s=90, color=col, zorder=3, edgecolors="white", linewidth=1)
        ax.annotate(lab.replace("layer 1, ", "L1 "), (e["same_class"] / e["diff_class"], e["shift_1px"]), textcoords="offset points", xytext=(8, -3), fontsize=8)
    ax.set_xlabel("class contrast (same-class ÷ different-class similarity)"); ax.set_ylabel("tolerance to a 1-px shift"); ax.grid(True, alpha=0.6)
    ax.set_xlim(1, 4.6); ax.set_ylim(0.2, 0.85)
    ax.set_title("(b) The trade: layer 1 buys class contrast with fragility")
    fig.suptitle("A6 - The layer-1 code is fragile: a fresh Poisson draw of the SAME image keeps only 75% of it, a 1-px shift keeps 40%; pixels keep 97% and 75%",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig, "400 test images; rate-cosine of spike-count vectors; same seeds for every source"); fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "A6_code_stability.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A7
def fig_A7():
    rows = np.load(C / "mechanism_rows.npy"); m = np.load(C / "mechanism.npz")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]; mg = rows[:, 0]
    ax.hist(np.clip(mg, 0, 40), bins=np.arange(0, 41, 1), color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("expected gap between the two fastest features (ms; 40 = 40 or more)"); ax.set_ylabel("(image, active window) pairs")
    ax.set_title(f"(a) Most windows are a photo-finish: median gap {np.median(mg):.1f} ms,\n{np.mean(mg < 2) * 100:.0f}% under 2 ms, first crossing at ~{np.median(rows[:, 3]):.0f} ms"); ax.grid(True, alpha=0.6)
    ax = axes[1]; bins = np.array([0, 1, 2, 3, 5, 7, 10, 15, 20, 40, 1e9]); cen = []; agr = []; fst = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (mg >= lo) & (mg < hi)
        if k.sum() > 20: cen.append((lo + min(hi, 60)) / 2); agr.append(rows[k, 2].mean() * 100); fst.append(rows[k, 1].mean() * 100)
    ax.plot(cen, fst, "o-", color=BLUE, linewidth=2, markersize=6, label="the expected-fastest feature wins")
    ax.plot(cen, agr, "s--", color=AQUA, linewidth=1.5, markersize=5, label="repeats agree with the most common winner")
    ax.set_xscale("symlog", linthresh=2); ax.set_xlabel("expected gap (ms)"); ax.set_ylabel("% of 8 Poisson repeats of the same image"); ax.set_ylim(40, 102)
    ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(True, alpha=0.6)
    ax.set_title("(b) Below a ~5 ms gap the winner is decided by the\nPoisson noise of the first ~40 ms, not by the image")
    ax = axes[2]; sim = m["sim"]
    ax.hist(sim, bins=np.linspace(-0.2, 1, 40), color=VIOLET, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("cosine to its most similar sibling (mean-centred templates)"); ax.set_ylabel("structured features in active windows")
    ax.set_title(f"(c) Why the gaps are small: siblings are near-copies\n(median {np.median(sim):.2f}; the same stroke at an offset or slight angle)"); ax.grid(True, alpha=0.6)
    fig.suptitle("A7 - Why the code is noisy: within a window the first feature to cross threshold takes it, and the race is usually a photo-finish between near-identical siblings",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig, "400 test images x 8 Poisson repeats; expected drive = template . (0.1 x intensity), time to cross from the LIF equation"); fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "A7_race.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- A8
def fig_A8():
    srcs = [("pixels", "raw pixels", GRAY), ("torch_f16", "layer-1 code (as trained)", BLUE), ("torch_f16_read17.5", "layer-1 code read out at 17.5 mV", AQUA)]
    E = {s: json.loads((C / "code_eval" / f"{s}.json").read_text()) for s, _, _ in srcs}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]; ks = [1, 4, 10, 40]
    for s, lab, col in srcs:
        ax.plot([k * 10 for k in ks], [E[s]["nn"][str(k)] for k in ks], "o-", color=col, linewidth=2, markersize=6, label=lab)
    ax.set_xscale("log"); ax.set_xticks([10, 40, 100, 400]); ax.set_xticklabels(["10", "40", "100", "400"])
    ax.set_xlabel("stored single examples (all classes), cf. 400 layer-2 neurons"); ax.set_ylabel("1-nearest-neighbour test accuracy (%)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right"); ax.grid(True, alpha=0.6)
    ax.set_title("(a) An imprinting reader: stored instances generalise WORSE in code space")
    ax = axes[1]; cats = ["nearest class centroid\n(averaging reader)", "400-prototype k-means\n(idealised competitive learner)", "actual D&C layer 2\n(2 epochs)"]
    vals = {s: [E[s]["centroid"], E[s]["kmeans400"]] for s, _, _ in srcs}
    dc = {"pixels": 86.8, "torch_f16": 67.6, "torch_f16_read17.5": np.nan}
    x = np.arange(3); w = 0.26
    for i, (s, lab, col) in enumerate(srcs):
        v = vals[s] + [dc[s]]
        ax.bar(x + (i - 1) * w, v, width=w * 0.92, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * w, (vi if np.isfinite(vi) else 0) + 1, f"{vi:.0f}" if np.isfinite(vi) else "not run", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5); ax.set_ylim(0, 100); ax.set_ylabel("test accuracy (%)"); ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("(b) The code helps an averaging reader and hurts a prototype learner")
    fig.suptitle("A8 - What the code is good for: +4 points for a reader that averages, -9 for an ideal competitive learner, -20 for the STDP layer that actually reads it",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig, "codes of 10,000 training / 1,000 test images; cosine similarity; 1-NN averaged over 5 random draws of the stored examples"); fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "A8_readers.png", dpi=150); plt.close(fig)




# ------------------------------------------------------------------------------------------------------------- A9
VARIANTS = [("torch_f16", "as trained (torch, 30k images)"), ("np_base30k", "numpy replica, 30k images"), ("np_base120k", "4x longer stage A (120k)"),
            ("np_tp03", "3x stronger conscience (θ+ 0.3)"), ("np_aug2", "±2 px shift augmentation"), ("np_f8", "8 features per window"),
            ("np_f4", "4 features per window"), ("torch_f32", "32 features per window (torch)"), ("torch_f16_read17.5", "as trained, read out at 17.5 mV")]


def fig_A9():
    E = {}
    for k, _ in VARIANTS:
        f = C / "code_eval" / f"{k}.json"
        if f.exists(): E[k] = json.loads(f.read_text())
    rows = [(k, lab) for k, lab in VARIANTS if k in E]; px = json.loads((C / "code_eval" / "pixels.json").read_text())
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.55 * len(rows) + 3.6), sharey=True, gridspec_kw={"width_ratios": [1, 0.8, 1.1, 1.3]})
    def base(ax, val):
        ax.axvline(val, color=INK2, linewidth=0.8, linestyle=":")
    ax = axes[0]
    for yi, (k, _) in zip(y, rows):
        e = E[k]; ax.plot([e["never_fire"] / e["neurons"] * 100, e["unstructured"] / e["neurons"] * 100], [yi, yi], color="#d8d6d0", linewidth=2, zorder=1)
        ax.scatter(e["unstructured"] / e["neurons"] * 100, yi, s=55, color=ORANGE, zorder=3, label="noise templates" if yi == y[0] else None)
        ax.scatter(e["never_fire"] / e["neurons"] * 100, yi, s=55, marker="D", color=GRAY, zorder=3, label="never fire (2,000 images)" if yi == y[0] else None)
    base(ax, E["np_base30k"]["unstructured"] / 10.24 if "np_base30k" in E else 33)
    ax.set_xlabel("% of layer-1 neurons"); ax.set_title("(a) Wasted capacity"); ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1); ax.grid(True, axis="x", alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in rows])
    ax = axes[1]
    for yi, (k, _) in zip(y, rows): ax.scatter(E[k]["eff_features"], yi, s=55, color=VIOLET, zorder=3)
    base(ax, E["np_base30k"]["eff_features"]); ax.set_xlabel("effective features per active window"); ax.set_title("(b) Features actually used"); ax.grid(True, axis="x", alpha=0.6)
    ax = axes[2]
    for yi, (k, _) in zip(y, rows):
        e = E[k]; ax.plot([e["shift_1px"], e["same_image"]], [yi, yi], color="#d8d6d0", linewidth=2, zorder=1)
        ax.scatter(e["same_image"], yi, s=55, color=BLUE, zorder=3, label="same image, new Poisson draw" if yi == y[0] else None)
        ax.scatter(e["shift_1px"], yi, s=55, marker="s", color=AQUA, zorder=3, label="shifted 1 px" if yi == y[0] else None)
        ax.scatter(e["same_class"], yi, s=55, marker="^", color=YELLOW, zorder=3, label="same class" if yi == y[0] else None)
        ax.text(1.02, yi, f"{e['same_class'] / e['diff_class']:.1f}x", va="center", fontsize=8, color=INK2)
    ax.text(1.02, y[0] + 0.55, "contrast", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.12); ax.set_xlabel("cosine similarity of the codes\n(pixels: 0.97 / 0.75 / 0.51, contrast 1.4x)")
    ax.set_title("(c) Stability; class contrast at right"); ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1); ax.grid(True, axis="x", alpha=0.6)
    ax = axes[3]
    for yi, (k, _) in zip(y, rows):
        e = E[k]
        ax.scatter(e["kmeans400"], yi, s=60, color=BLUE, zorder=3, label="400-prototype k-means" if yi == y[0] else None)
        ax.scatter(e["nn"]["40"], yi, s=55, marker="s", color=ORANGE, zorder=3, label="1-NN, 40 stored per class" if yi == y[0] else None)
        ax.scatter(e["centroid"], yi, s=55, marker="^", color=AQUA, zorder=3, label="class centroid" if yi == y[0] else None)
    for v, col in [(px["kmeans400"], BLUE), (px["nn"]["40"], ORANGE), (px["centroid"], AQUA)]: ax.axvline(v, color=col, linewidth=1, linestyle="--", alpha=0.6)
    ax.text(px["kmeans400"], y[0] + 0.55, "pixels", fontsize=8, color=INK2, ha="center")
    ax.set_xlabel("test accuracy of a reader trained on the code (%)\ndashed = the same reader on pixels"); ax.set_title("(d) What a layer 2 could get out of it")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1); ax.grid(True, axis="x", alpha=0.6); ax.set_xlim(68, 93)
    fig.suptitle("A9 - Layer-1 interventions: recruiting dead units or coarsening the features trades one problem for another; only the soft readout improves every reader",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    foot(fig, "dotted = numpy replica baseline; all layer 1s 7x7/3, 30k stage-A images unless stated; codes of 10k train / 1k test images"); fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(OUT / "A9_interventions.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    for w in which:
        fn = globals().get(f"fig_{w}")
        if fn: fn(); print("done", w)
