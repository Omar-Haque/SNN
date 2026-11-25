"""
three_layer_plots.py -- figures for one three_layer.py run (pure numpy + matplotlib; regenerate with
    python three_layer.py --replot results/<batch>/<run>

F1_accuracy.png        held-out and shifted accuracy across all three stages; the output is layer 1 in stage A, layer 2 in
                       stage A2 and layer 3 in stage B; green dashed lines mark the two freezes
F2_layer1_templates    the 16 layer-1 templates at four windows
F3_layer2_features     layer-2 neurons drawn in pixel space: each weight x the layer-1 template it reads, placed at that
                       template's window (a linear back-projection: what the weights favour, not exactly what makes it fire)
F4_layer3_templates    the 20 most active layer-3 neurons drawn in pixel space through layers 2 and 1, with their labels
F5_stages              per-stage diagnostics: layer-1 and layer-2 templates still unformed, layer-2 block activity, layer 3
F6_layer2_sta          the same layer-2 neurons as F3, shown as the average pixel patch each one fired for on the 1,000 test
                       digits (spike-triggered average): what actually drives the neuron, with no templates involved
F7_layer3_sta          the same 20 layer-3 neurons as F4, as the average test digit each one fired for
F6/F7 need the spike-triggered averages: saved in run.npz by three_layer.py, or in sta.npz by sta_replay.py for older runs.
Standalone (no torch needed):  python three_layer_plots.py <run dir> [<run dir> ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference accuracies (mean of the snapshots of epochs 4-6, held-out / shifted), from the two-layer study's results_2.
REFERENCES = {"single layer": (85.6, 60.6), "two layers, stride 2 + 8 mV (mean of 2 seeds)": (86.3, 57.0)}


def load_run(run_dir):
    run_dir = Path(run_dir)
    return dict(np.load(run_dir / "run.npz", allow_pickle=False)), json.loads((run_dir / "run.json").read_text())


def _k(x):
    return f"{x / 1000:.1f}k" if x < 10000 else f"{x / 1000:.0f}k"


def l2_projection(W1, W2, cfg):
    """(N2, c, c) pixel-space picture of every layer-2 neuron, c = l1_patch + (block - 1) x l1_stride. Input k of block q is
    (dy, dx, f1) -> layer-1 neuron (window (qy*s2 + dy, qx*s2 + dx), feature f1), whose window starts (dy*s1, dx*s1) pixels
    into the block's area."""
    p1, s1, F1, side1 = cfg["l1_patch"], cfg["l1_stride"], cfg["l1_features"], cfg["l1_side"]
    b, s2, F2, side2 = cfg["l2_block"], cfg["l2_stride"], cfg["l2_features"], cfg["l2_side"]
    c = p1 + (b - 1) * s1
    T1 = W1.reshape(side1 * side1, F1, p1, p1).astype(np.float32)
    W2r = W2.reshape(side2 * side2, F2, b, b, F1).astype(np.float32)
    out = np.zeros((side2 * side2, F2, c, c), np.float32)
    for q in range(side2 * side2):
        qy, qx = divmod(q, side2)
        for dy in range(b):
            for dx in range(b):
                p = (qy * s2 + dy) * side1 + (qx * s2 + dx)
                out[q, :, dy * s1:dy * s1 + p1, dx * s1:dx * s1 + p1] += np.einsum("gf,fij->gij", W2r[q, :, dy, dx, :], T1[p])
    return out.reshape(side2 * side2 * F2, c, c)


def l3_projection(W3_rows, proj2, cfg):
    """(n, 28, 28) pixel-space picture of layer-3 rows: each weight x the layer-2 neuron's picture, at its block's position."""
    s1, s2, F2, side2 = cfg["l1_stride"], cfg["l2_stride"], cfg["l2_features"], cfg["l2_side"]
    c = proj2.shape[-1]
    W3r = W3_rows.reshape(len(W3_rows), side2 * side2, F2).astype(np.float32)
    P2 = proj2.reshape(side2 * side2, F2, c, c)
    img = np.zeros((len(W3_rows), 28, 28), np.float32)
    for q in range(side2 * side2):
        qy, qx = divmod(q, side2); y0, x0 = qy * s2 * s1, qx * s2 * s1
        img[:, y0:y0 + c, x0:x0 + c] += np.einsum("ng,gij->nij", W3r[:, q, :], P2[q])
    return img


def l1_backprojection(rows, W1, cfg):
    """(n, 28, 28) pixel-space picture of weight rows over layer 1's lines (--l3-skip): each weight x its layer-1 template."""
    p1, s1, F1, side1 = cfg["l1_patch"], cfg["l1_stride"], cfg["l1_features"], cfg["l1_side"]
    T1 = W1.reshape(side1 * side1, F1, p1 * p1).astype(np.float32); R = rows.reshape(len(rows), side1 * side1, F1).astype(np.float32)
    img = np.zeros((len(rows), 28, 28), np.float32)
    for p in range(side1 * side1):
        wy, wx = divmod(p, side1)
        img[:, wy * s1:wy * s1 + p1, wx * s1:wx * s1 + p1] += np.einsum("nf,fk->nk", R[:, p], T1[p]).reshape(-1, p1, p1)
    return img


def load_sta(d, run_dir):
    """The spike-triggered averages: from run.npz (three_layer.py) or sta.npz (sta_replay.py, for older runs); None if absent."""
    if "sta2" in d: return {k: d[k] for k in ("sta2", "sta2_count", "sta3", "sta3_count")}
    f = Path(run_dir) / "sta.npz"
    return dict(np.load(f)) if f.exists() else None


def layer2_columns(th2, q, F2, n_max=16):
    """The features of block q to show: all of them, or the n_max that won most (largest theta rise), in index order."""
    if F2 <= n_max: return list(range(F2))
    return sorted(np.argsort(-th2[q * F2:(q + 1) * F2])[:n_max].tolist())


def _footer(fig, cfg):
    fig.text(0.01, 0.005, cfg.get("run_name", ""), fontsize=6, color="gray")


def make_figures(run_dir):
    run_dir = Path(run_dir)
    d, cfg = load_run(run_dir)
    f1 = float(cfg["l1"]["frozen_at"]); f2 = float(cfg["l2"]["frozen_at"])

    # ---- F1 accuracy across the three stages ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sh = cfg.get("shift", 2)
    if len(d.get("l1_accuracy", [])):
        ax.plot(d["l1_acc_points"], d["l1_accuracy"], color="teal", lw=2, marker="o", ms=3, label="stage A: layer 1 as the output")
        ax.plot(d["l1_acc_points"], d["l1_accuracy_shifted"], color="teal", lw=1.2, ls="--", alpha=0.8, label=f"stage A, shifted ±{sh} px")
    if len(d.get("l2_accuracy", [])):
        ax.plot(f1 + d["l2_acc_points"], d["l2_accuracy"], color="darkorange", lw=2, marker="o", ms=3, label="stage A2: layer 2 as the output")
        ax.plot(f1 + d["l2_acc_points"], d["l2_accuracy_shifted"], color="darkorange", lw=1.2, ls="--", alpha=0.8, label=f"stage A2, shifted ±{sh} px")
    x3 = f1 + f2 + d["points"]
    ax.plot(x3, d["accuracy"], color="blue", lw=2, label="stage B: layer 3 as the output (held-out, 1k test images)")
    ax.plot(x3, d["accuracy_shifted"], color="purple", lw=1.4, ls="--", label=f"stage B, same images shifted ±{sh} px")
    ax.axvline(f1, color="green", ls="--", lw=1.5, label=f"end of stage A: layer 1 frozen at {_k(f1)}")
    ax.axvline(f1 + f2, color="green", ls="--", lw=1.5, alpha=0.6, label=f"end of stage A2: layer 2 frozen at {_k(f1 + f2)}")
    for (name, (acc, _)), ls in zip(REFERENCES.items(), (":", "-.")):
        ax.axhline(acc, color="gray", lw=0.9, ls=ls, label=f"{name}: {acc:.1f} % (epochs 4–6)")
    ax.set_xlabel("Training examples (stage A: layer 1, stage A2: layer 2, stage B: layer 3)"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100); ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(f"Held-out accuracy — last 3 epochs {cfg.get('last3_epochs_accuracy', float('nan')):.1f} % "
                 f"(shifted {cfg.get('last3_epochs_accuracy_shifted', float('nan')):.1f} %)", loc="left", fontsize=11, fontweight="bold")
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(run_dir / "F1_accuracy.png", dpi=150); plt.close(fig)

    W1, W2, W3 = d["W1"], d["W2"], d["W3"].astype(np.float32)
    p1, F1, side1 = cfg["l1_patch"], cfg["l1_features"], cfg["l1_side"]
    F2, side2 = cfg["l2_features"], cfg["l2_side"]

    # ---- F2 layer-1 templates at four windows -----------------------------------------------------------------------
    mid1 = side1 // 2
    wins = [0, mid1 * side1 + mid1, mid1, mid1 * side1]
    fig, axes = plt.subplots(len(wins), F1, figsize=(1.0 * F1, 1.15 * len(wins)))
    vmax = W1.max()
    for r, p in enumerate(wins):
        for f in range(F1):
            a = axes[r, f]; a.axis("off"); a.imshow(W1[p * F1 + f].reshape(p1, p1), cmap="gray", vmin=0, vmax=vmax)
            if f == 0: a.set_title(f"window ({p // side1},{p % side1})", fontsize=7, loc="left")
    fig.suptitle(f"Layer-1 templates ({p1}×{p1}, stride {cfg['l1_stride']}): the {F1} features at four windows", fontsize=9)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(run_dir / "F2_layer1_templates.png", dpi=150); plt.close(fig)

    # ---- F3 layer-2 features in pixel space -------------------------------------------------------------------------
    proj2 = l2_projection(W1, W2, cfg)
    th2 = d["th2"]
    mid2 = side2 // 2
    blocks = sorted({mid2 * side2 + mid2, mid2, mid2 * side2, (side2 - 1) * side2 + mid2})
    ncol = min(F2, 16)
    fig, axes = plt.subplots(len(blocks), ncol, figsize=(1.0 * ncol, 1.25 * len(blocks)), squeeze=False)
    c = proj2.shape[-1]; o = cfg["l2_stride"] * cfg["l1_stride"]
    for r, q in enumerate(blocks):
        for j, f in enumerate(layer2_columns(th2, q, F2)):
            a = axes[r, j]; a.axis("off")
            a.imshow(proj2[q * F2 + f], cmap="gray")
            a.set_title(f"θ+{th2[q * F2 + f] + 52:.1f}", fontsize=6)
            if j == 0:
                qy, qx = divmod(q, side2)
                a.text(-0.15, 0.5, f"block ({qy},{qx})\npx rows {qy * o}–{qy * o + c - 1}\ncols {qx * o}–{qx * o + c - 1}", fontsize=7,
                       ha="right", va="center", transform=a.transAxes)
    fig.suptitle(f"Layer-2 features in pixel space ({cfg['l2_block']}×{cfg['l2_block']} layer-1 windows = {c}×{c} px); "
                 "title = θ rise in mV (higher = wins more); each image scaled to its own range"
                 + (f"; the {ncol} of {F2} features per block that won most" if F2 > ncol else ""), fontsize=8.5)
    _footer(fig, cfg); fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.04, hspace=0.45, wspace=0.15)
    fig.savefig(run_dir / "F3_layer2_features.png", dpi=150); plt.close(fig)

    # ---- F4 layer-3 templates in pixel space ------------------------------------------------------------------------
    ws = d["window_spikes"][-1] if len(d["window_spikes"]) else np.zeros(len(W3))
    order = np.argsort(-ws)[:20]
    lab = d["labels"] if len(d["labels"]) else np.full(len(W3), -1)
    skip = bool(cfg.get("l3_skip")); N1 = cfg["l1_neurons"]
    fig, axes = plt.subplots(2, 10, figsize=(15, 4.2))
    if not skip:
        imgs = l3_projection(W3[order], proj2, cfg)
        for i, a in enumerate(axes.ravel()):
            a.axis("off")
            if i < len(order): a.imshow(imgs[i], cmap="gray"); a.set_title(f"n{order[i]} → {lab[order[i]]}", fontsize=8)
        fig.suptitle("Layer-3 templates in pixel space, through layers 2 and 1 (20 most active neurons; title = neuron → assigned digit)", fontsize=9)
    else:                    # skip connection: the two pathways drawn separately for the 10 most active neurons
        o10 = order[:10]; via2 = l3_projection(W3[o10][:, N1:], proj2, cfg); via1 = l1_backprojection(W3[o10][:, :N1], W1, cfg)
        share = W3[o10][:, N1:].sum(1) / W3[o10].sum(1)
        for i in range(len(o10)):
            for r, (img, t) in enumerate(((via2[i], f"n{o10[i]} → {lab[o10[i]]}\nvia layer 2 ({100 * share[i]:.0f} %)"), (via1[i], "via layer 1"))):
                a = axes[r, i]; a.axis("off"); a.imshow(img, cmap="gray"); a.set_title(t, fontsize=7)
        fig.suptitle("Layer-3 templates in pixel space, by pathway (--l3-skip; 10 most active neurons; % = share of the neuron's weight on layer-2 lines)", fontsize=9)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    if skip: fig.subplots_adjust(hspace=0.45)
    fig.savefig(run_dir / "F4_layer3_templates.png", dpi=150); plt.close(fig)

    # ---- F5 per-stage diagnostics -----------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 7.5)); (a1, a2, a3), (a4, a5, a6) = axes
    if len(d["l1_points"]):
        a1.plot(d["l1_points"], d["l1_reservoir"], color="black", lw=2)
        a1.set_title(f"Stage A: unstructured layer-1 templates (of {cfg['l1_neurons']})", loc="left", fontweight="bold", fontsize=10)
        a1.set_xlabel("layer-1 training images")
    if len(d["l2_points"]):
        a2.plot(d["l2_points"], d["l2_unmoved"], color="black", lw=2, label="templates still at their initial weights")
        a2b = a2.twinx(); a2b.plot(d["l2_points"], d["l2_theta_med"], color="tab:blue", lw=1.2, label="median θ (mV)"); a2b.set_ylabel("θ (mV)", color="tab:blue")
        a2.set_title(f"Stage A2: layer-2 templates (of {cfg['l2_neurons']})", loc="left", fontweight="bold", fontsize=10); a2.legend(loc="upper right", fontsize=8)
        a3.plot(d["l2_points"], d["l2_distinct"], color="black", lw=2, label="distinct features that fired")
        a3.plot(d["l2_points"], d["l2_tie"], color="tab:red", lw=1.5, label="features in the block's first spiking ms")
        a3.plot(d["l2_points"], d["l2_spikes_per_block"], color="tab:blue", lw=1, ls="--", label="spikes")
        a3.axhline(1, color="gray", lw=0.8); a3.set_ylim(bottom=0); a3.legend(fontsize=8)
        a3.set_title("Stage A2: per active layer-2 block", loc="left", fontweight="bold", fontsize=10)
        a4.plot(d["l2_points"], 100 * d["l2_active_frac"], color="tab:green", lw=2, label="% of blocks with input that fired")
        a4b = a4.twinx(); a4b.plot(d["l2_points"], d["l2_latency"], color="tab:red", lw=1.5, label="first-spike latency (ms)"); a4b.set_ylabel("ms", color="tab:red")
        a4.set_ylim(0, 100); a4.legend(loc="lower left", fontsize=8); a4b.legend(loc="lower right", fontsize=8)
        a4.set_title("Stage A2: layer-2 blocks firing and latency", loc="left", fontweight="bold", fontsize=10)
        for a in (a2, a3, a4): a.set_xlabel("layer-2 training images")
    a5.plot(d["points"], d["l3_unmoved"], color="black", lw=2, label="layer-3 rows still at their initial weights")
    a5b = a5.twinx(); a5b.plot(d["points"], d["l3_spikes_per_presentation"], color="tab:blue", lw=1.5, label="layer-3 spikes per presentation"); a5b.set_ylabel("spikes", color="tab:blue")
    a5.legend(loc="upper right", fontsize=8); a5.set_title("Stage B: layer 3", loc="left", fontweight="bold", fontsize=10); a5.set_xlabel("layer-3 training images")
    a6.plot(d["points"], d["l3_theta_med"], color="tab:blue", lw=2); a6.set_title("Stage B: layer-3 median θ (mV)", loc="left", fontweight="bold", fontsize=10)
    a6.set_xlabel("layer-3 training images")
    for a in (a1, a2, a3, a4, a5, a6): a.grid(True, ls="--", alpha=0.5)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(run_dir / "F5_stages.png", dpi=150); plt.close(fig)

    sta = load_sta(d, run_dir)
    if sta is None: return
    # ---- F6 layer-2 spike-triggered averages (same neurons and layout as F3) ---------------------------------------
    S2, n2 = sta["sta2"], sta["sta2_count"]; cs = S2.shape[-1]
    fig, axes = plt.subplots(len(blocks), ncol, figsize=(1.0 * ncol, 1.25 * len(blocks)), squeeze=False)
    for r, q in enumerate(blocks):
        for j, f in enumerate(layer2_columns(th2, q, F2)):
            a = axes[r, j]; a.axis("off"); n = q * F2 + f
            if n2[n] > 0: a.imshow(S2[n].reshape(cs, cs), cmap="gray")
            else: a.imshow(np.zeros((cs, cs)), cmap="gray", vmin=0, vmax=1)
            a.set_title(f"{int(n2[n])} spk", fontsize=6)
            if j == 0:
                qy, qx = divmod(q, side2)
                a.text(-0.15, 0.5, f"block ({qy},{qx})\npx rows {qy * o}–{qy * o + cs - 1}\ncols {qx * o}–{qx * o + cs - 1}", fontsize=7,
                       ha="right", va="center", transform=a.transAxes)
    fig.suptitle(f"Layer 2: the average {cs}×{cs} pixel patch each neuron fired for (spike-triggered average over the 1,000 test digits; "
                 "same neurons as F3; title = spikes; black = never fired)", fontsize=8.5)
    _footer(fig, cfg); fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.04, hspace=0.45, wspace=0.15)
    fig.savefig(run_dir / "F6_layer2_sta.png", dpi=150); plt.close(fig)

    # ---- F7 layer-3 spike-triggered averages (same neurons as F4) ---------------------------------------------------
    S3, n3 = sta["sta3"], sta["sta3_count"]
    fig, axes = plt.subplots(2, 10, figsize=(15, 4.2))
    for i, a in enumerate(axes.ravel()):
        a.axis("off")
        if i < len(order):
            n = order[i]; a.imshow(S3[n] if n3[n] > 0 else np.zeros((28, 28)), cmap="gray")
            a.set_title(f"n{n} → {lab[n]}, {int(n3[n])} spk", fontsize=8)
    fig.suptitle("Layer 3: the average test digit each neuron fired for (spike-triggered average; same neurons as F4)", fontsize=9)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.03, 1, 0.93)); fig.savefig(run_dir / "F7_layer3_sta.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        make_figures(rd); print("figures written:", rd)
