"""
summarize_three_layer.py -- every finished three_layer.py run under a results folder (SMOKE runs skipped) -> summary.csv and
S1_accuracy_overlay.png in that folder.

Per run: group (batch folder), run, variant, seed, the freeze points of layers 1 and 2, each lower layer's own accuracy at its
freeze (held-out / shifted), layer 3's final accuracy (last 3 snapshots), the mean of the last three epochs' snapshots
(held-out / shifted; the steadiest number), the held-out - shifted gap, the best snapshot, layer 2's and layer 3's
calibrated input rates, layer-3 spikes per presentation, minutes.
    python summarize_three_layer.py [--results results]
"""
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def label(c):
    lab = f"L2 {c['l2_block']}x{c['l2_block']}/{c['l2_stride']} F{c['l2_features']}"
    if c.get("l2_readout_inhibition") is not None: lab += f" rd{c['l2_readout_inhibition']:g}"
    if c.get("l2_drive_mult", 1.0) != 1.0: lab += f" drv×{c['l2_drive_mult']:g}"
    if c.get("l2_lr_mult", 1.0) != 1.0: lab += f" lr×{c['l2_lr_mult']:g}"
    if c.get("l2_theta_plus", 0.1) != 0.1: lab += f" θ+{c['l2_theta_plus']:g}"
    if c.get("l1_stride", 2) != 2 or c.get("l1_readout_inhibition", 8.0) != 8.0: lab += f" | L1 s{c['l1_stride']} rd{c['l1_readout_inhibition']:g}"
    if c.get("l3_drive", 0.36) != 0.36 or c.get("l3_lr_mult", 10.0) != 10.0: lab += f" | L3 drv{c['l3_drive']:g} lr×{c['l3_lr_mult']:g}"
    return lab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results")); a = ap.parse_args()
    root = Path(a.results); rows = []; curves = []
    for r in sorted(p.parent for p in root.rglob("run.json")):
        c = json.loads((r / "run.json").read_text())
        if c.get("smoke") or "SMOKE" in r.name or "last3_epochs_accuracy" not in c or not (r / "run.npz").exists(): continue
        l1, l2 = c.get("l1") or {}, c.get("l2") or {}
        rows.append({"group": r.parent.name if r.parent != root else "", "run": r.name, "variant": label(c), "seed": c["seed"],
                     "l1_frozen_at": l1.get("frozen_at"), "l2_frozen_at": l2.get("frozen_at"),
                     "l1_alone": l1.get("readout_accuracy_at_freeze"), "l1_alone_shifted": l1.get("readout_accuracy_shifted_at_freeze"),
                     "l2_alone": l2.get("readout_accuracy_at_freeze"), "l2_alone_shifted": l2.get("readout_accuracy_shifted_at_freeze"),
                     "final_acc": c["final_accuracy"], "final_shifted": c["final_accuracy_shifted"],
                     "last3_epochs": c["last3_epochs_accuracy"], "last3_epochs_shifted": c["last3_epochs_accuracy_shifted"],
                     "gap": c["last3_epochs_accuracy"] - c["last3_epochs_accuracy_shifted"], "max_acc": c["max_accuracy"],
                     "l2_input_per_ms_active_block": c.get("l2_input_rate_per_ms_active_block"), "l3_input_per_ms": c.get("l3_input_rate_per_ms"),
                     "l3_spikes_per_presentation": c.get("final_l3_spikes_per_presentation"), "minutes": c["minutes"]})
        z = np.load(r / "run.npz"); curves.append((label(c) + f" seed {c['seed']}", z["points"], z["accuracy"], z["accuracy_shifted"]))
    if not rows:
        print("no finished runs under", root); return
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'variant':44s} {'seed':>4s} {'L1':>5s} {'L2':>5s} {'final':>6s} {'last3':>6s} {'shift':>6s} {'gap':>5s} {'min':>5s}")
    for r in rows:
        f = lambda v: f"{v:5.1f}" if isinstance(v, (int, float)) else "    -"
        print(f"{r['variant'][:44]:44s} {r['seed']:4d} {f(r['l1_alone'])} {f(r['l2_alone'])} {r['final_acc']:6.1f} {r['last3_epochs']:6.1f} "
              f"{r['last3_epochs_shifted']:6.1f} {r['gap']:5.1f} {r['minutes']:5.0f}")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, (lab, P, A, S) in enumerate(curves):
        col = plt.cm.tab10(i % 10)
        ax.plot(P, A, color=col, lw=1.5, label=lab); bx.plot(P, S, color=col, lw=1.5, ls="--", label=lab)
    for a, ttl in ((ax, "Layer 3, held-out"), (bx, "Layer 3, shifted ±2 px")):
        a.set_title(ttl, loc="left", fontweight="bold"); a.set_xlabel("layer-3 training images"); a.grid(True, ls="--", alpha=0.5)
    ax.axhline(85.6, color="gray", ls=":", lw=1, label="single layer, epochs 4–6 (85.6)"); bx.axhline(60.6, color="gray", ls=":", lw=1, label="single layer (60.6)")
    ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=7, loc="lower right"); bx.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(root / "S1_accuracy_overlay.png", dpi=150); plt.close(fig)
    print("wrote", root / "summary.csv", "and S1_accuracy_overlay.png")


if __name__ == "__main__":
    main()
