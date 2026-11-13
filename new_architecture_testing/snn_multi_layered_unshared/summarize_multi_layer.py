"""
summarize_multi_layer.py -- every finished multi_layer.py run under results/ (SMOKE ignored) -> summary.csv, S1, S2.

Per run: variant, seed, l1_converged (layer-1 examples until no flat template remained), l1_frozen_at, final_acc
(last 3 snapshots), final_shifted (same, shifted test set), shift_drop = final_acc - final_shifted, t78 (first snapshot
after which held-out accuracy stays >= 78%), max_acc, spikes_per_presentation at the end, minutes.
S1: held-out (solid) and shifted (dashed) accuracy curves of every run. S2: final accuracy vs shifted accuracy, one
point per run -- the single-layer reference should sit far below the diagonal, a translation-tolerant front end closer to it.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_layer_plots import load_run, spike_time_from_curve  # noqa: E402

def label(cfg):
    if cfg.get("no_l1"): lab = "single layer"
    else: lab = f"{'shared' if cfg.get('shared') else 'unshared'} {cfg['patch']}x{cfg['patch']}/{cfg['stride']} F{cfg['features']}" + (" pool2" if cfg.get("pool2") else "")
    lab += f" n{cfg.get('num_neurons', 400)}" if cfg.get("num_neurons", 400) != 400 else ""
    lab += f" τ{cfg['theta_tau'] / 1000:g}k" if cfg.get("theta_tau") else ""
    lab += f" rd{cfg['l1_readout_inhibition']:g}" if cfg.get("l1_readout_inhibition") is not None and not cfg.get("no_l1") else ""
    lab += f" drv{cfg['l2_drive_arg']:g}" if cfg.get("l2_drive_arg", 1.2) != 1.2 and not cfg.get("no_l1") else ""
    lab += f" lr{cfg['l2_lr_mult']:g}" if cfg.get("l2_lr_mult", 1.0) != 1.0 else ""
    lab += " wta" if cfg.get("l1_wta") or cfg.get("l2_wta") else ""
    lab += f" ep{cfg['num_epochs']}" if cfg.get("num_epochs", 2) != 2 else ""
    return lab

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results")); a = ap.parse_args()
    root = Path(a.results); rows = []
    for r in sorted(root.iterdir()):
        if not r.is_dir() or "SMOKE" in r.name or not (r / "run.json").exists(): continue
        try: d, cfg = load_run(r)
        except Exception as e: print("skip", r.name, e); continue
        pts, acc = d["points"], d["accuracy"]; sh = d.get("accuracy_shifted", np.full(len(pts), np.nan))
        l1 = cfg.get("l1") or {}
        ws = d["window_spikes"]; spp = float(ws[-1].sum() / (cfg["snapshot_interval"] * cfg["batch_size"]))
        rows.append({"run": r.name, "variant": label(cfg), "seed": cfg["seed"], "l1_converged": l1.get("converged_at"), "l1_frozen_at": l1.get("frozen_at"),
                     "final_acc": float(np.mean(acc[-3:])), "final_shifted": float(np.nanmean(sh[-3:])), "shift_drop": float(np.mean(acc[-3:]) - np.nanmean(sh[-3:])),
                     "t78": spike_time_from_curve(pts, acc, 78.0), "max_acc": float(acc.max()), "spikes_per_presentation": spp,
                     "unmoved_rows_end": int(d["reservoir_init"][-1]) if "reservoir_init" in d else None,
                     "cos_init_median_end": float(np.median(d["cos_init_snap"][-1])) if "cos_init_snap" in d else None, "minutes": cfg.get("minutes", float("nan")),
                     "_pts": pts, "_acc": acc, "_sh": sh})
    if not rows: print("no finished runs"); return
    rows.sort(key=lambda r: (r["variant"], r["seed"]))
    cols = [c for c in rows[0] if not c.startswith("_")]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in rows]
    print(f"{'variant':30s} {'seed':>4s} {'l1conv':>7s} {'l1froz':>7s} {'final':>6s} {'shift':>6s} {'drop':>5s} {'t78':>6s} {'max':>6s} {'spk/p':>5s} {'min':>5s}")
    for r in rows:
        t = r["t78"]; t = "none" if (t is None or (isinstance(t, float) and np.isnan(t))) else f"{t / 1000:.0f}k"
        print(f"{r['variant'][:30]:30s} {r['seed']:4d} {str(r['l1_converged']):>7s} {str(r['l1_frozen_at']):>7s} {r['final_acc']:6.1f} {r['final_shifted']:6.1f} {r['shift_drop']:5.1f} {t:>6s} {r['max_acc']:6.1f} {r['spikes_per_presentation']:5.1f} {r['minutes']:5.0f}")
    variants = sorted({r["variant"] for r in rows}); cmap = plt.get_cmap("tab10"); color = {v: cmap(i % 10) for i, v in enumerate(variants)}
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rows:
        ax.plot(r["_pts"], r["_acc"], color=color[r["variant"]], linewidth=1.6, label=f"{r['variant']} (s{r['seed']})")
        ax.plot(r["_pts"], r["_sh"], color=color[r["variant"]], linewidth=1.2, linestyle="--")
    ax.set_xlabel("Layer-2 training examples"); ax.set_ylabel("Accuracy (%)  solid = held-out, dashed = shifted test set"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Held-out and shifted accuracy, every run", loc="left", fontweight="bold"); ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(root / "S1_accuracy_overlay.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for r in rows:
        ax.scatter(r["final_acc"], r["final_shifted"], color=color[r["variant"]], s=60, zorder=3); ax.annotate(f"{r['variant']} s{r['seed']}", (r["final_acc"], r["final_shifted"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    lo = min(min(r["final_acc"], r["final_shifted"]) for r in rows) - 5; ax.plot([lo, 100], [lo, 100], color="gray", linewidth=0.8)
    ax.set_xlim(lo, 100); ax.set_ylim(lo, 100); ax.set_xlabel("Final held-out accuracy (%)"); ax.set_ylabel("Final accuracy on the shifted test set (%)"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Translation tolerance: distance below the diagonal = accuracy lost to a ±2 px shift", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig(root / "S2_shift_tolerance.png", dpi=150); plt.close(fig)
    print("wrote", root / "summary.csv", "S1, S2")

if __name__ == "__main__":
    main()
