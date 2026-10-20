"""
summarize_sweep.py -- collect every finished sweep.py run under results/ (SMOKE runs ignored) into one table and four
cross-run figures. Torch-free: reads only run.npz / run.json.

Per run (summary.csv):
  theta_plus, inhibition_mV, init, theta_init, epochs, seed, examples (total presented),
  final_acc        held-out accuracy averaged over the last 3 snapshots (the ceiling actually reached)
  max_acc          best single snapshot
  t75, t78         first snapshot after which held-out accuracy never drops below 75% / 78% again (NaN = never);
                   t78 is hypo_1's spike criterion, t75 is a softer convergence time that low-ceiling variants can reach
  acc_at_30k       held-out accuracy at the first snapshot >= 30k examples (speed read-out independent of a threshold)
  reservoir_end    untrained neurons (top-100 share < 0.35) at the last snapshot
  hijack_max       largest hijacker index over the run (plateau present <=> ~0.5; absent <=> stays ~0.1-0.2)
  theta_med_end    median theta at the last snapshot (mV); theta runaway shows up here
  minutes          wall-clock

Figures (results/):
  S1_accuracy_overlay.png   held-out curves of every run; colour = theta_plus, solid = uniform init, dashed = imprint
  S2_ceiling_vs_theta_plus  final_acc against theta_plus (one point per run, means joined); imprint runs as a band
  S3_speed_vs_theta_plus    t75 against theta_plus, same layout
  S4_tradeoff               final_acc against t75: the convergence-speed / accuracy trade-off, one point per run

USAGE  python summarize_sweep.py [--results DIR] [--level 75]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_plots import spike_time_from_curve, load_run  # noqa: E402

def _k(x):
    return "none" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x / 1000:.0f}k"

def summarise_run(run_dir: Path, level: float) -> dict | None:
    try:
        d, cfg = load_run(run_dir)
    except Exception as e:  # unfinished run
        print(f"skip {run_dir.name}: {e}"); return None
    pts, acc = d["points"], d["accuracy"]
    if len(pts) < 3: return None
    i30 = int(np.searchsorted(pts, 30_000))
    th_end = d["theta_snap"][-1]
    return {
        "run": run_dir.name, "theta_plus": cfg["theta_plus"], "inhibition_mV": cfg["inhibition_mV"], "init": cfg.get("init", "uniform"),
        "theta_init": cfg.get("theta_init"), "epochs": cfg["num_epochs"], "seed": cfg["seed"], "examples": int(pts[-1]),
        "final_acc": float(np.mean(acc[-3:])), "max_acc": float(np.max(acc)),
        "t75": spike_time_from_curve(pts, acc, level), "t78": spike_time_from_curve(pts, acc, 78.0),
        "acc_at_30k": float(acc[min(i30, len(acc) - 1)]),
        "reservoir_end": int(d["reservoir_count"][-1]), "hijack_max": float(np.nanmax(d["hijack_index"])),
        "theta_med_end": float(np.median(th_end)), "minutes": cfg.get("minutes", float("nan")),
        "_pts": pts, "_acc": acc,
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--level", type=float, default=75.0, help="accuracy level for the convergence time t75 (default 75)")
    a = ap.parse_args()
    root = Path(a.results)
    runs = [r for r in sorted(root.iterdir()) if r.is_dir() and "SMOKE" not in r.name and (r / "run.json").exists()]
    rows = [x for x in (summarise_run(r, a.level) for r in runs) if x]
    if not rows:
        print("no finished runs under", root); return
    cols = [c for c in rows[0] if not c.startswith("_")]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in cols})
    print(f"{'run':62s} {'tp':>5s} {'inh':>5s} {'init':>8s} {'seed':>4s} {'final':>6s} {'max':>6s} {'t'+str(int(a.level)):>5s} {'t78':>5s} {'res':>4s} {'hjmax':>5s} {'thmed':>6s}")
    for r in rows:
        print(f"{r['run'][:62]:62s} {r['theta_plus']:5g} {r['inhibition_mV']:5g} {r['init']:>8s} {r['seed']:4d} {r['final_acc']:6.1f} {r['max_acc']:6.1f} "
              f"{_k(r['t75']):>5s} {_k(r['t78']):>5s} {r['reservoir_end']:4d} {r['hijack_max']:5.2f} {r['theta_med_end']:6.1f}")

    tps = sorted({r["theta_plus"] for r in rows})
    cmap = plt.get_cmap("viridis"); color = {tp: cmap(i / max(1, len(tps) - 1)) for i, tp in enumerate(tps)}
    def style(r):
        return dict(color=color[r["theta_plus"]], linestyle="--" if r["init"] == "imprint" else "-",
                    label=f"θ+ {r['theta_plus']:g}, {r['init']}" + (f", inh {r['inhibition_mV']:g}" if r["inhibition_mV"] != 200 else "") + f" (s{r['seed']})")

    # S1 overlay
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rows: ax.plot(r["_pts"], r["_acc"], linewidth=1.4, alpha=0.9, **style(r))
    ax.axhline(a.level, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Number of training examples"); ax.set_ylabel("Held-out accuracy (%)"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Held-out accuracy, every sweep run", loc="left", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right"); fig.tight_layout(); fig.savefig(root / "S1_accuracy_overlay.png", dpi=150); plt.close(fig)

    # S2 / S3: metric against theta_plus (uniform init), imprint runs drawn as horizontal band(s)
    for fname, key, ylabel, title in [("S2_ceiling_vs_theta_plus.png", "final_acc", "Final held-out accuracy (%)", "Accuracy ceiling against θ+"),
                                      ("S3_speed_vs_theta_plus.png", "t75", f"Examples to reach and hold {a.level:g}% (k)", f"Convergence time against θ+")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        uni = [r for r in rows if r["init"] == "uniform" and r["inhibition_mV"] == 200]
        imp = [r for r in rows if r["init"] == "imprint"]
        other = [r for r in rows if r not in uni and r not in imp]
        def val(r):
            v = r[key]; return v / 1000 if key == "t75" else v
        if uni:
            ax.scatter([r["theta_plus"] for r in uni], [val(r) for r in uni], color="tab:blue", zorder=3, label="uniform init, 200 mV (one point per seed)")
            for tp in tps:
                vs = [val(r) for r in uni if r["theta_plus"] == tp and not (isinstance(val(r), float) and np.isnan(val(r)))]
                if vs: ax.plot([tp], [np.mean(vs)], marker="_", markersize=18, color="tab:blue")
            means = [(tp, np.nanmean([val(r) for r in uni if r["theta_plus"] == tp])) for tp in tps if any(r["theta_plus"] == tp for r in uni)]
            ax.plot([m[0] for m in means], [m[1] for m in means], color="tab:blue", linewidth=1, alpha=0.6)
        for r in imp:
            v = val(r)
            if not (isinstance(v, float) and np.isnan(v)):
                ax.axhline(v, color="tab:red", linestyle="--", linewidth=1, alpha=0.8, label=f"imprint at zero, θ+ {r['theta_plus']:g} (s{r['seed']})")
        if other:
            ax.scatter([r["theta_plus"] for r in other], [val(r) for r in other], color="tab:green", marker="s", zorder=3, label="other inhibition / θ init")
        ax.set_xscale("log"); ax.set_xticks(tps); ax.set_xticklabels([f"{t:g}" for t in tps])
        ax.set_xlabel("θ+ (per-spike threshold increment, applied as sum/16)"); ax.set_ylabel(ylabel); ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(title, loc="left", fontweight="bold"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(root / fname, dpi=150); plt.close(fig)

    # S4 trade-off
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        t = r["t75"]
        x = (r["examples"] + 5000) / 1000 if (isinstance(t, float) and np.isnan(t)) else t / 1000   # never reached: park right of the run length
        ax.scatter(x, r["final_acc"], color=color[r["theta_plus"]], marker="^" if r["init"] == "imprint" else ("s" if r["inhibition_mV"] != 200 else "o"), s=50, zorder=3)
        lab = f"θ+{r['theta_plus']:g}" + (" imp" if r["init"] == "imprint" else "") + ("" if r["inhibition_mV"] == 200 else f" inh{r['inhibition_mV']:g}")
        ax.annotate(lab, (x, r["final_acc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"Examples to reach and hold {a.level:g}% (k)   [runs that never do are parked right of their length]")
    ax.set_ylabel("Final held-out accuracy (%)"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Speed / accuracy trade-off, one point per run", loc="left", fontweight="bold")
    fig.text(0.995, 0.004, "circle = uniform init · triangle = imprint at zero · square = other inhibition / θ init · colour = θ+", ha="right", va="bottom", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(root / "S4_tradeoff.png", dpi=150); plt.close(fig)
    print("wrote", root / "summary.csv", "and S1..S4")

if __name__ == "__main__":
    main()
