"""
summarize_test3.py -- collect every finished test3.py run under results/ (SMOKE runs ignored) into one table and the
cross-run figures of the continual-learning test. Torch-free: reads only run.npz / run.json.

Per run (summary.csv); "old" = the first phase's classes, "new" = every later phase's classes, "boundary" = the first
phase change:
  acc_old_boundary   mean old-class accuracy at the last snapshot before the boundary (what there was to forget)
  acc_old_end        mean old-class accuracy over the last 3 snapshots
  forgetting         acc_old_boundary - acc_old_end (points)
  acc_new_end        mean new-class accuracy over the last 3 snapshots                        (plasticity: how much)
  t_new              examples after the boundary until new-class accuracy first reaches 80% of acc_new_end  (plasticity: how fast)
  acc_all_end        all-class accuracy over the last 3 snapshots
  repurposed_frac    of the neurons labelled with an OLD class at the boundary, the fraction labelled with a NEW class at the end
  new_from_reserve   new-class neurons at the end that were ever in the reserve (allocated)
  new_repurposed     new-class neurons at the end that were never in the reserve (overwritten old neurons)
  reserve_boundary   reserve size at the boundary; recruits_after: recruits out of the reserve after it; recycled: total recycled
  fs_old_b, fs_new_b first-spike latency (ms) of the trained population on old / new classes at the boundary (gate placement check)
  minutes

Figures (results/):
  S1_old_new_accuracy.png   old-class accuracy (top) and new-class accuracy (bottom) for every run
  S2_stability_plasticity.png  one point per run: new-class accuracy at the end (x) vs old-class accuracy at the end (y)
  S3_allocation.png         per run: new-class neurons that were allocated from the reserve vs repurposed

USAGE  python summarize_test3.py [--results DIR]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test3_plots import load_run  # noqa: E402

def _k(x):
    return "none" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x / 1000:.0f}k"

def variant_label(cfg) -> str:
    K = cfg.get("reserve", 0)
    lab = "V1 no reserve" if not K else (f"V3 reserve {K} rep" if cfg.get("replenish") else f"V2 reserve {K}")
    if cfg.get("init") == "uniform": lab = "baseline" if cfg.get("theta_plus_arg", cfg["theta_plus"]) < 0.1 else lab + " uniform"
    lab += f" θ+{cfg.get('theta_plus_arg', cfg['theta_plus']):g}"
    lab += f" τ{cfg['theta_tau'] / 1000:g}k" if cfg.get("theta_tau") is not None else " nodecay"
    if K and (cfg.get("reserve_delay_ms", 10) != 10 or cfg.get("reserve_theta", "pin") != "pin"):
        lab += f" gate{cfg.get('reserve_delay_ms'):g}ms/{cfg.get('reserve_theta')}"
    if cfg.get("phases", "0-4,5-9") != "0-4,5-9": lab += f" [{cfg.get('phases')}]"
    return lab

def summarise_run(run_dir: Path) -> dict | None:
    try:
        d, cfg = load_run(run_dir)
    except Exception as e:
        print(f"skip {run_dir.name}: {e}"); return None
    if "first_spike_mean" not in d: print(f"skip {run_dir.name}: not a test3 run"); return None
    pts, acc = d["points"], d["accuracy"]
    if len(pts) < 3: return None
    phases = cfg.get("phase_classes") or [list(range(10))]
    old = list(phases[0]); new = [c for ph in phases[1:] for c in ph]
    conf = d["conf_snap"].astype(float); rows = conf.sum(2)
    acc_c = np.where(rows > 0, 100.0 * np.einsum("sii->si", conf) / np.maximum(rows, 1), np.nan)
    acc_old = np.nanmean(acc_c[:, old], axis=1)
    acc_new = np.nanmean(acc_c[:, new], axis=1) if new else np.full(len(pts), np.nan)
    bounds = cfg.get("phase_boundaries") or []
    ib = int(np.searchsorted(pts, bounds[0])) if bounds else len(pts)      # first snapshot at/after the boundary
    ib_before = max(0, ib - 1)
    asg = d["assignments_snap"].astype(int); ever = d["ever_reserve"].astype(bool)
    old_at_b = np.isin(asg[ib_before], old); new_at_end = np.isin(asg[-1], new)
    repurposed_frac = float((old_at_b & new_at_end).sum() / max(1, old_at_b.sum())) if bounds else float("nan")
    acc_new_end = float(np.nanmean(acc_new[-3:])) if new else float("nan")
    t_new = float("nan")
    if new and bounds and not np.isnan(acc_new_end):
        after = np.where((pts >= bounds[0]) & (acc_new >= 0.8 * acc_new_end))[0]
        if len(after): t_new = float(pts[after[0]] - bounds[0])
    ev = d["reserve_recruit_events"].reshape(-1, 2)
    fs = d["first_spike_mean"]
    return {
        "run": run_dir.name, "variant": variant_label(cfg), "seed": cfg["seed"], "examples": int(pts[-1]),
        "acc_old_boundary": float(acc_old[ib_before]) if bounds else float("nan"),
        "acc_old_end": float(np.nanmean(acc_old[-3:])),
        "forgetting": float(acc_old[ib_before] - np.nanmean(acc_old[-3:])) if bounds else float("nan"),
        "acc_new_end": acc_new_end, "t_new": t_new, "acc_all_end": float(np.mean(acc[-3:])),
        "repurposed_frac": repurposed_frac,
        "new_from_reserve": int((new_at_end & ever).sum()), "new_repurposed": int((new_at_end & ~ever).sum()),
        "reserve_boundary": int(d["reserve_count"][ib_before]) if bounds else int(d["reserve_count"][-1]),
        "recruits_after": int((ev[:, 0] >= bounds[0]).sum()) if (bounds and len(ev)) else 0,
        "recycled": int(d["recycled_cum"][-1]),
        "fs_old_b": float(np.nanmean(fs[ib_before, old])), "fs_new_b": float(np.nanmean(fs[ib_before, new])) if new else float("nan"),
        "minutes": cfg.get("minutes", float("nan")),
        "_pts": pts, "_acc_old": acc_old, "_acc_new": acc_new, "_bounds": bounds,
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()
    root = Path(a.results)
    runs = [r for r in sorted(root.iterdir()) if r.is_dir() and "SMOKE" not in r.name and (r / "run.json").exists()]
    rows = [x for x in (summarise_run(r) for r in runs) if x]
    if not rows:
        print("no finished runs under", root); return
    rows.sort(key=lambda r: (r["variant"], r["seed"]))
    cols = [c for c in rows[0] if not c.startswith("_")]
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in cols})
    print(f"{'variant':34s} {'seed':>4s} {'old@b':>6s} {'oldEnd':>6s} {'forget':>6s} {'newEnd':>6s} {'t_new':>5s} {'allEnd':>6s} {'repurp':>6s} {'alloc':>5s} {'rep':>4s} {'res@b':>5s} {'recr':>4s} {'recyc':>5s} {'fsOld':>5s} {'fsNew':>5s}")
    for r in rows:
        print(f"{r['variant'][:34]:34s} {r['seed']:4d} {r['acc_old_boundary']:6.1f} {r['acc_old_end']:6.1f} {r['forgetting']:6.1f} {r['acc_new_end']:6.1f} {_k(r['t_new']):>5s} "
              f"{r['acc_all_end']:6.1f} {r['repurposed_frac']:6.2f} {r['new_from_reserve']:5d} {r['new_repurposed']:4d} {r['reserve_boundary']:5d} {r['recruits_after']:4d} {r['recycled']:5d} "
              f"{r['fs_old_b']:5.1f} {r['fs_new_b']:5.1f}")

    variants = sorted({r["variant"] for r in rows})
    cmap = plt.get_cmap("tab10"); color = {v: cmap(i % 10) for i, v in enumerate(variants)}
    def style(r):
        return dict(color=color[r["variant"]], linestyle="-" if r["seed"] == 0 else "--", linewidth=1.5, alpha=0.9, label=f"{r['variant']} (s{r['seed']})")

    # S1 old / new accuracy
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for r in rows:
        ax.plot(r["_pts"], r["_acc_old"], **style(r)); bx.plot(r["_pts"], r["_acc_new"], **style(r))
    for b in (rows[0]["_bounds"] or []):
        ax.axvline(b, color="black", linewidth=1); bx.axvline(b, color="black", linewidth=1)
    ax.set_ylabel("Old-class accuracy (%)"); ax.set_ylim(0, 100); ax.grid(True, linestyle="--", alpha=0.5); ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_title("Old classes: stability (should stay flat after the boundary)", loc="left", fontweight="bold")
    bx.set_ylabel("New-class accuracy (%)"); bx.set_ylim(0, 100); bx.set_xlabel("Number of training examples"); bx.grid(True, linestyle="--", alpha=0.5)
    bx.set_title("New classes: plasticity (how fast and how high after the boundary)", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(root / "S1_old_new_accuracy.png", dpi=150); plt.close(fig)

    # S2 stability-plasticity plane
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in rows:
        if np.isnan(r["acc_new_end"]): continue
        ax.scatter(r["acc_new_end"], r["acc_old_end"], color=color[r["variant"]], marker="o" if r["seed"] == 0 else "s", s=60, zorder=3)
        ax.annotate(f"{r['variant']} s{r['seed']}", (r["acc_new_end"], r["acc_old_end"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("New-class accuracy at the end (%)  →  plasticity"); ax.set_ylabel("Old-class accuracy at the end (%)  →  stability")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Stability-plasticity plane, one point per run", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(root / "S2_stability_plasticity.png", dpi=150); plt.close(fig)

    # S3 allocation vs repurposing
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(rows) + 3), 5))
    xs = np.arange(len(rows))
    ax.bar(xs, [r["new_from_reserve"] for r in rows], color="tab:green", label="new-class neurons from the reserve (allocated)")
    ax.bar(xs, [r["new_repurposed"] for r in rows], bottom=[r["new_from_reserve"] for r in rows], color="tab:red", alpha=0.8, label="new-class neurons never in the reserve (repurposed)")
    ax.set_xticks(xs); ax.set_xticklabels([f"{r['variant']} s{r['seed']}" for r in rows], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Neurons labelled with a new class at the end"); ax.grid(True, axis="y", linestyle="--", alpha=0.5); ax.legend(fontsize=8)
    ax.set_title("Where the new classes' neurons came from", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(root / "S3_allocation.png", dpi=150); plt.close(fig)
    print("wrote", root / "summary.csv", "and S1..S3")

if __name__ == "__main__":
    main()
