"""
summarize_test2.py -- collect every finished test2.py run under results/ (SMOKE runs ignored) into one table and the
cross-run figures for this round. Torch-free: reads only run.npz / run.json.

Per run (summary.csv):
  variant          short label built from the knobs, e.g. "θ+0.5 τ30k", "θ+0.5 max0", "θ+0.5→0.05@15k", "θ+1 no decay"
  final_acc        held-out accuracy averaged over the last 3 snapshots
  max_acc          best single snapshot
  acc_60k          accuracy at ~60k (end of epoch 1) -- compare with final_acc to see whether a run DECLINED
  t75 / t78        first snapshot after which accuracy never drops below 75% / 78% again (NaN = never)
  spk_20k, spk_60k, spk_end     output spikes per presentation (window spikes / 3200) at ~20k, ~60k and the end
  theta_med_60k, theta_med_end  median theta (mV); stationary if the two are close
  theta_spread_end              p90 - p10 of theta at the end (mV); the clamp is expected to compress this
  drift_last_60k                change of the median theta over the last 60k examples, mV per 1k (0 = stationary)
  minutes

Figures (results/):
  S1_accuracy_overlay.png        held-out curves of every run
  S2_spikes_per_presentation.png the collapse readout: spikes per presentation vs examples
  S3_theta.png                   two panels: median theta vs examples; p10-p90 spread vs examples
  S4_ceiling_bars.png            final accuracy per variant (mean over seeds, individual seeds as dots), with acc_60k
                                 drawn as a hollow marker so a decline shows as a gap

USAGE  python summarize_test2.py [--results DIR]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test2_plots import spike_time_from_curve, load_run, moving_average  # noqa: E402

def _k(x):
    return "none" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x / 1000:.0f}k"

def variant_label(cfg) -> str:
    tp0 = cfg.get("theta_plus_arg", cfg["theta_plus"])
    lab = f"θ+{tp0:g}"
    if cfg.get("anneal_at") is not None: lab += f"→{cfg['theta_plus_late']:g}@{cfg['anneal_at'] // 1000}k"
    lab += f" τ{cfg['theta_tau'] / 1000:g}k" if cfg.get("theta_tau") is not None else " no decay"
    if cfg.get("theta_max") is not None: lab += f" max{cfg['theta_max']:g}"
    if cfg.get("inhibition_mV", 200) != 200: lab += f" inh{cfg['inhibition_mV']:g}"
    if cfg.get("init", "uniform") != "uniform": lab += f" {cfg['init']}"
    return lab

def at(pts, series, x):
    """value of series at the first snapshot >= x (or the last one)."""
    i = int(np.searchsorted(pts, x)); return float(series[min(i, len(series) - 1)])

def summarise_run(run_dir: Path) -> dict | None:
    try:
        d, cfg = load_run(run_dir)
    except Exception as e:
        print(f"skip {run_dir.name}: {e}"); return None
    pts, acc = d["points"], d["accuracy"]
    if len(pts) < 3: return None
    spk = d["window_spikes"].sum(1) / (cfg["snapshot_interval"] * cfg["batch_size"])   # spikes per presentation
    th = d["theta_snap"]; med = np.median(th, axis=1); spread = np.percentile(th, 90, axis=1) - np.percentile(th, 10, axis=1)
    last = pts[-1]; i_prev = int(np.searchsorted(pts, last - 60_000))
    drift = (med[-1] - med[i_prev]) / max(1e-9, (pts[-1] - pts[i_prev]) / 1000) if pts[-1] > pts[i_prev] else float("nan")
    return {
        "run": run_dir.name, "variant": variant_label(cfg), "seed": cfg["seed"], "epochs": cfg["num_epochs"], "examples": int(last),
        "final_acc": float(np.mean(acc[-3:])), "max_acc": float(np.max(acc)), "acc_60k": at(pts, acc, 60_000),
        "t75": spike_time_from_curve(pts, acc, 75.0), "t78": spike_time_from_curve(pts, acc, 78.0),
        "spk_20k": at(pts, spk, 20_000), "spk_60k": at(pts, spk, 60_000), "spk_end": float(spk[-1]),
        "theta_med_60k": at(pts, med, 60_000), "theta_med_end": float(med[-1]), "theta_spread_end": float(spread[-1]),
        "drift_last_60k": float(drift), "minutes": cfg.get("minutes", float("nan")),
        "_pts": pts, "_acc": acc, "_spk": spk, "_med": med, "_spread": spread,
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
    print(f"{'variant':24s} {'seed':>4s} {'final':>6s} {'@60k':>6s} {'t75':>5s} {'spk20k':>6s} {'spk60k':>6s} {'spkEnd':>6s} {'θ60k':>6s} {'θEnd':>6s} {'spread':>6s} {'drift':>6s}")
    for r in rows:
        print(f"{r['variant']:24s} {r['seed']:4d} {r['final_acc']:6.1f} {r['acc_60k']:6.1f} {_k(r['t75']):>5s} {r['spk_20k']:6.1f} {r['spk_60k']:6.1f} {r['spk_end']:6.1f} "
              f"{r['theta_med_60k']:6.1f} {r['theta_med_end']:6.1f} {r['theta_spread_end']:6.1f} {r['drift_last_60k']:6.2f}")

    variants = sorted({r["variant"] for r in rows})
    cmap = plt.get_cmap("tab10"); color = {v: cmap(i % 10) for i, v in enumerate(variants)}
    def style(r):
        return dict(color=color[r["variant"]], linestyle="-" if r["seed"] == 0 else "--", linewidth=1.4, alpha=0.9,
                    label=f"{r['variant']} (s{r['seed']})")

    # S1 accuracy
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rows: ax.plot(r["_pts"], r["_acc"], **style(r))
    ax.axhline(78, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Number of training examples"); ax.set_ylabel("Held-out accuracy (%)"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Held-out accuracy, every test2 run", loc="left", fontweight="bold"); ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(root / "S1_accuracy_overlay.png", dpi=150); plt.close(fig)

    # S2 spikes per presentation
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rows: ax.plot(r["_pts"], r["_spk"], **style(r))
    ax.set_xlabel("Number of training examples"); ax.set_ylabel("Output spikes per presentation (window total / 3,200)"); ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.set_title("Spikes per presentation: falling = θ outrunning the input drive; flat = stationary", loc="left", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right"); fig.tight_layout(); fig.savefig(root / "S2_spikes_per_presentation.png", dpi=150); plt.close(fig)

    # S3 theta median + spread
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for r in rows:
        ax.plot(r["_pts"], r["_med"], **style(r)); bx.plot(r["_pts"], r["_spread"], **style(r))
    ax.axhline(-52, color="gray", linewidth=0.8, label="θ_base = −52 mV")
    ax.set_ylabel("Median θ (mV)"); ax.grid(True, linestyle="--", alpha=0.5); ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Median θ: drifting or stationary?", loc="left", fontweight="bold")
    bx.set_ylabel("θ spread, p90 − p10 (mV)"); bx.set_xlabel("Number of training examples"); bx.grid(True, linestyle="--", alpha=0.5)
    bx.set_title("θ spread across neurons: the clamp should compress it, decay should not", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(root / "S3_theta.png", dpi=150); plt.close(fig)

    # S4 ceiling bars
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(variants) + 3), 5))
    xs = np.arange(len(variants))
    for i, v in enumerate(variants):
        rs = [r for r in rows if r["variant"] == v]
        ax.bar(i, np.mean([r["final_acc"] for r in rs]), width=0.6, color=color[v], alpha=0.35)
        ax.scatter([i] * len(rs), [r["final_acc"] for r in rs], color=color[v], zorder=3, label="final (last 3 snapshots)" if i == 0 else None)
        ax.scatter([i] * len(rs), [r["acc_60k"] for r in rs], facecolors="none", edgecolors=color[v], zorder=3, label="at 60k" if i == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
    lo = min(min(r["final_acc"], r["acc_60k"]) for r in rows); ax.set_ylim(max(0, lo - 10), 92)
    ax.set_ylabel("Held-out accuracy (%)"); ax.grid(True, axis="y", linestyle="--", alpha=0.5); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ceiling per variant (bar = mean over seeds; hollow = accuracy at 60k, so a gap below the dot = decline)", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig(root / "S4_ceiling_bars.png", dpi=150); plt.close(fig)
    print("wrote", root / "summary.csv", "and S1..S4")

if __name__ == "__main__":
    main()
