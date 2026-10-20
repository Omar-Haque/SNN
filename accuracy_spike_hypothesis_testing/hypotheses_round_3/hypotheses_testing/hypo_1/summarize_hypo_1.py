"""
summarize_hypo_1.py -- overlay every finished hypo_1 run and extract the numbers the predictions are about.

    python summarize_hypo_1.py                 # looks in ./results
    python summarize_hypo_1.py path/to/results

Outputs (written next to the run folders, in results/):
    summary.csv                  one row per run: variation, n_refill, intervention step, seed, spike time, floor, ceiling, ...
    S1_accuracy_overlay.png      held-out accuracy of all runs, refill-type variations left, drain-type variations right
    S2_spike_time.png            left: spike time vs. number of refilled neurons, with the prediction line
                                 right: spike time per variation (baseline / pseudo_refill / sabotage / imprint)
    S3_reservoir_overlay.png     reservoir count of all runs
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # hypo_1_plots.py sits next to this file; importable from any CWD
from hypo_1_plots import load_run, spike_time_from_curve, SPIKE_LEVEL

ARM_COLORS = {'baseline': 'black', 'refill': 'tab:red', 'refill_sham': 'tab:pink', 'disarm': 'tab:green', 'imprint': 'tab:olive'}
NAME = {'baseline': 'baseline', 'refill': 'refill', 'refill_sham': 'pseudo refill', 'disarm': 'sabotage', 'imprint': 'imprint'}   # display names


def summarise_run(run_dir):
    d, cfg = load_run(run_dir)
    pts, acc, res = d['points'], d['accuracy'], d['reservoir_count']
    spike = spike_time_from_curve(pts, acc)
    at = cfg.get('intervention_step_actual')
    at = np.nan if at is None else float(at)
    # floor: mean held-out accuracy from 20k up to 5k before the spike (or to the end if no spike)
    hi = spike - 5000 if spike == spike else pts[-1]
    pre = (pts >= 20000) & (pts < hi)
    floor = float(acc[pre].mean()) if pre.any() else np.nan
    post = pts > (spike + 10000) if spike == spike else np.zeros_like(pts, bool)
    ceiling = float(acc[post].mean()) if post.any() else np.nan
    # reservoir just before and just after the intervention
    res_before = res_after = np.nan
    if at == at:
        b = np.where(pts <= at)[0]; a = np.where(pts > at)[0]
        if len(b): res_before = int(res[b[-1]])
        if len(a): res_after = int(res[a[0]])
    # drain rate (neurons per 1k examples) between 100k and the spike, for baseline runs
    rate = np.nan
    if cfg['arm'] == 'baseline' and spike == spike:
        m = (pts >= 100000) & (pts <= spike) & (res > 0)
        if m.sum() >= 3:
            slope = np.polyfit(pts[m], res[m], 1)[0]
            rate = -slope * 1000.0
    return dict(run=Path(run_dir).name, arm=cfg['arm'], variation=NAME[cfg['arm']], n_refill=cfg['n_refill'] if cfg['arm'] in ('refill', 'refill_sham') else 0,
                intervention_at=at, seed=cfg['seed'], spike_time=spike, floor=floor, ceiling=ceiling,
                reservoir_before=res_before, reservoir_after=res_after, drain_rate_per_1k=rate, minutes=cfg.get('minutes', np.nan),
                smoke=cfg.get('smoke', False), _d=d, _cfg=cfg)


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results'
    runs = sorted(p for p in results.iterdir() if (p / 'run.npz').exists() and (p / 'run.json').exists()) if results.exists() else []
    rows = [summarise_run(p) for p in runs]
    rows = [r for r in rows if not r['smoke']]
    if not rows:
        print(f"no finished (non-smoke) runs under {results}"); return

    # ---- table ---------------------------------------------------------------------------------------------
    cols = ['run', 'variation', 'n_refill', 'intervention_at', 'seed', 'spike_time', 'floor', 'ceiling', 'reservoir_before', 'reservoir_after', 'drain_rate_per_1k', 'minutes']
    with open(results / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows: w.writerow([r[c] for c in cols])
    print(f"{'variation':<15}{'n':>4}{'at':>9}{'seed':>5}{'spike':>9}{'floor':>7}{'ceil':>7}{'res<':>6}{'res>':>6}{'rate':>7}")
    for r in rows:
        f = lambda v, p=0: '   -  ' if v != v else f"{v:.{p}f}"
        print(f"{r['variation']:<15}{r['n_refill']:>4}{f(r['intervention_at']):>9}{r['seed']:>5}{f(r['spike_time']):>9}{f(r['floor'],1):>7}{f(r['ceiling'],1):>7}{f(r['reservoir_before']):>6}{f(r['reservoir_after']):>6}{f(r['drain_rate_per_1k'],2):>7}")

    base = [r for r in rows if r['arm'] == 'baseline' and r['spike_time'] == r['spike_time']]
    T0 = np.mean([r['spike_time'] for r in base]) if base else np.nan
    rates = [r['drain_rate_per_1k'] for r in base if r['drain_rate_per_1k'] == r['drain_rate_per_1k']]
    rate = np.mean(rates) if rates else np.nan
    print(f"\nbaseline spike time: {T0:.0f}   baseline drain rate: {rate:.2f} neurons per 1k examples" if base else "\n(no baseline run yet)")

    # ---- S1 accuracy overlay -------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    for ax, fam, title in ((axes[0], ('baseline', 'refill', 'refill_sham'), 'Held-out accuracy: refill variations'), (axes[1], ('baseline', 'disarm', 'imprint'), 'Held-out accuracy: drain variations')):
        for r in rows:
            if r['arm'] not in fam: continue
            d = r['_d']; lab = f"{r['variation']}" + (f" n={r['n_refill']}" if r['n_refill'] else '') + f" s{r['seed']}"
            ax.plot(d['points'], d['accuracy'], color=ARM_COLORS[r['arm']], alpha=0.8 if r['arm'] != 'baseline' else 0.5, linewidth=1.5, label=lab)
            if r['intervention_at'] == r['intervention_at']:
                ax.axvline(r['intervention_at'], color=ARM_COLORS[r['arm']], linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(SPIKE_LEVEL, color='green', linewidth=0.6, alpha=0.4)
        ax.set_title(title, loc='left', fontweight='bold'); ax.set_xlabel('Number of training examples'); ax.grid(True, linestyle='--', alpha=0.5); ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('Held-out accuracy (%)')
    fig.tight_layout(); fig.savefig(results / 'S1_accuracy_overlay.png', dpi=150); plt.close(fig)

    # ---- S2 spike time -------------------------------------------------------------------------------------
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.5))
    for r in rows:
        if r['arm'] in ('baseline', 'refill', 'refill_sham'):
            ax.scatter(r['n_refill'], r['spike_time'] / 1000, s=70, color=ARM_COLORS[r['arm']], marker='o' if r['arm'] != 'refill_sham' else 'D',
                       facecolors='none' if r['arm'] == 'refill_sham' else ARM_COLORS[r['arm']], label=r['variation'])
    if T0 == T0 and rate == rate and rate > 0:
        ns = np.linspace(0, max([r['n_refill'] for r in rows] + [100]), 50)
        ax.plot(ns, (T0 + ns / rate * 1000) / 1000, color='red', linestyle='--', label=f'expected if delay = n / baseline drain rate ({rate:.2f} per 1k)')
    ax.set_xlabel('Neurons re-initialised at the intervention'); ax.set_ylabel('Spike time (k examples)'); ax.grid(True, linestyle='--', alpha=0.5)
    h, l = ax.get_legend_handles_labels(); uniq = dict(zip(l, h)); ax.legend(uniq.values(), uniq.keys(), fontsize=9)
    ax.set_title('Spike time vs. number of re-initialised neurons', loc='left', fontweight='bold')
    order = ['baseline', 'refill_sham', 'disarm', 'imprint']
    for i, arm in enumerate(order):
        ys = [r['spike_time'] / 1000 for r in rows if r['arm'] == arm]
        xs = np.full(len(ys), i) + np.linspace(-0.1, 0.1, len(ys)) if ys else []
        bx.scatter(xs, ys, s=70, color=ARM_COLORS[arm])
        nos = sum(1 for r in rows if r['arm'] == arm and r['spike_time'] != r['spike_time'])
        if nos: bx.text(i, 0.5, f'{nos} run(s)\nno spike', ha='center', fontsize=8, color=ARM_COLORS[arm])
    if T0 == T0: bx.axhline(T0 / 1000, color='black', linestyle='--', linewidth=0.8, label='baseline mean')
    bx.set_xticks(range(len(order))); bx.set_xticklabels([NAME[a] for a in order]); bx.set_ylabel('Spike time (k examples)'); bx.grid(True, linestyle='--', alpha=0.5)
    bx.set_title('Spike time by variation', loc='left', fontweight='bold'); bx.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(results / 'S2_spike_time.png', dpi=150); plt.close(fig)

    # ---- S3 reservoir overlay ------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in rows:
        d = r['_d']; lab = f"{r['variation']}" + (f" n={r['n_refill']}" if r['n_refill'] else '') + f" s{r['seed']}"
        ax.plot(d['points'], d['reservoir_count'], color=ARM_COLORS[r['arm']], alpha=0.8, linewidth=1.5, label=lab)
        if r['spike_time'] == r['spike_time']:
            ax.axvline(r['spike_time'], color=ARM_COLORS[r['arm']], linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('Untrained neurons (reservoir)'); ax.grid(True, linestyle='--', alpha=0.5); ax.legend(fontsize=8)
    ax.set_title("Untrained neurons, all runs (dotted: each run's spike time)", loc='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(results / 'S3_reservoir_overlay.png', dpi=150); plt.close(fig)
    print(f"\nwrote {results / 'summary.csv'}, S1_accuracy_overlay.png, S2_spike_time.png, S3_reservoir_overlay.png")


if __name__ == '__main__':
    main()
