"""
sweep_plots.py (copy of hypo_1_plots.py) -- figures for a single hypo_1 run, and shared helpers used by summarize_hypo_1.py.

Pure numpy + matplotlib (no torch), so figures can be regenerated from a finished run with
    python hypo_1.py --replot results/<run_dir>
Every figure is described in README.md ("How to read the figures").
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Spike time = first snapshot after which held-out accuracy never drops below SPIKE_LEVEL again. 78 sits between the
# plateau's best snapshots (~74) and the post-spike floor (~83); 'never again' guards against a single lucky snapshot.
SPIKE_LEVEL = 78.0          # held-out accuracy (%) that must be reached AND never lost again
YOUNG_AGE = 10_000          # a neuron recruited less than this many examples ago counts as "young"
TOP_WINNERS = 5


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------
def spike_time_from_curve(points, acc, level=SPIKE_LEVEL):
    """First snapshot after which held-out accuracy never drops below `level` again. NaN if there is none
    (or if that only happens at the very last snapshot, which cannot be distinguished from a lucky point)."""
    points, acc = np.asarray(points, float), np.asarray(acc, float)
    if len(acc) < 3:
        return float('nan')
    for i in range(len(acc) - 1):
        if np.all(acc[i:] >= level):
            return float(points[i])
    return float('nan')


def moving_average(y, k):
    y = np.asarray(y, float)
    if len(y) < k or k <= 1:
        return y
    return np.convolve(y, np.ones(k) / k, mode='same')


def load_run(run_dir):
    run_dir = Path(run_dir)
    d = dict(np.load(run_dir / 'run.npz', allow_pickle=False))
    cfg = json.loads((run_dir / 'run.json').read_text())
    return d, cfg


def _k(x):
    """format a number of examples as '110k'."""
    return f"{x / 1000:.0f}k"


def _vlines(ax, cfg, spike_time):
    at = cfg.get('intervention_step_actual')
    if at is not None and not (isinstance(at, float) and np.isnan(at)):
        ax.axvline(at, color='red', linestyle='--', linewidth=1.2, label=f"intervention at {_k(at)}")
    if spike_time == spike_time:  # not NaN
        ax.axvline(spike_time, color='green', linestyle=':', linewidth=1.5, label=f"spike at {_k(spike_time)}")


def _run_desc(cfg):
    """Short run descriptor for the right-hand title, e.g. 'refill: 50 neurons re-initialised at 110k, θ = θ_base  ·  seed 0'."""
    if cfg.get('sweep'):
        # sweep.py runs: describe the architecture knobs instead of the (absent) intervention.
        ti = cfg.get('theta_init'); ti_s = f", θ init {ti:g} mV" if ti is not None else ''
        ep = cfg.get('num_epochs'); ep_s = f", {ep} epoch{'s' if ep != 1 else ''}" if ep is not None else ''
        return f"θ+ = {cfg['theta_plus']:g} (sum/16), inhibition {cfg['inhibition_mV']:g} mV, init {cfg['init']}{ti_s}{ep_s}  ·  seed {cfg['seed']}"
    arm = cfg['arm']; at = cfg.get('intervention_step_actual')
    at_s = _k(at) if at is not None else ''
    if arm == 'baseline':
        desc = 'baseline (no intervention)'
    elif arm == 'refill':
        desc = f"refill: {cfg['n_refill']} neurons re-initialised at {at_s}, θ = θ_base"
    elif arm == 'refill_sham':
        desc = f"pseudo refill: {cfg['n_refill']} neurons re-initialised at {at_s}, θ = trained median"
    elif arm == 'disarm':
        desc = f"sabotage: untrained neurons' θ raised to the trained median at {at_s}"
    elif arm == 'imprint':
        desc = f"imprint: untrained neurons given random image templates at {at_s}"
    else:
        desc = arm
    return f"{desc}  ·  seed {cfg['seed']}"


def _titles(ax, what, cfg):
    """Figure title, left-aligned. The run descriptor goes into a footer (see _footer) so it never collides."""
    ax.set_title(what, loc='left', fontsize=12, fontweight='bold')


def _footer(fig, cfg):
    """Run descriptor as a small grey line at the bottom-right of the figure."""
    fig.text(0.995, 0.004, _run_desc(cfg), ha='right', va='bottom', fontsize=8.5, color='dimgray')


def winner_age_series(d):
    # Inputs (from run.npz): points (S,), window_spikes (S, N) = spikes of each neuron in the 3,200-example window ending
    # at each snapshot, recruit_step_snap (S, N) = recruitment time of each neuron as known at that snapshot (NaN = untrained).
    # For each snapshot s: rank neurons by window spikes, take the TOP_WINNERS most active, and look at their ages
    # (points[s] - recruitment time). Age 0 = recruited inside this very window; NaN = still untrained.
    """Per snapshot: median/min age (examples since recruitment) of the TOP_WINNERS most active neurons,
    share of window spikes from young neurons (age < YOUNG_AGE), share from untrained neurons."""
    pts = d['points']; ws = d['window_spikes']; rs = d['recruit_step_snap']
    S = len(pts)
    med_age = np.full(S, np.nan); min_age = np.full(S, np.nan)
    young_share = np.full(S, np.nan); untrained_share = np.full(S, np.nan)
    top_winner_idx = np.full(S, -1)
    for s in range(S):
        w = ws[s]; tot = w.sum()
        if tot <= 0:
            continue
        order = np.argsort(w)[::-1][:TOP_WINNERS]
        top_winner_idx[s] = order[0]
        age = pts[s] - rs[s]                      # NaN for untrained neurons
        ages = age[order]
        if np.any(~np.isnan(ages)):
            med_age[s] = np.nanmedian(ages); min_age[s] = np.nanmin(ages)
        young = (~np.isnan(age)) & (age < YOUNG_AGE)
        young_share[s] = 100.0 * w[young].sum() / tot
        untrained_share[s] = 100.0 * w[np.isnan(age)].sum() / tot
    return med_age, min_age, young_share, untrained_share, top_winner_idx


# ------------------------------------------------------------------------------------------------------------
# per-run figures
# ------------------------------------------------------------------------------------------------------------
def make_run_figures(run_dir):
    run_dir = Path(run_dir)
    d, cfg = load_run(run_dir)
    pts = d['points']; acc = d['accuracy']
    spike_time = spike_time_from_curve(pts, acc)
    cfg['spike_time'] = spike_time
    (run_dir / 'run.json').write_text(json.dumps(cfg, indent=2))

    # ---- F1 accuracy ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if len(d['cooper_points']):
        ax.plot(d['cooper_points'], d['cooper_all'], color='tab:gray', linewidth=0.6, alpha=0.35, label='BindsNET "all" (raw, 224-sample windows)')
        ax.plot(d['cooper_points'], moving_average(d['cooper_all'], 15), color='tab:orange', linewidth=1.2, label='BindsNET "all" (15-window average)')
    ax.plot(pts, acc, color='blue', linewidth=2, label='Held-out (assign 2k train / score 1k test)')
    _vlines(ax, cfg, spike_time)
    ax.axhline(SPIKE_LEVEL, color='green', linewidth=0.6, alpha=0.4)
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('Accuracy (%)'); ax.grid(True, linestyle='--', alpha=0.5)
    _titles(ax, 'Held-out accuracy', cfg); ax.legend(loc='lower right', fontsize=9)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F1_accuracy.png', dpi=150); plt.close(fig)

    # ---- F2 reservoir -----------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(pts, d['reservoir_count'], color='black', linewidth=2, label='untrained neurons (top-100 share < 0.35)')
    ax.set_ylabel('Untrained neurons (reservoir)'); ax.set_xlabel('Number of training examples'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(pts, acc, color='blue', alpha=0.35, linewidth=1.5, label='held-out accuracy'); ax2.set_ylabel('Accuracy (%)', color='blue')
    _vlines(ax, cfg, spike_time)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc='center left', fontsize=9)
    _titles(ax, 'Untrained neurons (reservoir)', cfg)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F2_reservoir.png', dpi=150); plt.close(fig)

    # ---- F3 hijacker index --------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(pts, d['hijack_index'], color='purple', linewidth=2, label='share of weight change carried by the 20 biggest movers')
    ax.set_ylim(0, 1); ax.set_ylabel('Hijacker index'); ax.set_xlabel('Number of training examples'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(pts, acc, color='blue', alpha=0.35, linewidth=1.5, label='held-out accuracy'); ax2.set_ylabel('Accuracy (%)', color='blue')
    _vlines(ax, cfg, spike_time)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc='center right', fontsize=9)
    _titles(ax, 'Hijacker index', cfg)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F3_hijacker_index.png', dpi=150); plt.close(fig)

    # ---- F4 winner age ------------------------------------------------------------------------------------
    # Top panel: median and minimum age of the 5 most active neurons per window (k examples since recruitment).
    # Bottom panel: % of the window's spikes produced by neurons younger than YOUNG_AGE, and by untrained neurons.
    # Hijacking predicts: top winners have age ~0 during the plateau (recruited in the same window), and the young
    # share is high; after the spike there are no young neurons, so both change abruptly.
    med_age, min_age, young_share, untrained_share, top_idx = winner_age_series(d)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax.plot(pts, med_age / 1000, color='darkred', linewidth=2, label=f'median age of the top-{TOP_WINNERS} (examples since recruitment)')
    ax.plot(pts, min_age / 1000, color='salmon', linewidth=1.2, label=f'youngest of the top-{TOP_WINNERS}')
    ax.axhline(YOUNG_AGE / 1000, color='gray', linewidth=0.8, linestyle='--', label=f'"young" = recruited < {_k(YOUNG_AGE)} ago')
    ax.set_ylabel('Age of top winners (k examples)'); ax.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax.twinx(); ax2.plot(pts, acc, color='blue', alpha=0.35, linewidth=1.5); ax2.set_ylabel('Accuracy (%)', color='blue')
    _vlines(ax, cfg, spike_time); ax.legend(loc='upper left', fontsize=8)
    _titles(ax, 'Age of the five most active neurons in each window', cfg)
    bx.plot(pts, young_share, color='darkorange', linewidth=2, label=f'% of window spikes from neurons recruited < {_k(YOUNG_AGE)} ago')
    bx.plot(pts, untrained_share, color='gray', linewidth=1.5, label='% of window spikes from untrained neurons')
    bx.set_title("Where each window's spikes come from", loc='left', fontsize=12, fontweight='bold')
    bx.set_ylabel('% of spikes in window'); bx.set_xlabel('Number of training examples'); bx.grid(True, linestyle='--', alpha=0.5); bx.set_ylim(0, 100)
    bx2 = bx.twinx(); bx2.plot(pts, acc, color='blue', alpha=0.35, linewidth=1.5); bx2.set_ylabel('Accuracy (%)', color='blue')
    _vlines(bx, cfg, spike_time); bx.legend(loc='upper right', fontsize=8)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F4_winner_age.png', dpi=150); plt.close(fig)

    # ---- F5 activity by recruitment order (the mechanism picture) -------------------------------------
    # Rows: the 400 neurons sorted by their (final) recruitment time, earliest at the top, never-trained at the bottom.
    # Columns: snapshots. Colour: sqrt of the neuron's share of the window's spikes (sqrt compresses the range so
    # the many small values stay visible). Cyan dot: the neuron's recruitment time. A recruit that hijacks shows up
    # as a bright streak starting at its dot; all recruits together form a diagonal that should end at the spike.
    ws = d['window_spikes']; rs_final = d['recruit_step_snap'][-1]
    share = ws / np.maximum(ws.sum(1, keepdims=True), 1e-9)          # [S, N] fraction of each window's spikes
    order = np.argsort(np.where(np.isnan(rs_final), np.inf, rs_final), kind='stable')   # earliest recruits first, never-trained last
    img = np.sqrt(share[:, order].T)                                  # sqrt compresses the dynamic range
    fig, ax = plt.subplots(figsize=(11, 7))
    # Column s is the window ENDING at pts[s]. With a left edge of pts[0] - spacing every column is exactly one
    # snapshot interval wide and its right edge is its snapshot time; drawing from pts[0] instead would compress the
    # columns to (pts[-1] - pts[0]) / S each and slide them by up to one window over the run.
    spacing = pts[1] - pts[0] if len(pts) > 1 else 1
    extent = [pts[0] - spacing, pts[-1], len(order) - 0.5, -0.5]
    im = ax.imshow(img, aspect='auto', cmap='magma', extent=extent, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label('sqrt(share of window spikes)')
    rows = np.arange(len(order)); rs_sorted = rs_final[order]
    ok = ~np.isnan(rs_sorted)
    ax.scatter(rs_sorted[ok], rows[ok], s=4, color='cyan', label='recruitment time of this neuron')
    imp = d['imprinted'][order]; ref = d['refilled'][order]
    if imp.any() and cfg.get('intervention_step_actual') is not None:   # sweep: imprint-at-zero marks all rows but has no intervention step
        ax.scatter(np.full(imp.sum(), cfg['intervention_step_actual']), rows[imp], s=10, marker='>', color='lime', label='imprinted by intervention')
    if ref.any() and cfg.get('intervention_step_actual') is not None:
        ax.scatter(np.full(ref.sum(), cfg['intervention_step_actual']), rows[ref], s=10, marker='<', color='white', label='re-initialised by intervention')
    _vlines(ax, cfg, spike_time)
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('Neurons, sorted by recruitment time (never trained at the bottom)')
    _titles(ax, 'Window activity by recruitment order', cfg)
    ax.legend(loc='lower left', fontsize=8)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F5_activity_by_recruitment.png', dpi=150); plt.close(fig)

    # ---- F6 theta -----------------------------------------------------------------------------------------
    # Percentiles of theta over the trained neurons (10/50/90/max) and the median over untrained ones, per snapshot.
    # pct() takes a (S, N) boolean mask and returns the q-th percentile of theta within the mask at each snapshot.
    th = d['theta_snap']; sh = d['sharp_snap']
    trained = sh >= cfg['untrained_threshold']
    def pct(mask, q):
        out = np.full(len(pts), np.nan)
        for s in range(len(pts)):
            v = th[s][mask[s]]
            if v.size: out[s] = np.percentile(v, q)
        return out
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(pts, pct(trained, 10), pct(trained, 90), color='tab:blue', alpha=0.2, label='trained neurons: 10th-90th percentile of θ')
    ax.plot(pts, pct(trained, 50), color='tab:blue', linewidth=2, label='trained neurons: median θ')
    ax.plot(pts, pct(trained, 100), color='tab:blue', linewidth=0.8, linestyle='--', label='trained neurons: max θ')
    ax.plot(pts, pct(~trained, 50), color='black', linewidth=2, label='untrained neurons: median θ')
    ax.axhline(cfg['theta_base'], color='gray', linewidth=0.8, label=f"θ_base = {cfg['theta_base']} mV")
    _vlines(ax, cfg, spike_time)
    ax.set_xlabel('Number of training examples'); ax.set_ylabel('θ (mV)'); ax.grid(True, linestyle='--', alpha=0.5)
    _titles(ax, 'Adaptive thresholds θ', cfg); ax.legend(loc='upper left', fontsize=8)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F6_theta.png', dpi=150); plt.close(fig)

    # ---- F7 class bias ------------------------------------------------------------------------------------
    # conf_snap[s] is the held-out confusion matrix (rows = true digit, columns = predicted digit). Summing over rows
    # gives how often each digit was PREDICTED, summing over columns how often it was TRUE; the difference is the
    # over-prediction (red) / under-prediction (blue) of each class at each snapshot. The black x marks the held-out
    # assignment of the window's single most active neuron when that neuron is young. NOTE: on the baseline runs the
    # x's match the most over-predicted class only at chance level (6-12%), so the one-to-one claim 'bias class =
    # top winner's class' is not supported; what the panel does show is a concentrated bias that rotates quickly.
    conf = d['conf_snap']                                              # [S, 10, 10] rows = true, cols = predicted
    excess = conf.sum(1) - conf.sum(2)                                 # predicted count - true count, per class  [S, 10]
    fig, ax = plt.subplots(figsize=(11, 5))
    vmax = max(1.0, np.abs(excess).max())
    spacing = pts[1] - pts[0] if len(pts) > 1 else 1
    im = ax.imshow(excess.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=[pts[0] - spacing, pts[-1], 9.5, -0.5], interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label('predicted − true count (held-out 1k)  [red = over-predicted]')
    assign = d['assignments_snap']
    xs, ys = [], []
    for s in range(len(pts)):
        i = top_idx[s]
        if i < 0: continue
        age = pts[s] - d['recruit_step_snap'][s][i]
        if not np.isnan(age) and age < YOUNG_AGE and assign[s][i] >= 0:
            xs.append(pts[s]); ys.append(assign[s][i])
    if xs:
        ax.scatter(xs, ys, marker='x', color='black', s=40, label=f'class of the top winner when it is young (< {_k(YOUNG_AGE)})')
        ax.legend(loc='upper right', fontsize=8)
    _vlines(ax, cfg, spike_time)
    ax.set_yticks(range(10)); ax.set_xlabel('Number of training examples'); ax.set_ylabel('Digit class')
    _titles(ax, 'Class bias of the held-out predictions', cfg)
    _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / 'F7_class_bias.png', dpi=150); plt.close(fig)

    return spike_time
