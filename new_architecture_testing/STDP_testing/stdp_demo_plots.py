"""
stdp_demo_plots.py -- figures for stdp_demo.py runs (torch-free; reads results/<run>/run.npz + run.json).

TEST 1  T1_raster_<N>k.png     four presentations from the batch logged at N examples: every spike of every neuron
                                 that fired, in time order (winner in red), with first-spike time, winner spikes,
                                 distinct neurons and "neurons that fired after the winner's first spike".
        T1_membrane_<N>k.png   membrane potential of the winner and its two closest runners-up on the first image of
                                 that batch, with their thresholds: the winner climbs, crosses, resets and re-fires;
                                 the others are kicked -200 mV at every winner spike and never recover.
        T1_stats.png           over all logged presentations: distinct neurons per presentation, the winner's share of
                                 spikes, first-spike latency, the winner's inter-spike interval, late intruders.
        T1_ties_<N>k.png       every presentation of the batch as a row: when several neurons cross threshold in the
                                 same millisecond they all fire (inhibition arrives one step later), each is reset and
                                 gets -200 mV from each of the others, and the presentation is silent until the
                                 strongest climbs out; group size, silence length and the top neuron's share vs size.
        T1_membrane_tie_<N>k.png membrane potentials on the biggest-tie presentation of that batch (fresh runs only).
TEST 2  T2_recruit<k>_n<id>.png one fresh hijacker: the image it won, its template every 50 ms through the first
                                 presentations and at batch ends afterwards; template sharpness and similarity to its
                                 final template vs examples since the first win; its theta vs the trained population's.
        T2_timescales.png      per recruit: examples until the template was formed vs examples until theta reached
                                 the trained population's level.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_K, FLAT = 100, 0.35


def sharpness(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64); return float(np.sort(w)[-TOP_K:].sum() / max(1e-9, w.sum()))


def cosine(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _footer(fig, cfg, extra=""):
    fig.text(0.995, 0.004, f"baseline network: θ+ {cfg.get('theta_plus', 0.05):g}, {cfg.get('inhibition', 200):g} mV inhibition, seed {cfg.get('seed', 0)}{extra}",
             ha="right", va="bottom", fontsize=8, color="dimgray")


def presentation_stats(spikes_b: np.ndarray):
    """spikes_b: (T, n) 0/1 for one presentation -> dict of the quantities test 1 is about."""
    T, n = spikes_b.shape
    counts = spikes_b.sum(0)
    fired = np.where(counts > 0)[0]
    if len(fired) == 0:
        return {"fired": fired, "first": None, "first_t": None, "winner": None, "winner_spikes": 0, "winner_share": np.nan, "distinct": 0, "late": 0, "isi": np.array([]), "first_t_of": {}}
    first_t_of = {int(i): int(np.argmax(spikes_b[:, i] > 0)) for i in fired}
    first_t = min(first_t_of.values()); firsts = [i for i, t in first_t_of.items() if t == first_t]
    winner = int(fired[np.argmax(counts[fired])])
    wt = np.where(spikes_b[:, winner] > 0)[0]
    return {"fired": fired, "first": firsts, "first_t": first_t, "winner": winner, "winner_spikes": int(counts[winner]),
            "winner_share": float(counts[winner] / counts.sum()), "distinct": int(len(fired)),
            "late": int(sum(1 for i, t in first_t_of.items() if t > first_t)), "isi": np.diff(wt), "first_t_of": first_t_of}


def make_figures(run_dir):
    run_dir = Path(run_dir)
    d = dict(np.load(run_dir / "run.npz")); cfg = json.loads((run_dir / "run.json").read_text())
    T = int(cfg.get("steps", 350))

    # =========================================================== TEST 1 ===========================================
    pooled = {"distinct": [], "share": [], "first_t": [], "isi": [], "late": [], "which": []}
    k = 0
    while f"raster{k}_spikes" in d:
        sp = d[f"raster{k}_spikes"]; v = d[f"raster{k}_v_img0"]; th = d[f"raster{k}_theta"]; labels = d[f"raster{k}_labels"]
        ex = int(d[f"raster{k}_examples"]); tag = f"{ex // 1000}k"
        Tk, B, n = sp.shape
        stats = [presentation_stats(sp[:, b, :]) for b in range(B)]
        for s in stats:
            if s["winner"] is None: continue
            pooled["distinct"].append(s["distinct"]); pooled["share"].append(s["winner_share"]); pooled["first_t"].append(s["first_t"])
            pooled["isi"].extend(s["isi"].tolist()); pooled["late"].append(s["late"]); pooled["which"].append(tag)
        # ---- T1 raster: four presentations ----
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for b, ax in enumerate(axes.ravel()):
            s = stats[b]
            if s["winner"] is None:
                ax.text(0.5, 0.5, "no spikes", ha="center"); continue
            order = sorted(s["fired"], key=lambda i: s["first_t_of"][int(i)])
            for rank, i in enumerate(order):
                ts = np.where(sp[:, b, i] > 0)[0]
                ax.vlines(ts, rank - 0.4, rank + 0.4, color=("red" if i == s["winner"] else "gray"), linewidth=1.2)
            ax.set_yticks(range(len(order))); ax.set_yticklabels([f"n{int(i)}" + ("  ← winner" if i == s["winner"] else "") for i in order], fontsize=8)
            ax.set_xlim(0, Tk); ax.set_xlabel("time in the presentation (ms)"); ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            ax.set_ylim(-0.7, len(order) - 0.3)
            ax.set_title(f"presentation {b} (digit {int(labels[b])}) — first spike at {s['first_t']} ms by n{s['first'][0]}\n"
                         f"winner n{s['winner']}: {s['winner_spikes']} spikes ({100 * s['winner_share']:.0f}% of all)  ·  {s['distinct']} distinct neuron(s)  ·  {s['late']} fired after the first spike", fontsize=8.5, loc="left")
        fig.suptitle(f"TEST 1 — who spikes, in order, during one presentation (batch logged at {ex:,} examples)", fontsize=12, fontweight="bold")
        _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 0.96)); fig.savefig(run_dir / f"T1_raster_{tag}.png", dpi=150); plt.close(fig)
        # ---- T1 membrane potentials, image 0 ----
        s = stats[0]
        if s["winner"] is not None:
            w = s["winner"]; ft = s["first_t"]
            pre = v[: max(1, ft)]                                       # potentials up to the first spike
            drive = pre.max(0) if ft > 0 else v[0]
            cand = [i for i in np.argsort(-drive) if i != w][:2]
            fig, ax = plt.subplots(figsize=(13, 5.5))
            colors = ["red", "tab:blue", "tab:green"]
            th_t = d.get(f"raster{k}_theta_t")                            # (T, n) if the run logged theta per ms
            for c, i in zip(colors, [w] + cand):
                ax.plot(np.arange(Tk), v[:, i], color=c, linewidth=1.3, label=f"n{int(i)}" + (" (winner)" if i == w else " (runner-up)"))
                if th_t is not None: ax.plot(np.arange(Tk), th_t[:, i], color=c, linewidth=0.9, linestyle="--", alpha=0.9)
                else: ax.axhline(th[i], color=c, linewidth=0.8, linestyle="--", alpha=0.8)
            if th_t is not None:
                # inset: the winner's theta during the presentation, zoomed -- it rises by spikes x theta_plus / 16
                ins = ax.inset_axes([0.62, 0.42, 0.36, 0.3])
                tp = cfg.get('theta_plus', 0.05); rise = float(th_t[-1, w] - th_t[0, w])
                ins.plot(np.arange(Tk), th_t[:, w] - th_t[0, w], color="red", linewidth=1.5)         # relative to the start of the batch
                ins.set_title(f"winner's θ, zoomed: rises {rise:+.3f} mV over the batch (from {th_t[0, w]:.2f} mV)\n"
                              f"each spike adds θ⁺/16 = {tp / 16:.4f} mV; θ is shared by the batch's 32 presentations", fontsize=7)
                ins.set_xlabel("ms", fontsize=7); ins.set_ylabel("Δθ (mV)", fontsize=7); ins.tick_params(labelsize=7); ins.grid(True, linestyle="--", alpha=0.4)
            wt = np.where(sp[:, 0, w] > 0)[0]
            for t_ in wt: ax.axvline(t_, color="red", alpha=0.12, linewidth=1)
            ax.axhline(-65, color="gray", linewidth=0.6, linestyle=":", label="E_rest = −65 mV")
            ax.set_xlabel("time in the presentation (ms)"); ax.set_ylabel("membrane potential (mV)"); ax.set_xlim(0, Tk); ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="lower right", fontsize=9)
            isi_txt = f"re-fires every {np.mean(s['isi']):.0f} ms" if len(s['isi']) else "fires only once in the logged window"
            ax.set_title(f"TEST 1 — membrane potentials on presentation 0 (digit {int(labels[0])}): the winner crosses its threshold (dashed) at {ft} ms, is reset to E_rest = −65 mV "
                         f"(no refractory period)\nand {isi_txt}; each winner spike (pink lines) knocks every other neuron down by {cfg.get('inhibition', 200):g} mV, "
                         f"so nobody else can reach threshold for the rest of the presentation", fontsize=9.5, loc="left")
            _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / f"T1_membrane_{tag}.png", dpi=150); plt.close(fig)
        # ---- T1 ties: the whole batch, one row per presentation ----
        per_ms = sp.sum(2)                                                                   # (T, B) spikes per ms
        rows = {}
        for b in range(B):
            s = stats[b]
            if s["winner"] is None: rows[b] = (0, np.nan, np.nan); continue
            t0 = s["first_t"]; ksz = int(per_ms[t0, b]); nxt = np.where(per_ms[t0 + 1:, b] > 0)[0]
            rows[b] = (ksz, int(nxt[0]) + 1 if len(nxt) else Tk - t0, s["winner_share"])        # (neurons in the first ms, ms until the next spike, top share)
        order = sorted(range(B), key=lambda b: (rows[b][0], rows[b][1] if np.isfinite(rows[b][1]) else -1))   # solo-first at the bottom, biggest ties on top
        fig = plt.figure(figsize=(20, 10.5)); gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1], hspace=0.45, wspace=0.12)
        ax = fig.add_subplot(gs[:, 0])
        for y, b in enumerate(order):
            s = stats[b]; ksz, gap, share = rows[b]
            if s["winner"] is None:
                ax.text(Tk / 2, y, "no spikes", ha="center", va="center", fontsize=7, color="gray"); continue
            for i in s["fired"]:
                ts = np.where(sp[:, b, i] > 0)[0]
                ax.vlines(ts, y - 0.38, y + 0.38, color=("red" if i == s["winner"] else "gray"), linewidth=1.1, alpha=(1.0 if i == s["winner"] else 0.8))
            if ksz > 1:
                ax.plot(s["first_t"], y, "o", color="black", markersize=4 + 0.6 * ksz, alpha=0.35)
                ax.text(s["first_t"] + 4 + 0.4 * ksz, y, f"×{ksz}", fontsize=8, va="center", fontweight="bold")
                ax.annotate("", xy=(s["first_t"] + gap, y), xytext=(s["first_t"] + 1, y), arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=0.8, shrinkA=0, shrinkB=0))
                ax.text(Tk + 4, y, f"silent {gap} ms → n{s['winner']} {100 * share:.0f}%", fontsize=7.5, va="center", color="tab:blue")
            else:
                ax.text(Tk + 4, y, f"n{s['winner']} {100 * share:.0f}%", fontsize=7.5, va="center", color="dimgray")
        ax.set_yticks(range(B)); ax.set_yticklabels([f"pres {b} · digit {int(labels[b])}" for b in order], fontsize=7.5)
        ax.axvline(Tk, color="black", linewidth=0.8)
        ax.set_ylim(-0.7, B - 0.3); ax.set_xlim(0, Tk + 100); ax.set_xticks(np.arange(0, Tk + 1, 50)); ax.set_xlabel("time in the presentation (ms)"); ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        n_tie = sum(1 for b in rows if rows[b][0] > 1)
        ax.set_title(f"all {B} presentations of the batch, sorted by the number of neurons in the FIRST spiking millisecond (●×k = k neurons fired together; "
                     f"red = the neuron with the most spikes, gray = everyone else)\n{n_tie}/{B} presentations begin with a tie; a tie is followed by "
                     f"silence (blue arrow) because every tied neuron is reset to −65 mV and gets −{cfg.get('inhibition', 200):g} mV from each of the others", fontsize=9, loc="left")
        ks = np.array([rows[b][0] for b in range(B)]); gaps = np.array([rows[b][1] for b in range(B)], float); shares = np.array([rows[b][2] for b in range(B)], float)
        ok = ks > 0
        ax1 = fig.add_subplot(gs[0, 1]); ax1.hist(ks[ok], bins=np.arange(0.5, max(2, ks.max()) + 1.5, 1), color="black", alpha=0.6)
        ax1.set_title("neurons in the first spiking ms\n(1 = a solo first spiker)", fontsize=9); ax1.set_xlabel("k"); ax1.grid(True, linestyle="--", alpha=0.4)
        ax2 = fig.add_subplot(gs[1, 1]); ax2.scatter(ks[ok], gaps[ok], color="tab:blue", s=22, zorder=3, label="measured")
        kk = np.arange(1, max(2, ks.max()) + 1)
        for D, ls in [(2, ":"), (3, "--"), (4, "-.")]:
            ax2.plot(kk, 100 * np.log((100 * D + 200 * (kk - 1)) / (100 * D - 35)), color="gray", linestyle=ls, linewidth=0.9, label=f"LIF recovery, drive {D} mV/ms")
        ax2.set_title("ms until the next spike after the first spiking ms\ncurves: a tied neuron's LIF recovery from −65 − 200(k−1) mV to θ = −30 mV,\n"
                      "t = τ_m · ln[(100·D + 200(k−1)) / (100·D − 35)] for a drive of D mV/ms", fontsize=8)
        ax2.set_xlabel("k"); ax2.grid(True, linestyle="--", alpha=0.4); ax2.legend(fontsize=7, loc="lower right")
        ax3 = fig.add_subplot(gs[2, 1]); ax3.scatter(ks[ok], shares[ok], color="red", s=22, zorder=3)
        ax3.set_title("top neuron's share of the presentation's spikes\n(the low-share presentations in T1_stats are the ties)", fontsize=9); ax3.set_xlabel("k"); ax3.set_ylim(0, 1.05); ax3.grid(True, linestyle="--", alpha=0.4)
        fig.suptitle(f"TEST 1 — ties: when several neurons cross threshold in the same millisecond they all fire (inhibition only arrives at the next step) and knock each other down;\n"
                     f"the presentation then goes silent until the strongest neuron climbs out (batch logged at {ex:,} examples)", fontsize=11, fontweight="bold")
        _footer(fig, cfg); fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.06); fig.savefig(run_dir / f"T1_ties_{tag}.png", dpi=150); plt.close(fig)
        # ---- T1 membrane potentials on the biggest-tie presentation (needs the v_tie log of a fresh run) ----
        vt = d.get(f"raster{k}_v_tie"); th_t = d.get(f"raster{k}_theta_t")
        if vt is not None and int(d[f"raster{k}_tie_pres"]) in rows and rows[int(d[f"raster{k}_tie_pres"])][0] > 1:
            b = int(d[f"raster{k}_tie_pres"]); s = stats[b]; ksz, gap, share = rows[b]; t0 = s["first_t"]; w = s["winner"]
            group = np.where(sp[t0, b, :] > 0)[0]
            fig, ax = plt.subplots(figsize=(19, 8))
            for t in np.where(per_ms[:, b] > 0)[0]: ax.axvline(t, color="pink", linewidth=0.6, alpha=0.7, zorder=0)
            for i in group:
                if i != w: ax.plot(np.arange(Tk), vt[:, i], color="gray", linewidth=0.9, alpha=0.7)
            ax.plot([], [], color="gray", linewidth=0.9, label=f"the {ksz} neurons that fired together at {t0} ms")
            ax.plot(np.arange(Tk), vt[:, w], color="red", linewidth=1.6, label=f"n{w}: the neuron that ends up with the most spikes ({s['winner_spikes']}){' — was in the tie' if w in group else ' — was NOT in the tie'}")
            if th_t is not None: ax.plot(np.arange(Tk), th_t[:, w], color="red", linestyle="--", linewidth=0.9, label=f"n{w}'s threshold")
            else: ax.axhline(th[w], color="red", linestyle="--", linewidth=0.9, label=f"n{w}'s threshold")
            ax.axhline(-65, color="black", linestyle=":", linewidth=0.8, label="E_rest = −65 mV")
            ax.set_xlim(0, Tk); ax.set_xlabel("time in the presentation (ms)"); ax.set_ylabel("membrane potential (mV)"); ax.grid(True, linestyle="--", alpha=0.4); ax.legend(fontsize=8, loc="lower right")
            ax.set_title(f"TEST 1 — a tied presentation (presentation {b}, digit {int(labels[b])}): {ksz} neurons cross threshold in the same millisecond ({t0} ms); each is reset to −65 mV and receives "
                         f"−{(ksz - 1) * cfg.get('inhibition', 200):g} mV from the other {ksz - 1}, everyone else receives −{ksz * cfg.get('inhibition', 200):g} mV;\nnothing fires for {gap} ms, "
                         f"then n{w} climbs out first and takes the rest — the top neuron holds only {100 * share:.0f}% of the presentation's spikes (pink lines = every spike in the presentation)", fontsize=9.5, loc="left")
            _footer(fig, cfg); fig.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(run_dir / f"T1_membrane_tie_{tag}.png", dpi=150); plt.close(fig)
        k += 1
    if pooled["distinct"]:
        tags = sorted(set(pooled["which"])); cmap = plt.get_cmap("tab10"); col = {t: cmap(i) for i, t in enumerate(tags)}
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        which = np.array(pooled["which"])
        for (key, title, bins) in [("distinct", "distinct neurons that fired\nper presentation", np.arange(0.5, 6.5, 1)), ("share", "winner's share of the\npresentation's spikes", np.linspace(0, 1, 21)),
                                   ("first_t", "first-spike latency (ms)", np.linspace(0, T, 36)), ("late", "neurons that fired AFTER\nthe winner's first spike", np.arange(-0.5, 5.5, 1))]:
            ax = axes[["distinct", "share", "first_t", "late"].index(key)]
            arr = np.array(pooled[key], dtype=float)
            for t in tags:
                ax.hist(arr[which == t], bins=bins, alpha=0.6, color=col[t], label=f"logged at {t}")
            ax.set_title(title, fontsize=10); ax.grid(True, linestyle="--", alpha=0.4); ax.legend(fontsize=8)
        ax = axes[4]; ax.hist(np.array(pooled["isi"]), bins=np.arange(0.5, 40.5, 1), color="red", alpha=0.7); ax.set_title("winner's inter-spike interval (ms)", fontsize=10); ax.grid(True, linestyle="--", alpha=0.4)
        fig.suptitle(f"TEST 1 — over {len(pooled['distinct'])} logged presentations: a presentation has one winner, it produces nearly all the spikes, and nobody fires after it has started"
                     f" — except when the first spike is a tie (see T1_ties): the tied neurons knock each other down and the spikes are split", fontsize=11, fontweight="bold")
        _footer(fig, cfg); fig.tight_layout(rect=(0, 0.03, 1, 0.93)); fig.savefig(run_dir / "T1_stats.png", dpi=150); plt.close(fig)

    # =========================================================== TEST 2 ===========================================
    k = 0; summary = []
    while f"rec{k}_frames" in d:
        nid = int(d[f"rec{k}_neuron"]); won = d[f"rec{k}_won_image"]; lab = int(d[f"rec{k}_won_label"]); fw = int(d[f"rec{k}_first_win_examples"])
        inb = d[f"rec{k}_inbatch_frames"]; inb_ms = d[f"rec{k}_inbatch_ms"]; fr = d[f"rec{k}_frames"]; ex = d[f"rec{k}_examples"]
        th = d[f"rec{k}_theta"]; wins = d[f"rec{k}_wins"]; spk = d[f"rec{k}_spikes"]; thm = d[f"rec{k}_th_med"]; thp = d[f"rec{k}_th_p10"]
        if len(fr) == 0: k += 1; continue
        final = fr[-1]
        sh_in = np.array([sharpness(f) for f in inb]); cos_in = np.array([cosine(f, final) for f in inb])
        sh_b = np.array([sharpness(f) for f in fr]); cos_b = np.array([cosine(f, final) for f in fr])
        # --- filmstrip selection: in-batch frames of the first presentation batch + chosen batch ends ---
        strip = [("won image\n(digit %d)" % lab, won.reshape(28, 28), None)]
        n_in_first = int(np.sum(inb_ms < T)) if len(inb_ms) else 0
        for j in range(n_in_first):
            strip.append((f"{int(inb_ms[j])} ms", inb[j].reshape(28, 28), sh_in[j]))
        for bidx in [1, 2, 3, 5, 10, 20, 50, 100, 200, 300]:
            if bidx <= len(fr): strip.append((f"+{bidx} batch{'es' if bidx > 1 else ''}\n(+{int(ex[bidx - 1])} ex)", fr[bidx - 1].reshape(28, 28), sh_b[bidx - 1]))
        strip = strip[:20]
        fig = plt.figure(figsize=(max(14, 1.15 * len(strip)), 10))
        gs = fig.add_gridspec(3, len(strip), height_ratios=[0.55, 1.15, 1.15], hspace=0.5, wspace=0.25)
        vmax = max(float(s[1].max()) for s in strip[1:]) if len(strip) > 1 else 1
        for j, (title, img, sh) in enumerate(strip):
            ax = fig.add_subplot(gs[0, j]); ax.axis("off")
            ax.imshow(img, cmap="gray" if j else "Blues", vmin=0, vmax=(vmax if j else img.max()))
            ax.set_title(title + (f"\nshare {sh:.2f}" if sh is not None else ""), fontsize=7.5)
        # --- row 2: template formation ---
        ax1 = fig.add_subplot(gs[1, : max(1, len(strip) // 3)]); ax2 = fig.add_subplot(gs[1, max(1, len(strip) // 3):])
        if len(inb_ms):
            ax1.plot(inb_ms, sh_in, color="black", marker="o", markersize=3, label="top-100 share of the template")
            ax1.plot(inb_ms, cos_in, color="tab:blue", marker="o", markersize=3, label="cosine to the final template")
            for bb in range(1, int(np.ceil(inb_ms.max() / T)) + 1): ax1.axvline(bb * T, color="gray", linewidth=0.7, linestyle=":")
        ax1.axhline(FLAT, color="gray", linewidth=0.8, linestyle="--", label="0.35 = 'digit-like'")
        ax1.set_xlabel("ms since the start of the first-win presentation (dotted = presentation boundaries)"); ax1.set_ylim(0, 1.02); ax1.grid(True, linestyle="--", alpha=0.4); ax1.legend(fontsize=8, loc="lower right")
        ax1.set_title("within the first presentations", loc="left", fontsize=10, fontweight="bold")
        ax2.plot(ex, sh_b, color="black", marker=".", linewidth=1.2, label="top-100 share"); ax2.plot(ex, cos_b, color="tab:blue", marker=".", linewidth=1.2, label="cosine to final")
        ax2.axhline(FLAT, color="gray", linewidth=0.8, linestyle="--"); ax2.set_xscale("log"); ax2.set_ylim(0, 1.02); ax2.grid(True, linestyle="--", alpha=0.4, which="both")
        ax2.set_xlabel("examples since the first win (log)"); ax2.legend(fontsize=8, loc="lower right"); ax2.set_title("over the following batches", loc="left", fontsize=10, fontweight="bold")
        # --- row 3: theta vs the trained population; wins per batch ---
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(ex, th, color="red", linewidth=2, label=f"θ of neuron {nid}")
        ax3.plot(ex, thm, color="tab:blue", linewidth=1.5, label="trained population: median θ"); ax3.plot(ex, thp, color="tab:blue", linewidth=1, linestyle="--", label="trained population: 10th percentile")
        ax3.axhline(-52, color="gray", linewidth=0.7, label="θ_base = −52 mV"); ax3.set_xscale("log"); ax3.set_ylabel("θ (mV)"); ax3.grid(True, linestyle="--", alpha=0.4, which="both")
        ax3.set_xlabel("examples since the first win (log)")
        ax3b = ax3.twinx(); ax3b.bar(ex, wins, width=np.array(ex) * 0.08, color="orange", alpha=0.45, label="presentations won in the batch (of 32)"); ax3b.set_ylabel("presentations won per batch", color="darkorange")
        ax3b.set_ylim(0, 32)
        # the two timescales
        # "formed": the first batch-end snapshot at which the template is digit-like (top-100 share >= 0.35) AND already has
        # its final shape (cosine to the final template >= 0.95); never, if the final template is still flat.
        ok_f = (sh_b >= FLAT) & (cos_b >= 0.95)
        formed = ex[np.argmax(ok_f)] if (sh_b[-1] >= FLAT and ok_f.any()) else np.nan
        caught = ex[np.argmax(th >= thp)] if np.any(th >= thp) else np.nan                  # theta reaches the trained population's 10th percentile
        caught_med = ex[np.argmax(th >= thm)] if np.any(th >= thm) else np.nan              # ... and its median
        if not np.isnan(formed): ax3.axvline(formed, color="black", linewidth=1.2, linestyle=":"); ax3.text(formed, ax3.get_ylim()[1], " template formed", va="top", fontsize=8)
        else: ax3.text(ex[0], ax3.get_ylim()[1], f" template never formed (share stays {sh_b[-1]:.2f})", va="top", fontsize=8)
        if not np.isnan(caught): ax3.axvline(caught, color="red", linewidth=1.2, linestyle=":"); ax3.text(caught, ax3.get_ylim()[0], " θ reaches trained p10", va="bottom", fontsize=8, color="red")
        if not np.isnan(caught_med): ax3.axvline(caught_med, color="darkred", linewidth=1.2, linestyle="-."); ax3.text(caught_med, ax3.get_ylim()[0] + 4, " θ reaches trained median", va="bottom", fontsize=8, color="darkred")
        h1, l1 = ax3.get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels(); ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
        ax3.set_title("θ of the recruit vs the trained population  ·  how many presentations it wins per batch (the hijack)", loc="left", fontsize=10, fontweight="bold")
        fmt = lambda v: "never" if np.isnan(v) else f"~{v:,.0f}"
        fig.suptitle(f"TEST 2 — fresh hijacker n{nid}: first win at {fw:,} examples on a '{lab}' ({int(d[f'rec{k}_won_spikes'])} spikes on that image; θ {float(d[f'rec{k}_theta_at_win']):.1f} mV vs trained median {thm[0]:.1f}; "
                     f"{int(np.sum(wins))} wins in the next {int(ex[-1]):,} examples).\nTemplate formed (digit-like and ≥ 0.95 cosine to its final shape) after {fmt(formed)} examples; "
                     f"θ reached the trained 10th percentile after {fmt(caught)}, the trained median after {fmt(caught_med)}.", fontsize=11, fontweight="bold")
        _footer(fig, cfg); fig.savefig(run_dir / f"T2_recruit{k}_n{nid}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        summary.append((nid, lab, formed, caught, float(th[-1]), float(thm[-1]), int(np.sum(wins)), caught_med))
        k += 1
    if summary:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(summary)); formed = np.array([s[2] for s in summary], float); caught = np.array([s[3] for s in summary], float); cmed = np.array([s[7] for s in summary], float)
        top = np.nanmax(np.r_[formed, caught, cmed, 1]) * 1.6                                # bar height used for "never" / "not yet"
        ax.bar(x - 0.27, np.where(np.isnan(formed), top, formed), width=0.27, color=np.where(np.isnan(formed), "lightgray", "black"), label="examples until the template was formed (digit-like, ≥ 0.95 cosine to its final shape)")
        ax.bar(x, np.where(np.isnan(caught), top, caught), width=0.27, color="red", alpha=0.7, label="examples until θ reached the trained population's 10th percentile")
        ax.bar(x + 0.27, np.where(np.isnan(cmed), top, cmed), width=0.27, color="darkred", alpha=0.7, hatch="//", label="examples until θ reached the trained population's median")
        for i, (nid, lab, f_, c_, _a, _b, nw, m_) in enumerate(summary):
            if np.isnan(f_): ax.text(i - 0.27, top, "never\nformed", ha="center", va="bottom", fontsize=7, color="gray")
            if np.isnan(c_): ax.text(i, top, "not\nyet", ha="center", va="bottom", fontsize=7, color="red")
            if np.isnan(m_): ax.text(i + 0.27, top, "not\nyet", ha="center", va="bottom", fontsize=7, color="darkred")
            if not np.isnan(f_) and not np.isnan(m_): ax.text(i, top * 1.9, f"×{m_ / max(f_, 1):.0f}", ha="center", fontsize=9)
            ax.text(i, 1.15, f"{nw} wins", ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
        ax.set_yscale("log"); ax.set_ylim(1, top * 3.2); ax.set_xticks(x); ax.set_xticklabels([f"n{s[0]} ('{s[1]}')" for s in summary]); ax.set_ylabel("training examples after the first win (log)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, which="both"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1, frameon=False)
        ax.set_title("TEST 2 — the two timescales for every tracked hijacker (×: examples to the trained-median θ ÷ examples to a formed template)", loc="left", fontsize=10, fontweight="bold")
        _footer(fig, cfg); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(run_dir / "T2_timescales.png", dpi=150); plt.close(fig)
    return len(summary)
