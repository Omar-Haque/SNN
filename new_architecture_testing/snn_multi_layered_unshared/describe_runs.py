"""Write a README.md into each run folder of the two-layer study (and its single-layer references) that spells out the run's
settings and its main results.

The folder names only carry a few tags (e.g. unshared_p7s2f16_n400_tp0.5_tau30k_ep6_seed0_l1wta_rd8_l2wta_drv0.36_lr10_...);
this writes the full set of settings, layer by layer, from the run's run.json, the main numbers from run.npz, and the Unity
job it came from when the batch's logs are there (job name, node, the flags from unity/submit_multi_layer.sh). Needs Python 3
and numpy (no torch); run from anywhere:

    python new_architecture_testing/snn_multi_layered_unshared/describe_runs.py \
        new_architecture_testing/snn_multi_layered_unshared/results_2/09_batch8_input_norm_6ep_20261126
    python .../describe_runs.py RUN_DIR [RUN_DIR ...]          # single run folders work too

A README.md this script wrote earlier is replaced; any other README.md in a run folder is left alone.

multi_layer_unshared.py calls write_run_readme() at the end of every run, so new run folders get theirs automatically, with
the exact command line the run was started with (and the Slurm job name on Unity).
"""
import argparse
import json
import os
import re
import shlex
from pathlib import Path

try:
    import numpy as np
except ImportError:                                   # the settings part still works; the per-epoch numbers need numpy
    np = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                             # repo root (holds unity/)
MARK = "<!-- written by new_architecture_testing/snn_multi_layered_unshared/describe_runs.py; rerun it to refresh -->"
EPOCH = 60000


def submit_jobs(script: Path) -> tuple[str, dict]:
    """Read a submit script's variables and `job NAME TIME ARGS... # comment` lines -> (OUT, {name: (flags, comment)})."""
    if not script.exists():
        return "", {}
    vars_: dict = {}
    jobs: dict = {}

    def expand(s: str) -> str:
        return re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: vars_.get(m.group(1), ""), s)

    for line in script.read_text().splitlines():
        line = line.strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\S+)', line)
        if m:
            vars_[m.group(1)] = expand(m.group(2).strip('"'))
            continue
        m = re.match(r"^job\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if m:
            args, _, comment = m.group(3).partition(" #")
            jobs[m.group(1)] = (" ".join(expand(args).split()), comment.strip().lstrip("#").strip())
    return vars_.get("OUT", ""), jobs


def unity_logs(logs: Path) -> dict:
    """run folder name -> {job, node, gpu, minutes} from the Slurm logs' 'node ...' and 'run finished ...' lines."""
    found: dict = {}
    if not logs.is_dir():
        return found
    for f in logs.glob("*.out"):
        node = gpu = None
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"^node (\S+) \| gpu: ([^,\n]+)", text, re.M)
        if m:
            node, gpu = m.group(1), m.group(2).strip()
        m = re.search(r"^run finished in ([0-9.]+) min .*\| (\S+)\s*$", text, re.M)
        if m:
            found[Path(m.group(2)).name] = {"job": re.sub(r"-\d+$", "", f.stem), "node": node, "gpu": gpu,
                                            "minutes": float(m.group(1))}
    return found


def mv(x) -> str:
    return f"{x:g} mV"


def tau(t) -> str:
    return "no decay" if not t else f"τ {t:,.0f} images"


def describe(run: Path, jobs: dict, out_dir: str, logs: dict, live: dict | None = None) -> str:
    """live (from write_run_readme, at the end of a run): {argv, job, node, gpu}; otherwise the Slurm logs are used."""
    d = json.loads((run / "run.json").read_text())
    single = bool(d.get("no_l1"))
    data = {"mnist": "MNIST", "fmnist": "Fashion-MNIST"}.get(d.get("dataset", "mnist"), d.get("dataset"))
    inorm = bool(d.get("input_norm"))
    onoff = lambda b: "on" if b else "off"
    n2 = d.get("num_neurons", 400)
    l2_inh = d.get("inhibition_mV", d.get("inhibition_arg", 200.0))
    l2_tp = d.get("theta_plus", d.get("theta_plus_arg"))
    lr2 = d.get("l2_lr_mult", 1.0)
    k2 = d.get("l2_k")
    winners = (f"k = {k2}: the first {k2} neurons to reach threshold share each image (no inhibition among them, none at all "
               f"until there are {k2}); every other neuron is silenced" if k2 else
               "the usual Diehl & Cook competition: the winner's spikes inhibit all others"
               + (" (and at most one spike per millisecond, tie-break)" if d.get("l2_wta") else ""))

    L = [MARK, f"# {run.name}", ""]
    info = live if live is not None else logs.get(run.name)
    minutes = (info or {}).get("minutes") or d.get("minutes")
    kind = "single-layer reference (pixels straight into one Diehl & Cook layer)" if single else "two-layer network"
    where = f"{kind} · {data}" + (" with per-image input normalisation" if inorm else "") + \
            f" · seed {d['seed']} · {d['num_epochs']} epochs × 60,000 images"
    if info and info.get("job"):
        where += f" · Unity job `{info['job']}` on {info.get('node')} ({info.get('gpu')})"
    elif info:
        where += f" · run locally ({info.get('gpu')})"
    if minutes:
        where += f", {minutes:.0f} min"
    L += [where, ""]

    if not single:
        P, S, F = d["patch"], d["stride"], d["features"]
        side = (28 - P) // S + 1
        n1 = side * side * F
        rd = d.get("l1_readout_inhibition")
        rd_eff = rd if rd is not None else d["l1_inhibition"]
        pool = bool(d.get("pool2"))
        L += [f"**In short:** layer 1 = {'shared' if d.get('shared') else 'unshared'} {P}×{P} windows at stride {S} px "
              f"({side}×{side} windows × {F} features = {n1:,} neurons), learned at {mv(d['l1_inhibition'])} and read at "
              f"{mv(rd_eff)}; tie-break {onoff(d.get('l1_wta'))}. "
              + ("2×2 spike-OR pooling between the layers. " if pool else "")
              + f"Layer 2 = {n2} neurons over all of layer 1, calibrated for {d.get('l2_drive_arg', 1.2):g} mV/ms, "
              f"STDP × {lr2:g}, lateral inhibition {mv(l2_inh)}, "
              + (f"**{k2} winners per image**." if k2 else f"tie-break {onoff(d.get('l2_wta'))}."), ""]
    else:
        L += [f"**In short:** {n2} neurons reading the 784 pixels directly (the base Diehl & Cook network, row sum "
              f"{d.get('l2_norm', 78.4):g}, w_max {d.get('l2_wmax', 1.0):g}), lateral inhibition {mv(l2_inh)}, "
              + (f"**{k2} winners per image**." if k2 else f"tie-break {onoff(d.get('l2_wta'))}."), ""]

    from_script = bool(info and info.get("job") in jobs and Path(out_dir).name == run.parent.name)
    if info and info.get("argv") is not None:
        L += ["**Started with** (every setting not on this line is the default):",
              "```", "python multi_layer_unshared.py " + shlex.join(info["argv"]), "```"]
    elif from_script:
        L += [f"**Submitted with** (`unity/submit_multi_layer.sh`, job `{info['job']}`; every other setting is the default):",
              "```", f"python multi_layer_unshared.py {jobs[info['job']][0]}", "```"]
    if from_script and jobs[info["job"]][1]:
        L += [f"Note in the submit script: *{jobs[info['job']][1]}*"]
    if info and (info.get("argv") is not None or from_script):
        L += [""]

    L += ["## Settings", ""]
    if single:
        L += ["| | the one layer |", "|---|---|",
              f"| reads | all 784 pixels |",
              f"| layout | {n2} neurons |",
              f"| competition (lateral inhibition) | among all {n2}, {mv(l2_inh)} |",
              f"| tie-break (`wta`: same-step winners → only the strongest fires) | {onoff(d.get('l2_wta'))}"
              + (" (replaced by the k-winner rule)" if k2 else "") + " |",
              f"| winners per image | {winners} |",
              f"| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | {l2_tp:g}, {tau(d.get('theta_tau'))} |",
              f"| weight scale | row sum {d.get('l2_norm', 78.4):g}, w_max {d.get('l2_wmax', 1.0):g} (not calibrated) |",
              f"| STDP rates lr+ / lr− | × {lr2:g} = {1e-2 * lr2:g} / {1e-4 * lr2:g} |",
              f"| training | {d['num_epochs']} epochs × 60,000 images |", ""]
    else:
        l1d = d.get("l1", {})
        n_in2 = d.get("l2_inputs", n1)
        L += ["| | layer 1 | layer 2 (output) |", "|---|---|---|",
              f"| reads | {P}×{P}-pixel windows of the image | "
              + (f"layer 1 after 2×2 spike-OR pooling ({n_in2:,} lines)" if pool else f"every layer-1 neuron ({n_in2:,} lines)") + " |",
              f"| layout | {side}×{side} windows, stride {S} px, × {F} features = {n1:,} neurons | {n2} neurons |",
              f"| inputs per neuron | {P * P} pixels | {n_in2:,} |",
              f"| weights | {'shared across positions' if d.get('shared') else 'unshared: each window has its own templates'} "
              "| fully connected |",
              f"| competition (lateral inhibition) | among the {F} features of a window | among all {n2} |",
              f"| inhibition while it learns | {mv(d['l1_inhibition'])} (stage A) | {mv(l2_inh)} (stage B) |",
              f"| inhibition afterwards | {mv(rd_eff)} in stage B and scoring"
              + (" (`--l1-readout-inhibition`)" if rd is not None else " (unchanged)") + " | – |",
              f"| tie-break (`wta`: same-step winners → only the strongest fires) | {onoff(d.get('l1_wta'))} | {onoff(d.get('l2_wta'))}"
              + (" (replaced by the k-winner rule)" if k2 else "") + " |",
              f"| winners per image | – (competition is within each window, above) | {winners} |",
              f"| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | {d['l1_theta_plus']:g}, {tau(d.get('l1_theta_tau'))}"
              + (", one θ per feature (shared)" if d.get("l1_theta_shared") else ", one θ per neuron")
              + f" | {l2_tp:g}, {tau(d.get('theta_tau'))} |",
              f"| weight scale | row sum {d['l1_norm']:g} (mean {d['l1_norm'] / (P * P):.3f}), w_max {d['l1_wmax']:g} "
              f"| calibrated for {d.get('l2_drive_arg', 1.2):g} mV/ms from layer 1's measured rate: mean "
              f"{d.get('l2_mean_weight', float('nan')):.4f}, row sum {d.get('l2_norm', float('nan')):.1f}, "
              f"w_max {d.get('l2_wmax', float('nan')):.3f} |",
              f"| STDP rates lr+ / lr− | 1e-2 / 1e-4" + (" (updates averaged over positions)" if d.get("l1_stdp_reduce") == "mean" else "")
              + f" | × {lr2:g} = {1e-2 * lr2:g} / {1e-4 * lr2:g} |",
              f"| training | stage A: up to {d['l1_examples']:,} images"
              + (", stops early once the templates stop changing" if d.get("l1_stop_stationary") else "")
              + (f"; froze at {l1d['frozen_at']:,}" if l1d.get("frozen_at") else "")
              + f" | stage B: {d['num_epochs']} epochs × 60,000 images |",
              f"| pooling | {'2×2 spike-OR into layer 2' if pool else 'none'} | – |", ""]

    L += ["Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold "
          f"−52 mV + θ); pixels → Poisson spikes at 0.1 × intensity per ms, {d['num_steps']} ms per image"
          + ("; each image first scaled to the training set's mean intensity (`--input-norm`, pixels capped at 10)" if inorm
             else ", no input normalisation")
          + "; STDP traces τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32"
          + ("" if single else "; layer 1 frozen after stage A")
          + ". Scoring: neurons labelled from training images 0–1,999; accuracy on test images 0–999 and on the same images "
          f"shifted by up to ±{d.get('shift', 2)} px, every {d['snapshot_interval']} batches ({d['snapshot_interval'] * 32:,} images). "
          "Other keys in run.json (arm, sweep, init, theta_init, anneal_at, top_k, ...) are fixed placeholders for the shared "
          "plotting code.", ""]

    L += ["## Results", ""]
    npz = run / "run.npz"
    if np is not None and npz.exists():
        z = np.load(npz, allow_pickle=True)
        p, a, s = z["points"], z["accuracy"], z["accuracy_shifted"]
        last3 = p > (d["num_epochs"] - 3) * EPOCH
        ep = [a[(p > (k - 1) * EPOCH) & (p <= k * EPOCH)] for k in range(1, d["num_epochs"] + 1)]
        L += [f"- **Held-out / shifted ±{d.get('shift', 2)} px, mean of the snapshots of the last 3 epochs: "
              f"{a[last3].mean():.1f} / {np.nanmean(s[last3]):.1f} %** (best snapshot {a.max():.1f} %; last 3 snapshots "
              f"{d['final_accuracy']:.1f} / {d['final_accuracy_shifted']:.1f} %)",
              "- Per epoch: " + " · ".join(f"{e.mean():.1f}" for e in ep if len(e))]
    elif "final_accuracy" in d:
        L += [f"- Last 3 snapshots, held-out / shifted: {d['final_accuracy']:.1f} / {d['final_accuracy_shifted']:.1f} %"]
    if not single and d.get("l1", {}).get("readout_accuracy_at_freeze") is not None:
        L += [f"- Layer 1 alone (end of stage A, as the output layer): {d['l1']['readout_accuracy_at_freeze']:.1f} / "
              f"{d['l1']['readout_accuracy_shifted_at_freeze']:.1f} %"]
    if not single and d.get("l2_input_rate_per_ms") is not None:
        L += [f"- Layer 1's output into layer 2: {d['l2_input_rate_per_ms']:.2f} spikes per ms per image"]
    if d.get("test_distinct_per_image") is not None:
        pur = d.get("class_purity_median")
        L += [f"- On the 1,000 test digits: {d['test_distinct_per_image']:.1f} distinct layer-2 neurons and "
              f"{d['test_spikes_per_image']:.1f} layer-2 spikes per image; {d['test_neurons_fired']} of {n2} neurons fired; "
              + (f"median class purity {pur:.2f} (share of a neuron's spikes from its top digit: ~1 = one-digit prototype)" if pur is not None
                 else "class purity n/a")
              + " — F16 shows what each neuron fired for"]
    if d.get("spikes_per_presentation") is not None:
        L += [f"- Output-layer spikes per image: {d['spikes_per_presentation']:.1f}"]
    if "minutes" in d:
        L += [f"- Run time: {d['minutes']:.0f} min"]
    L += ["", "The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.", ""]
    return "\n".join(L)


def write_run_readme(run_dir, argv: list | None = None, hardware: str | None = None) -> None:
    """Called by multi_layer_unshared.py at the end of a run: writes run_dir/README.md from the run.json / run.npz just saved,
    with the exact command line (argv) and, on Unity, the Slurm job name and node; the submit script's comment for that job if
    it has one."""
    run = Path(run_dir)
    target = run / "README.md"
    if target.exists() and MARK not in target.read_text(errors="replace"):
        return
    on_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    live = {"argv": list(argv) if argv is not None else None, "gpu": hardware,
            "job": os.environ.get("SLURM_JOB_NAME") if on_slurm else None, "node": os.environ.get("SLURMD_NODENAME")}
    out_dir, jobs = submit_jobs(ROOT / "unity" / "submit_multi_layer.sh")
    target.write_text(describe(run, jobs, out_dir, {}, live))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="batch folders (every run folder inside) or run folders")
    p.add_argument("--submit", default=str(ROOT / "unity" / "submit_multi_layer.sh"), help="submit script to take job flags from")
    p.add_argument("--logs", default=str(ROOT / "unity" / "logs"), help="folder of Slurm logs (job name, node, run time)")
    a = p.parse_args()
    out_dir, jobs = submit_jobs(Path(a.submit))
    logs = unity_logs(Path(a.logs))
    runs = []
    for x in map(Path, a.paths):
        runs += [x] if (x / "run.json").exists() else sorted(r.parent for r in x.glob("*/run.json"))
    for run in runs:
        target = run / "README.md"
        if target.exists() and MARK not in target.read_text(errors="replace"):
            print(f"skip (has its own README.md): {run}")
            continue
        target.write_text(describe(run, jobs, out_dir, logs))
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
