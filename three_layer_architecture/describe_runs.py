"""Write a README.md into each three-layer run folder that spells out the run's settings and its main results.

The folder names only carry a few tags (e.g. 3L_l1p7s2f16rd8a2inh200_l2b3s1f16_l2rd4_l3n400_ep6_seed0_...); this writes the
full set of settings, layer by layer, from the run's run.json, plus the Unity job it came from when the batch's logs are
there (job name, node, the flags from unity/submit_three_layer.sh). Needs only Python 3 (no torch, no numpy); run from
anywhere:

    python three_layer_architecture/describe_runs.py three_layer_architecture/results/03_batch3_sparse_trained_layer2_6ep_20261126
    python three_layer_architecture/describe_runs.py RUN_DIR [RUN_DIR ...]        # single run folders work too

A README.md this script wrote earlier is replaced; any other README.md in a run folder is left alone.

three_layer.py calls write_run_readme() at the end of every run, so new run folders get theirs automatically, with the exact
command line the run was started with (and the Slurm job name on Unity).
"""
import argparse
import json
import os
import re
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # repo root (holds unity/)
MARK = "<!-- written by three_layer_architecture/describe_runs.py; rerun it to refresh -->"


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


def describe(run: Path, jobs: dict, out_dir: str, logs: dict, live: dict | None = None) -> str:
    """live (from write_run_readme, at the end of a run): {argv, job, node, gpu}; otherwise the Slurm logs are used."""
    d = json.loads((run / "run.json").read_text())
    l1d, l2d = d.get("l1", {}), d.get("l2", {})
    l1_learn, l1_rd = d["l1_inhibition"], d["l1_readout_inhibition"]
    l1_a2 = d["l1_a2_inhibition"] if d.get("l1_a2_inhibition") is not None else l1_rd
    l2_learn = d["l2_inhibition"]
    l2_rd = d["l2_readout_inhibition"] if d.get("l2_readout_inhibition") is not None else l2_learn
    skip = bool(d["l3_skip"])
    B, s2, F2 = d["l2_block"], d["l2_stride"], d["l2_features"]
    span = (B - 1) * d["l1_stride"] + d["l1_patch"]
    n_l3_in = d["l3_inputs"] + (d["l1_neurons"] if skip and d["l3_inputs"] == d["l2_neurons"] else 0)
    onoff = lambda b: "on" if b else "off"
    tau = lambda t: "no decay" if not t else f"τ {t:,.0f} images"

    L = [MARK, f"# {run.name}", ""]
    info = live if live is not None else logs.get(run.name)
    minutes = (info or {}).get("minutes") or d.get("minutes")
    where = f"three-layer network · MNIST · seed {d['seed']} · {d['num_epochs']} epochs × 60,000 images in stage B"
    if info and info.get("job"):
        where += f" · Unity job `{info['job']}` on {info.get('node')} ({info.get('gpu')})"
    elif info:
        where += f" · run locally ({info.get('gpu')})"
    if minutes:
        where += f", {minutes:.0f} min"
    L += [where, ""]

    # one-paragraph summary of the choices that differ between runs
    L += ["**In short:** layer 1 learned with " + mv(l1_learn) + " within-window inhibition; it then ran at " + mv(l1_a2)
          + " while layer 2 learned and at " + mv(l1_rd) + " in stage B and for scoring. Layer 2 = "
          + f"{B}×{B}-window blocks ({span}×{span} px) at a stride of {s2} window{'s' if s2 > 1 else ''}, "
          + f"{F2} features per block ({d['l2_neurons']:,} neurons); it learned at {mv(l2_learn)} and was read at {mv(l2_rd)}. "
          + f"Layer 3 = {d['l3_neurons']} neurons reading " + ("layer 2 **and layer 1** (skip connection)." if skip else "layer 2 only."),
          ""]

    from_script = bool(info and info.get("job") in jobs and Path(out_dir).name == run.parent.name)
    if info and info.get("argv") is not None:
        L += ["**Started with** (every setting not on this line is the default):",
              "```", "python three_layer.py " + shlex.join(info["argv"]), "```"]
    elif from_script:
        L += [f"**Submitted with** (`unity/submit_three_layer.sh`, job `{info['job']}`; every other setting is the default):",
              "```", f"python three_layer.py {jobs[info['job']][0]}", "```"]
    if from_script and jobs[info["job"]][1]:
        L += [f"Note in the submit script: *{jobs[info['job']][1]}*"]
    if info and (info.get("argv") is not None or from_script):
        L += [""]

    L += ["## Settings", "",
          "| | layer 1 | layer 2 | layer 3 |", "|---|---|---|---|",
          f"| reads | {d['l1_patch']}×{d['l1_patch']}-pixel windows of the image | {B}×{B} neighbouring layer-1 windows "
          f"({span}×{span} px), all {d['l1_features']} features of each | every layer-2 neuron"
          + (" and every layer-1 neuron (skip)" if skip else "") + " |",
          f"| layout | {d['l1_side']}×{d['l1_side']} windows, stride {d['l1_stride']} px, × {d['l1_features']} features "
          f"= {d['l1_neurons']:,} neurons | {d['l2_side']}×{d['l2_side']} blocks, stride {s2} window{'s' if s2 > 1 else ''}, "
          f"× {F2} features = {d['l2_neurons']:,} neurons | {d['l3_neurons']} neurons |",
          f"| inputs per neuron | {d['l1_patch'] ** 2} pixels | {d['l2_inputs']} | {n_l3_in:,} |",
          "| weights | unshared: each window has its own 16 templates | unshared: each block has its own | fully connected |",
          f"| competition (lateral inhibition) | among the {d['l1_features']} features of a window | among the {F2} features of a block "
          f"| among all {d['l3_neurons']} |",
          f"| inhibition while it learns | {mv(l1_learn)} (stage A) | {mv(l2_learn)} (stage A2) | {mv(d['l3_inhibition'])} (stage B) |",
          f"| inhibition afterwards | {mv(l1_a2)} while layer 2 learns (stage A2); {mv(l1_rd)} in stage B and scoring "
          f"| {mv(l2_rd)} in stage B and scoring | – |",
          f"| tie-break (`wta`: same-step winners → only the strongest fires) | {onoff(d['l1_wta'])} | {onoff(d['l2_wta'])} "
          f"| {onoff(d['l3_wta'])} |",
          f"| adaptive threshold: θ+ (θ rises θ+/16 mV per spike), decay | {d['l1_theta_plus']:g}, {tau(d['l1_theta_tau'])} "
          f"| {d['l2_theta_plus']:g}, {tau(d['l2_theta_tau'])} | {d['l3_theta_plus']:g}, {tau(d['l3_theta_tau'])} |",
          f"| weight scale | row sum {d['l1_norm']:g} (mean {d['l1_norm'] / d['l1_patch'] ** 2:.3f}), w_max {d['l1_wmax']:g} "
          f"| calibrated (drive × {d['l2_drive_mult']:g}): mean {d.get('l2_mean_weight', float('nan')):.3f}, "
          f"row sum {d.get('l2_norm', float('nan')):.1f}, w_max {d.get('l2_wmax', float('nan')):.2f} "
          f"| calibrated for {d['l3_drive']:g} mV/ms: mean {d.get('l3_mean_weight', float('nan')):.4f}, "
          f"row sum {d.get('l3_norm', float('nan')):.1f}, w_max {d.get('l3_wmax', float('nan')):.3f} |",
          f"| STDP rates lr+ / lr− | 1e-2 / 1e-4 | × {d['l2_lr_mult']:g} = {1e-2 * d['l2_lr_mult']:g} / {1e-4 * d['l2_lr_mult']:g} "
          f"| × {d['l3_lr_mult']:g} = {1e-2 * d['l3_lr_mult']:g} / {1e-4 * d['l3_lr_mult']:g} |",
          f"| training | stage A: up to {d['l1_examples']:,} images"
          + (", stops early once the templates stop changing" if d["l1_stop_stationary"] else "")
          + (f"; froze at {l1d['frozen_at']:,}" if l1d.get("frozen_at") else "")
          + f" | stage A2: up to {d['l2_examples']:,} images" + (", same early stop" if d["l2_stop_stationary"] else "")
          + (f"; froze at {l2d['frozen_at']:,}" if l2d.get("frozen_at") else "")
          + f" | stage B: {d['num_epochs']} epochs × 60,000 images |",
          "| pooling | none | none | – |",
          "",
          "Fixed in the code for every run: leaky integrate-and-fire neurons (rest and reset −65 mV, τ_m 100 ms, threshold −52 mV + θ); "
          f"pixels → Poisson spikes at 0.1 × intensity per ms, {d['num_steps']} ms per image, no input normalisation; STDP traces "
          "τ 20 ms, weights renormalised to the row sum and clipped to w_max after each update; batches of 32; layers frozen once "
          "their stage ends. Calibration: layer 2's and layer 3's weight scales are set from the measured input rate "
          "(mean weight = drive ÷ rate, w_max = 10 × mean for layer 3). Scoring: neurons labelled from training images 0–1,999; "
          f"accuracy on test images 0–999 and on the same images shifted by up to ±{d['shift']} px, every "
          f"{d['snapshot_interval']} batches ({d['snapshot_interval'] * 32:,} images).",
          ""]

    L += ["## Results", ""]
    if "last3_epochs_accuracy" in d:
        ep = d.get("epoch_accuracy") or []
        L += [f"- **Held-out / shifted ±{d['shift']} px, mean of the snapshots of the last 3 epochs: "
              f"{d['last3_epochs_accuracy']:.1f} / {d['last3_epochs_accuracy_shifted']:.1f} %** "
              f"(best snapshot {d['max_accuracy']:.1f} %; last 3 snapshots {d['final_accuracy']:.1f} / {d['final_accuracy_shifted']:.1f} %)",
              "- Per epoch: " + " · ".join(f"{a:.1f}" for a in ep)]
    else:
        L += ["- (stage B did not finish: no accuracy summary in run.json)"]
    if l1d.get("readout_accuracy_at_freeze") is not None:
        L += [f"- Layer 1 alone (end of stage A, as the output layer): {l1d['readout_accuracy_at_freeze']:.1f} / "
              f"{l1d['readout_accuracy_shifted_at_freeze']:.1f} %"]
    if l2d.get("readout_accuracy_at_freeze") is not None:
        L += [f"- Layer 2 alone (end of stage A2, at its learning inhibition): {l2d['readout_accuracy_at_freeze']:.1f} / "
              f"{l2d['readout_accuracy_shifted_at_freeze']:.1f} %"]
    if "l3_input_rate_per_ms" in d:
        L += [f"- Input into layer 3: {d['l3_input_rate_per_ms']:.2f} spikes per ms per image"
              + (" (both pathways)" if skip else "")]
    if d.get("l2_never_fired_on_test") is not None:
        L += [f"- Never fired on the 1,000 test digits: {d['l2_never_fired_on_test']:,} of {d['l2_neurons']:,} layer-2 neurons, "
              f"{d.get('l3_never_fired_on_test', '?')} of {d['l3_neurons']} layer-3 neurons"]
    if skip and d.get("l3_weight_share_l2") is not None:
        L += [f"- Layer 3's weight on layer-2 lines: {100 * d['l3_weight_share_l2_at_start']:.0f} % at the start → "
              f"{100 * d['l3_weight_share_l2']:.0f} % at the end (it follows layer 2's share of the input spikes; see batch 3's README)"]
    if "minutes" in d:
        L += [f"- Run time: {d['minutes']:.0f} min"]
    L += ["", "The batch's README (one folder up) has the questions behind this run and the comparison with the other runs.", ""]
    return "\n".join(L)


def write_run_readme(run_dir, argv: list | None = None, hardware: str | None = None) -> None:
    """Called by three_layer.py at the end of a run: writes run_dir/README.md from the run.json just saved, with the exact
    command line (argv) and, on Unity, the Slurm job name and node; the submit script's comment for that job if it has one."""
    run = Path(run_dir)
    target = run / "README.md"
    if target.exists() and MARK not in target.read_text(errors="replace"):
        return
    on_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    live = {"argv": list(argv) if argv is not None else None, "gpu": hardware,
            "job": os.environ.get("SLURM_JOB_NAME") if on_slurm else None, "node": os.environ.get("SLURMD_NODENAME")}
    out_dir, jobs = submit_jobs(ROOT / "unity" / "submit_three_layer.sh")
    target.write_text(describe(run, jobs, out_dir, {}, live))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="batch folders (every run folder inside) or run folders")
    p.add_argument("--submit", default=str(ROOT / "unity" / "submit_three_layer.sh"), help="submit script to take job flags from")
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
