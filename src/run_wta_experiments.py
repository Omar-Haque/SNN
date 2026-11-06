#!/usr/bin/env python3
"""
run_wta_experiments.py -- 100 neurons, inhibition 17.5 mV  x  WTA mode {off, vt, vt_reset}  x  presentation time {150, 200, 250, 350} steps.

WTA modes (see excitatory_neurons in snn_exp.py):
    off       no WTA
    vt        hard WTA, winner = highest (v - theta); losers keep their voltage
    vt_reset  hard WTA, winner = highest (v - theta); losers are reset to rest   <- the "real" hard WTA

    python run_wta_experiments.py --list                 # show the 12 runs
    python run_wta_experiments.py --quick                # tiny smoke test into results_wta_quick/ (a few minutes)
    python run_wta_experiments.py --jobs 4               # the full batch (8 runs, 4 at a time)
    python run_wta_experiments.py --only vt_reset_t150,off_t150 # pick runs by name
    python run_wta_experiments.py --plot_only            # redraw the figures (works while runs are still going)

Each run writes to results_wta_n100/<name>/ (config.json, cmd.txt, log.txt, curves.npz, accuracy_curve.png, DONE.json when
finished). Finished runs are skipped on the next call (--force to redo). When everything is done, plot_wta_compare.py
is called automatically and writes the figures + summary.csv into results_wta_n100/figures/.

Everything except --wta_mode and --num_steps is IDENTICAL across runs (inh 17.5, 100 neurons, 3 epochs, batch 32,
theta_plus 0.05/16, no refractory period, lr_scale 1, intensity 0.1, seed 0).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = "snn_exp.py"

INH = 17.5
MODES = ("off", "vt", "vt_reset")
STEPS = (150, 200, 250, 350)

# Fixed for every run. Only `wta_mode` and `num_steps` vary. Edit here to change e.g. the seed for the whole batch.
COMMON = dict(inh=INH, n_neurons=100, epochs=3, batch_size=32, theta_plus=0.05, theta_div=0, refrac=0, lr_scale=1.0,
              intensity=0.1, seed=0, snapshot_interval=50)

QUICK = dict(max_train=2240, epochs=1, snapshot_interval=10, assign_n=200, eval_n=100)


def run_name(mode: str, steps: int) -> str:
    return f"{mode}_t{steps}"


EXPERIMENTS: dict[str, dict] = {
    run_name(m, s): dict(wta_mode=m, steps=s, desc=f"100 neurons | inh {INH} mV | WTA {m:8s} | {s} steps")
    for m in MODES for s in STEPS
}


def flags(d: dict) -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        if v is True:
            out.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            out += [f"--{k}", str(v)]
    return out


def build_cmd(name: str, spec: dict, results_dir: str, quick: bool) -> list[str]:
    args = dict(COMMON, num_steps=spec["steps"], wta_mode=spec["wta_mode"])
    if quick:
        args.update(QUICK)
    return [sys.executable, SCRIPT, "--name", name, "--results_dir", results_dir] + flags(args)


def run_one(name: str, spec: dict, results_dir: str, quick: bool, threads: int) -> tuple[str, int, float]:
    run_dir = HERE / results_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(name, spec, results_dir, quick)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")
    (run_dir / "DONE.json").unlink(missing_ok=True)   # a re-run is not "done" until it finishes again
    t0 = time.time()
    with open(run_dir / "log.txt", "w") as log:
        rc = subprocess.call(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if rc == 0:
        (run_dir / "DONE.json").write_text(json.dumps(dict(name=name, minutes=round(dt / 60, 2), cmd=cmd, quick=quick)))
    return name, rc, dt


def prepare_data() -> None:
    """Download MNIST once, so parallel jobs do not race on the download."""
    from torchvision import datasets
    datasets.MNIST(root=str(HERE / "data"), train=True, download=True)
    datasets.MNIST(root=str(HERE / "data"), train=False, download=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of run names (default: all 12)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel jobs (small SNNs under-use the GPU; 2-4 usually helps)")
    ap.add_argument("--quick", action="store_true", help="tiny runs into results_wta_n100_quick/ to test the pipeline")
    ap.add_argument("--force", action="store_true", help="re-run even if DONE.json exists")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    a = ap.parse_args()

    results_dir = "results_wta_n100_quick" if a.quick else "results_wta_n100_theta_div0"

    if a.list:
        for n, s in EXPERIMENTS.items():
            print(f"{n:9s} {s['desc']}")
        return

    if not a.plot_only:
        names = [n.strip() for n in a.only.split(",") if n.strip()] if a.only else list(EXPERIMENTS)
        unknown = [n for n in names if n not in EXPERIMENTS]
        if unknown:
            sys.exit(f"unknown run names: {unknown}  (see --list)")
        todo = [n for n in names if a.force or not (HERE / results_dir / n / "DONE.json").exists()]
        # longest presentation first, so the parallel slots are used well
        todo.sort(key=lambda n: -EXPERIMENTS[n]["steps"])
        print(f"{len(todo)} run(s) to do: {todo}   (skipping {sorted(set(names) - set(todo))})")
        if a.dry_run:
            for n in todo:
                print(" ".join(build_cmd(n, EXPERIMENTS[n], results_dir, a.quick)))
            return
        if todo:
            prepare_data()
            threads = max(1, (os.cpu_count() or 2) // max(1, min(a.jobs, len(todo))))
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                futs = [ex.submit(run_one, n, EXPERIMENTS[n], results_dir, a.quick, threads) for n in todo]
                for f in as_completed(futs):
                    n, rc, dt = f.result()
                    status = "ok" if rc == 0 else f"FAILED (rc={rc}, see {results_dir}/{n}/log.txt)"
                    print(f"[{(time.time() - t0) / 60:7.1f} min] {n}: {status}  ({dt / 60:.1f} min)", flush=True)

    if not a.no_plot:
        subprocess.call([sys.executable, "plot_wta_compare.py", "--results_dir", results_dir], cwd=HERE)


if __name__ == "__main__":
    main()
