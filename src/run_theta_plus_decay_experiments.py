#!/usr/bin/env python3
"""
run_theta_plus_decay_experiments.py -- is there a ratio between theta_plus and theta_decay?  100 neurons.

Grid (theta_div fixed to 1, so theta_plus is the real per-spike increment; original script = 0.05/16 = 0.003125):
    theta_plus  in {0.0008, 0.003125, 0.0125, 0.05}
    theta_decay in 1 - lam, lam in {1e-4, 1e-5, 1e-6, 1e-7}   (lam=1e-7 ~ the original, effectively no decay)
16 runs. K = theta_plus / lam. Neighbouring grid points have K within ~2.5x of each other (e.g. 0.0125@1e-4 K=125
vs 0.003125@1e-5 K=312), and the collapse plot shows whether accuracy follows K across different theta_plus.

    python run_theta_plus_decay_experiments.py --list
    python run_theta_plus_decay_experiments.py --quick          # tiny smoke test -> results_theta_plus_decay_quick/
    python run_theta_plus_decay_experiments.py --jobs 4         # full grid
    python run_theta_plus_decay_experiments.py --plot_only
Finished runs (DONE.json) are skipped; --force redoes them.
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUS = (0.0008, 0.003125, 0.0125, 0.05)
LAM = (1e-4, 1e-5, 1e-6, 1e-7)          # lam = 1 - theta_decay
COMMON = dict(n_neurons=100, epochs=3, batch_size=32, inh=200.0, num_steps=350, intensity=0.1,
              seed=0, snapshot_interval=50)
QUICK = dict(max_train=2240, epochs=1, snapshot_interval=10, assign_n=200, eval_n=100)


def name_of(p, lam):
    return f"p{p:g}_lam{lam:.0e}".replace("+", "")


EXPERIMENTS = {name_of(p, l): dict(theta_plus=p, theta_decay=repr(1.0 - l)) for p in PLUS for l in LAM}


def build_cmd(name, spec, rd, quick):
    args = dict(COMMON, **spec)
    if quick:
        args.update(QUICK)
    out = [sys.executable, "snn_theta_plus_decay.py", "--name", name, "--results_dir", rd]
    for k, v in args.items():
        out += [f"--{k}", str(v)]
    return out


def run_one(name, spec, rd, quick, threads):
    d = HERE / rd / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE.json").unlink(missing_ok=True)
    cmd = build_cmd(name, spec, rd, quick)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(d / "log.txt", "w") as log:
        rc = subprocess.call(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    if rc == 0:
        (d / "DONE.json").write_text(json.dumps(dict(name=name, minutes=round((time.time() - t0) / 60, 2))))
    return name, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    rd = "results_theta_plus_decay_quick" if a.quick else "results_theta_plus_decay"
    if a.list:
        for n, s in EXPERIMENTS.items():
            lam = 1.0 - float(s["theta_decay"])
            print(f"{n:18s} plus={s['theta_plus']:<9g} theta_decay={s['theta_decay']:<22s} K={s['theta_plus']/lam:g}")
        return
    if not a.plot_only:
        names = [n for n in a.only.split(",") if n] or list(EXPERIMENTS)
        bad = [n for n in names if n not in EXPERIMENTS]
        if bad:
            sys.exit(f"unknown: {bad}")
        todo = [n for n in names if a.force or not (HERE / rd / n / "DONE.json").exists()]
        print(f"{len(todo)} run(s) to do")
        if todo:
            from torchvision import datasets
            datasets.MNIST(root=str(HERE / "data"), train=True, download=True)
            datasets.MNIST(root=str(HERE / "data"), train=False, download=True)
            threads = max(1, (os.cpu_count() or 2) // max(1, min(a.jobs, len(todo))))
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                for f in as_completed([ex.submit(run_one, n, EXPERIMENTS[n], rd, a.quick, threads) for n in todo]):
                    n, rc, dt = f.result()
                    print(f"[{(time.time()-t0)/60:6.1f} min] {n}: {'ok' if rc == 0 else f'FAILED rc={rc} (see {rd}/{n}/log.txt)'}", flush=True)
    subprocess.call([sys.executable, "plot_theta_plus_decay_compare.py", "--results_dir", rd], cwd=HERE)


if __name__ == "__main__":
    main()
