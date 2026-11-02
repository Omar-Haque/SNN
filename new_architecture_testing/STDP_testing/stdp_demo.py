"""
stdp_demo.py -- two mechanism demonstrations recorded during one baseline training run (theta_plus 0.05, uniform
init, 200 mV lateral inhibition, hard WTA off: the network of hypo_1 / the replicate file; dc_model.py is verbatim).

TEST 1 -- "the first neuron to fire takes the whole presentation"
  At the first batch at or after each example count in --raster-at (default 3200 and 32000: early and mid plateau;
  add e.g. 150000 with --examples 160000 for the post-spike regime), every output spike of every neuron is logged for
  every timestep of the batch's 32 presentations (a 350 x 32 x 400 raster), together with the membrane potential of
  all 400 neurons at every ms for the first image of the batch and theta. The figures show, per presentation, which
  neurons spiked in what order, how many distinct neurons ever fired, the winner's inter-spike interval, and the
  membrane potentials of the winner and its runners-up (the -200 mV kicks they receive at every winner spike).

TEST 2 -- "the template imprints much faster than theta catches up"
  From --watch-from examples on (default 20000, so a trained population exists), a neuron whose template is still flat
  (top-100 share < 0.35 at the start of the batch) and that wins at least one presentation in the batch (>= --win-spikes
  spikes on one image) is a fresh hijacker. For the first --n-recruits such neurons the run records:
    * the template every --frame-every ms (default 50) during the batch of the first win and the next
      --inbatch-batches batches (the imprint forming within presentations), and at the end of every batch for
      --track-batches batches (default 300 = 9,600 examples);
    * the neuron's theta, spikes and presentations won per batch, and the trained population's theta (median, p10)
      as the reference it has to catch up with;
    * the image it won first (what it imprinted) and its label.
  The figures show the template filmstrip next to the won image, template sharpness / similarity-to-final against
  examples since the first win, and theta against the same axis with the trained population's level -- the two
  timescales on one page -- plus a summary of "examples to a formed template" vs "examples to the trained theta level"
  for every recruit.

USAGE  python stdp_demo.py                      # ~15 min on MPS for 60k examples
       python stdp_demo.py --examples 160000 --raster-at 3200,32000,150000      # includes the post-spike raster
       python stdp_demo.py --smoke             # ~1 min
       python stdp_demo.py --replot results/<run>
"""
import argparse, json, math, random, time, sys
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from snntorch import spikegen
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dc_model import DiehlAndCookNetwork
from stdp_demo_plots import make_figures

script_dir = Path(__file__).resolve().parent
DATA_ROOT = script_dir.parents[1] / "data"
results_root = script_dir / "results"
train_dataset = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transforms.ToTensor())

TOP_K, FLAT = 100, 0.35


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--theta-plus", type=float, default=0.05); p.add_argument("--theta-tau", type=float, default=None)
    p.add_argument("--inhibition", type=float, default=200.0); p.add_argument("--neurons", type=int, default=400)
    p.add_argument("--examples", type=int, default=60_000, help="training examples to run (1 epoch = 60000)")
    p.add_argument("--raster-at", default="3200,32000", help="example counts at which to log a full spike raster for one batch")
    p.add_argument("--watch-from", type=int, default=20_000, help="start looking for fresh hijackers after this many examples")
    p.add_argument("--n-recruits", type=int, default=6); p.add_argument("--track-batches", type=int, default=300)
    p.add_argument("--inbatch-batches", type=int, default=3, help="batches (incl. the first-win batch) with within-presentation template frames")
    p.add_argument("--frame-every", type=int, default=50, help="ms between within-presentation template frames")
    p.add_argument("--win-spikes", type=int, default=15, help="spikes on one image that count as 'won the presentation'")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=350)
    p.add_argument("--tag", default=""); p.add_argument("--smoke", action="store_true"); p.add_argument("--replot", default=None, metavar="RUN_DIR")
    args = p.parse_args()
    if args.replot:
        make_figures(Path(args.replot)); print("figures regenerated in", args.replot); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    if args.smoke:
        args.examples, args.watch_from, args.raster_at = 32 * 40, 32 * 10, "320"
        args.track_batches, args.inbatch_batches, args.steps, args.frame_every, args.win_spikes = 12, 2, 25, 5, 3
    raster_at = sorted(int(x) for x in args.raster_at.split(",") if x.strip())
    subset = Subset(train_dataset, range(0, min(len(train_dataset), args.examples)))
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(subset, batch_size=32, shuffle=True, generator=gen)
    tau = 1e7 if args.theta_tau is None else args.theta_tau * args.steps / 32
    model = DiehlAndCookNetwork(theta_plus=args.theta_plus, inhibition_mV=args.inhibition, theta_decay_tau_steps=tau, num_neurons=args.neurons).to(device)
    n, T = args.neurons, args.steps

    run_name = f"demo_tp{args.theta_plus:g}_inh{args.inhibition:g}_ex{args.examples // 1000}k_seed{args.seed}" + (f"_{args.tag}" if args.tag else "") + ("_SMOKE" if args.smoke else "") + "_" + time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy(); cfg.update(run_name=run_name, raster_at_list=raster_at, device=str(device))
    print("config:", json.dumps(cfg))

    rasters: list[dict] = []
    recruits: list[dict] = []                        # tracked hijackers
    pending_raster = list(raster_at)
    total = 0; t0 = time.time(); batch_i = 0
    W_param = model.synapses.weight

    for data, targets in loader:
        B = data.shape[0]; data_d = data.to(device)
        spike_data = spikegen.rate(data_d * 0.1, num_steps=T).view(T, B, 784)        # type: ignore
        with torch.no_grad():
            flat_before = (W_param.topk(TOP_K, dim=1).values.sum(1) / 78.4) < FLAT   # (n,) templates still ~ random init
        do_raster = bool(pending_raster) and total >= pending_raster[0]
        # within-presentation template frames are kept for every batch once we are watching (we only learn at the end of
        # the batch whether a fresh hijacker appeared in it); for already-tracked recruits in their in-batch phase too.
        want_frames = total >= args.watch_from and (len(recruits) < args.n_recruits or any(r["batches"] < args.inbatch_batches for r in recruits if r["batches"] < args.track_batches))
        frames: list[torch.Tensor] = []
        if want_frames: frames.append(W_param.detach().clone())                       # frame at 0 ms (start of the batch)
        if do_raster:
            spikes_log = torch.zeros(T, B, n, dtype=torch.uint8, device=device); v_log = torch.zeros(T, n, device=device); theta_log = torch.zeros(T, n, device=device)
            v_all = torch.zeros(T, B, n, device=device)                 # every presentation's potentials; only the biggest-tie one is kept
        model.reset(B, device)
        bts = torch.zeros(B, n, device=device)
        for t in range(T):
            out = model(spike_data[t])
            bts += out
            if do_raster:
                spikes_log[t] = out.to(torch.uint8); v_log[t] = model.neurons.raw_v[0]; theta_log[t] = model.neurons.theta; v_all[t] = model.neurons.raw_v      # type: ignore
            if want_frames and (t + 1) % args.frame_every == 0:
                frames.append(W_param.detach().clone())
        total += B; batch_i += 1
        with torch.no_grad():
            theta = model.neurons.theta.detach().float().cpu().numpy().copy()               # type: ignore
            sharp_now = (W_param.topk(TOP_K, dim=1).values.sum(1) / 78.4).cpu().numpy()
            trained = sharp_now >= FLAT
            th_med = float(np.median(theta[trained])) if trained.any() else float("nan")
            th_p10 = float(np.percentile(theta[trained], 10)) if trained.any() else float("nan")
            bts_c = bts.cpu().numpy()
        # ---- TEST 1: raster ----
        if do_raster:
            pending_raster.pop(0)
            # ties: several neurons crossing threshold in the same millisecond (inhibition only arrives at the next step).
            # Keep the potentials of the presentation whose FIRST spiking millisecond had the most neurons in it.
            sp_np = spikes_log.cpu().numpy(); per_ms = sp_np.sum(2)                              # (T, B) spikes per ms
            first_ms = np.array([int(np.argmax(per_ms[:, b] > 0)) if per_ms[:, b].sum() else -1 for b in range(B)])
            first_size = np.array([int(per_ms[first_ms[b], b]) if first_ms[b] >= 0 else 0 for b in range(B)])
            tie_b = int(np.argmax(first_size))
            rasters.append({"examples": total - B, "spikes": sp_np, "v_img0": v_log.cpu().numpy().astype(np.float32),
                            "theta_t": theta_log.cpu().numpy().astype(np.float32),        # theta of every neuron at every ms of the batch
                            "v_tie": v_all[:, tie_b].cpu().numpy().astype(np.float32), "tie_pres": tie_b,   # potentials on the biggest-tie presentation
                            "theta": theta, "labels": targets.numpy().copy(), "trained": trained.copy()})
            del v_all
            print(f"[raster] logged batch at {total - B} examples: distinct neurons per presentation = {np.mean((bts_c > 0).sum(1)):.2f}, winner spikes = {np.mean(bts_c.max(1)):.1f}, "
                  f"presentations whose first spike is a tie = {int((first_size > 1).sum())}/{B} (biggest: {int(first_size.max())} neurons at once, presentation {tie_b})")
        # ---- TEST 2: fresh hijackers ----
        if total >= args.watch_from and len(recruits) < args.n_recruits:
            max_per_img = bts_c.max(0)
            tracked_ids = {r["neuron"] for r in recruits}
            fresh = np.where(flat_before.cpu().numpy() & (max_per_img >= args.win_spikes))[0]
            for i in fresh:
                if len(recruits) >= args.n_recruits or int(i) in tracked_ids: continue
                b_won = int(np.argmax(bts_c[:, i]))
                recruits.append({"neuron": int(i), "first_win_examples": total - B, "won_image": data[b_won, 0].numpy().copy(), "won_label": int(targets[b_won]),
                                 "won_spikes": int(bts_c[b_won, i]), "theta_at_win": float(theta[i]), "batches": 0,
                                 "inbatch_frames": [], "inbatch_ms": [], "frames": [], "theta": [], "spikes": [], "wins": [], "th_med": [], "th_p10": [], "examples": []})
                print(f"[recruit] neuron {i} won its first presentation at {total - B} examples (label {int(targets[b_won])}, {int(bts_c[b_won, i])} spikes); theta {theta[i]:.1f} vs trained median {th_med:.1f}")
        for r in recruits:
            if r["batches"] >= args.track_batches: continue
            i = r["neuron"]
            if r["batches"] < args.inbatch_batches and frames:
                for j, Wf in enumerate(frames):
                    r["inbatch_frames"].append(Wf[i].cpu().numpy().astype(np.float32)); r["inbatch_ms"].append(r["batches"] * T + j * args.frame_every)
            r["frames"].append(W_param[i].detach().cpu().numpy().astype(np.float32)); r["theta"].append(float(theta[i]))
            r["spikes"].append(float(bts_c[:, i].sum())); r["wins"].append(int((bts_c[:, i] >= args.win_spikes).sum()))
            r["th_med"].append(th_med); r["th_p10"].append(th_p10); r["examples"].append(total - r["first_win_examples"])
            r["batches"] += 1
        if batch_i % 100 == 0:
            print(f"{total:>7d} ex | flat templates {int((~trained).sum()):3d} | trained θ med {th_med:6.1f} | recruits tracked {len(recruits)} | {(time.time() - t0) / 60:4.1f} min")
        if total >= args.examples: break
    print("done training;", len(rasters), "rasters,", len(recruits), "recruits")

    # ---- save ----
    npz: dict = {"theta_final": model.neurons.theta.detach().float().cpu().numpy(), "W_final": W_param.detach().cpu().numpy().astype(np.float16)}   # type: ignore
    for k, r in enumerate(rasters):
        for key, val in r.items(): npz[f"raster{k}_{key}"] = np.asarray(val)
    for k, r in enumerate(recruits):
        for key in ("won_image", "won_label", "won_spikes", "theta_at_win", "neuron", "first_win_examples"): npz[f"rec{k}_{key}"] = np.asarray(r[key])
        for key in ("inbatch_frames", "inbatch_ms", "frames", "theta", "spikes", "wins", "th_med", "th_p10", "examples"): npz[f"rec{k}_{key}"] = np.asarray(r[key], dtype=np.float32)
    np.savez_compressed(run_dir / "run.npz", **npz)
    cfg.update(n_rasters=len(rasters), n_recruits_found=len(recruits), minutes=(time.time() - t0) / 60, examples_run=total)
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))
    make_figures(run_dir)
    print(f"finished in {cfg['minutes']:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
