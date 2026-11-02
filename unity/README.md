# Running this project on Unity (unityhpc.org)

Nothing in the code is Mac-specific: every script picks `mps → cuda → cpu`, so on a Unity GPU node it prints
"Using CUDA." and runs as-is. Results land in the same `results/` folders as on the Mac. (CUDA and MPS draw different
random numbers from the same seed, so a Unity run is a new seed in effect, not a bit-for-bit repeat of a Mac run.)

## One-time setup
1. SSH key: account portal → SSH Keys → Generate Key; move the downloaded private key to `~/.ssh/`, `chmod 600` it,
   and add to `~/.ssh/config`:
   ```
   Host unity
       HostName login.unityhpc.org
       User <username>_<organization>_edu
       IdentityFile ~/.ssh/unity-privkey.key
   ```
   Then `ssh unity`. (Open OnDemand at https://ood.unity.rc.umass.edu gives a browser shell and file browser instead.)
2. Pick a place: `/work/pi_<your PI's username>/` is the intended location for job files (1 TB per group);
   `ls /work` / `id` show your groups. If you are not in a PI group yet, `~` (100 GB) works for this project — the
   whole repo with results is well under 1 GB.
3. Get the code and build the environment (login node, has internet):
   ```
   cd /work/pi_<pi>/        # or cd ~
   git clone https://github.com/Omar-Haque/SNN.git cooper     # or rsync the folder from the Mac, see below
   cd cooper
   bash unity/setup_env.sh  # conda env "snn" (torch, torchvision, snntorch, numpy, matplotlib) + MNIST into data/
   ```
   `data/` and `temp/` are git-ignored, so MNIST must be downloaded on Unity; the setup script does it.

## Running
Interactive check (30 min on a GPU node, shows the queue is fine and the code sees CUDA):
```
salloc -p gpu-preempt -t 00:30:00 --gpus=1 --mem=8G -c 4
module load conda/latest && conda activate snn
cd new_architecture_testing/snn_multi_layered_unshared && python multi_layer_unshared.py --smoke
exit
```
Batch jobs, submitted from the repo root (`unity/run.sbatch` wraps any script; logs in `unity/logs/`):
```
sbatch unity/run.sbatch new_architecture_testing/snn_multi_layered_unshared/multi_layer_unshared.py --l1-wta --l2-wta --seed 0
sbatch -p gpu-preempt -t 01:00:00 -J stdp unity/run.sbatch new_architecture_testing/STDP_testing/stdp_demo.py
bash unity/submit_multi_layer.sh          # all variants of run_multi_layer_unshared.sh, as parallel jobs
squeue --me                               # state; PD = waiting, R = running
tail -f unity/logs/<name>-<jobid>.out     # the console output you normally watch
scancel <jobid>
```
Partitions: `gpu-preempt` for anything under 2 h (shortest wait; a preempted job is killed, and these scripts do not
checkpoint, so resubmit), `gpu` for longer runs (up to 48 h). Replotting/summaries are light enough for the login node.

## Getting results back
From the Mac (or push from Unity and pull):
```
rsync -rtlp unity:/work/pi_<pi>/cooper/new_architecture_testing/snn_multi_layered_unshared/results/ \
      ~/Desktop/Main/Labs/SNN/cooper/new_architecture_testing/snn_multi_layered_unshared/results/
```
To send the Mac copy up instead of cloning: `rsync -rtlp --exclude temp --exclude data --exclude __pycache__ ~/Desktop/Main/Labs/SNN/cooper unity:/work/pi_<pi>/`
