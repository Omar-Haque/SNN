#!/usr/bin/env bash
# One-time setup on a Unity LOGIN node (login nodes have internet; run this once after cloning the repo).
#   cd <repo root>  &&  bash unity/setup_env.sh
# Creates the conda env "snn" (in ~/.conda/envs) and downloads MNIST into <repo root>/data, which the scripts expect.
set -euo pipefail
module purge
module load conda/latest
eval "$(conda shell.bash hook)"
if ! conda env list | grep -qE '^snn\s'; then
    conda create -y -n snn python=3.12
fi
conda activate snn
# PyTorch wheels bundle their own CUDA runtime, but each build only carries kernels for some GPU generations. The default
# (newest CUDA) build supports compute capability 7.5+ only, and many Unity GPUs are older (GTX TITAN X / 1080 Ti / V100
# in gpu-preempt, "no kernel image is available for execution on the device"). The CUDA 12.6 build covers them all.
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install snntorch numpy matplotlib
python - <<'PY'
from torchvision import datasets
for train in (True, False):
    datasets.MNIST("data", train=train, download=True)        # -> data/MNIST/raw, the DATA_ROOT every script resolves to
print("MNIST ready in data/MNIST")
PY
python -c "import torch, snntorch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| snntorch', snntorch.__version__)"
echo "done: env 'snn' created, MNIST downloaded"
