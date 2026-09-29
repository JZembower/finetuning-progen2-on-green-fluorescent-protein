#!/usr/bin/env bash
# One-time setup on Bridges-2. Run from the deployed project directory:
#
#     cd /ocean/projects/<group>/<user>/gfp && bash psc/setup_env.sh
#
# Creates the conda environment from the assignment guidelines and downloads
# ProGen2-small. Installing packages is I/O heavy but not compute heavy, so the
# login node is acceptable here; everything else must go through sbatch.
set -euo pipefail

# deploy.sh mirrors the repo layout, so the Python lives in assignment/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p logs                     # sbatch needs this to exist before submission

# $HOME on Bridges-2 is capped at 25 GB, and conda would put the environment
# (~6 GB) plus its package cache (several GB more) there by default, which is
# enough to blow the quota. Keep all of it on Ocean next to the code instead.
# The job scripts activate by this same path.
export ENV_PREFIX="${ROOT}/envs/progen"
export CONDA_PKGS_DIRS="${ROOT}/.conda_pkgs"
mkdir -p "${ROOT}/envs" "${CONDA_PKGS_DIRS}"

cd assignment

echo "== Loading conda =="
module load anaconda3/2024.10-1

# `conda activate` needs the shell hook in a non-interactive script.
eval "$(conda shell.bash hook)"

if [ -d "${ENV_PREFIX}/bin" ]; then
  echo "environment already exists at ${ENV_PREFIX}, skipping creation"
else
  echo "== Creating environment at ${ENV_PREFIX} (python 3.9.16, per the guidelines) =="
  conda create -y -p "${ENV_PREFIX}" python=3.9.16
fi

conda activate "${ENV_PREFIX}"

# --no-cache-dir: the wheels are ~3 GB and pip would cache them in $HOME.
echo "== Installing torch 2.7.0 + cu128 and transformers 4.49.0 (several GB) =="
pip install --no-cache-dir --quiet torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir --quiet transformers==4.49.0

echo "== Downloading ProGen2-small =="
mkdir -p pretrained_model/progen2-small models design
if [ -f pretrained_model/progen2-small/pytorch_model.bin ]; then
  echo "checkpoint already present, skipping download"
else
  wget -q --show-progress -O /tmp/progen2-small.tar.gz \
    https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz
  tar -xzf /tmp/progen2-small.tar.gz -C pretrained_model/progen2-small
  rm -f /tmp/progen2-small.tar.gz
fi

# The package cache is only needed during installation; reclaim the space.
rm -rf "${CONDA_PKGS_DIRS}"

echo
echo "== disk usage =="
echo "environment: $(du -sh "${ENV_PREFIX}" 2>/dev/null | cut -f1) at ${ENV_PREFIX}"
du -sh "${HOME}" 2>/dev/null | awk '{print "  $HOME: "$1" (25 GB quota)"}'

echo
echo "Setup complete. Verify, then submit from ${ROOT}:"
echo "  ls -la ${ROOT}/assignment/pretrained_model/progen2-small"
echo "  cd ${ROOT} && sbatch psc/train_progen.job"
