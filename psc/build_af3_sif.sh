#!/usr/bin/env bash
# Convert the local AlphaFold 3 Docker image into a Singularity/Apptainer .sif
# for Bridges-2. Run inside WSL Ubuntu (not PowerShell):
#
#     wsl -d Ubuntu
#     cd /mnt/c/code_projects/GenAIBiomedicineAssignmentOne
#     bash psc/build_af3_sif.sh
#
# BEFORE YOU RUN THIS, check whether you need it at all. On Bridges-2:
#     module spider alphafold
#     ls /ocean/containers/
# If PSC already ships an AlphaFold 3 image, use theirs and skip this entirely -
# the .sif produced here is several GB and slow to upload from a home connection.
#
# Why build locally: Bridges-2 gives you no root, and `apptainer build` from a
# definition normally needs it. Your laptop has Docker and root, so build here.
# (PSC may permit `apptainer build --fakeroot`; if so, that avoids the upload.)
set -euo pipefail

OUT="${1:-$HOME/af3/alphafold3.sif}"

if ! docker image inspect alphafold3:latest >/dev/null 2>&1; then
  echo "ERROR: docker image 'alphafold3' not found." >&2
  echo "Build it first: docker build -t alphafold3 -f alphafold3/docker/Dockerfile alphafold3" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "== installing apptainer =="
  sudo apt-get update -qq
  sudo apt-get install -y -qq software-properties-common
  sudo add-apt-repository -y ppa:apptainer/ppa
  sudo apt-get update -qq
  sudo apt-get install -y -qq apptainer
fi

# Apptainer cannot read Docker's local image store directly, so publish the
# image to a throwaway registry on localhost and pull from there.
echo "== starting a temporary local registry =="
docker rm -f af3-registry >/dev/null 2>&1 || true
docker run -d -p 5000:5000 --name af3-registry registry:2 >/dev/null
cleanup() { docker rm -f af3-registry >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker tag alphafold3 localhost:5000/alphafold3
echo "== pushing to the local registry (a few minutes for 14 GB) =="
docker push localhost:5000/alphafold3

mkdir -p "$(dirname "$OUT")"
echo "== building ${OUT} =="
SINGULARITY_NOHTTPS=1 APPTAINER_NOHTTPS=1 \
  apptainer build --force "$OUT" docker://localhost:5000/alphafold3:latest

echo
ls -lh "$OUT"
cat <<EOF

Built. Verify it can see a GPU (run this on a Bridges-2 compute node, not here):
  apptainer exec --nv alphafold3.sif nvidia-smi

Upload it (expect a long transfer):
  rsync -P "$OUT" <user>@data.bridges2.psc.edu:/ocean/projects/<group>/<user>/af3/
EOF
