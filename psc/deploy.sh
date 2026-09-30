#!/usr/bin/env bash
# Copy this project to Bridges-2. Run from the repository root, inside WSL
# (Git Bash has ssh but no rsync):
#
#     bash psc/deploy.sh <psc-username>
#
# The group is resolved from $PROJECT on the cluster, so you do not have to
# know it. Pass it as a second argument only to override that.
#
# Sends only what the cluster needs: the assignment Python, the GFP data, the
# af3_eval helpers and these job scripts. Skips the local venv, the pretrained
# checkpoint (setup_env.sh re-downloads it there, which is far faster than
# pushing 606 MB over your home connection), any local checkpoints, and lectures/.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: bash psc/deploy.sh <psc-username> [project-group]

  <psc-username>   your PSC username (NOT your ACCESS ID)
  [project-group]  optional. The DIRECTORY name under /ocean/projects, e.g.
                   cis261594p -- not the allocation's display title. Omit it
                   and the script asks the cluster for $PROJECT directly,
                   which is the reliable way.
USAGE
  exit 2
}

[ $# -ge 1 ] && [ $# -le 2 ] || usage
USER_NAME="$1"
GROUP="${2:-}"

DTN="data.bridges2.psc.edu"   # Data transfer node - not the login node.

# This script pushes your laptop's copy TO the cluster. Run on the cluster it
# would rsync the deployed tree onto itself, and could never transfer a file
# that only exists locally - which is the whole point of running it.
if [ -d /ocean/projects ] || case "$(hostname -f 2>/dev/null || hostname)" in
     *bridges2*|*psc.edu) true ;; *) false ;; esac
then
  echo "ERROR: this looks like a PSC machine - deploy.sh runs on your LAPTOP." >&2
  echo >&2
  echo "It copies the local repository to Bridges-2, so running it here does" >&2
  echo "nothing useful. From the repo root on your laptop, in WSL:" >&2
  echo "  bash psc/deploy.sh ${USER_NAME}" >&2
  exit 1
fi

if [ -n "${GROUP}" ]; then
  # The allocation title ("CIS261594: 11781 Generative AI...") is not the group.
  # Unquoted it also word-splits into a broken path, so reject anything that
  # cannot be a single path component.
  case "${GROUP}" in
    *[!A-Za-z0-9_-]*)
      echo "ERROR: '${GROUP}' is not a valid group directory name." >&2
      echo "It must be one path component: letters, digits, '_' or '-' only." >&2
      echo "You probably passed the allocation title instead of the group." >&2
      echo >&2
      usage
      ;;
  esac
  PROJECT_DIR="/ocean/projects/${GROUP}/${USER_NAME}"
else
  # Ask the cluster instead of guessing. Group names often carry a suffix
  # letter (cis261594p, not cis261594), which is easy to get wrong by hand.
  echo "Resolving your project directories on ${DTN} ..."
  # Feed the snippet to `bash -s` on stdin rather than passing it as an ssh
  # command string. Transfer nodes often run a restricted or non-POSIX login
  # shell, and at least one of them hands `2>/dev/null` to ls as a FILENAME
  # instead of treating it as a redirect. Forcing bash sidesteps the question
  # of what the remote login shell is.
  #
  # `|| true` so an empty result is an empty string, not a fatal exit under
  # `set -e` - the "no allocation found" message below is far more useful than
  # a silent abort.
  ALL_PROJECTS="$(ssh "${USER_NAME}@${DTN}" bash -s <<'REMOTE' || true
shopt -s nullglob
for d in /ocean/projects/*/"$USER"; do
  [ -d "$d" ] && printf '%s\n' "$d"
done
REMOTE
)"

  if [ -z "${ALL_PROJECTS}" ]; then
    echo "ERROR: could not find any project directory for ${USER_NAME}." >&2
    echo "Log in and check manually:" >&2
    echo "  ssh ${USER_NAME}@bridges2.psc.edu" >&2
    echo "  echo \$PROJECT; projects" >&2
    exit 1
  fi

  # Being in more than one allocation is common, and $PROJECT just points at
  # one of them. Picking the wrong one spends another project's SUs, so ask
  # rather than assume.
  PROJECT_COUNT="$(printf '%s\n' "${ALL_PROJECTS}" | wc -l)"
  if [ "${PROJECT_COUNT}" -gt 1 ]; then
    echo "ERROR: you belong to ${PROJECT_COUNT} allocations:" >&2
    printf '%s\n' "${ALL_PROJECTS}" | sed 's|/ocean/projects/\([^/]*\)/.*|  \1|' >&2
    echo >&2
    echo "Which one is the course allocation? Run this to match titles to groups:" >&2
    echo "  ssh ${USER_NAME}@bridges2.psc.edu 'projects'" >&2
    echo >&2
    echo "Then pass it explicitly:" >&2
    echo "  bash psc/deploy.sh ${USER_NAME} <group>" >&2
    exit 1
  fi

  PROJECT_DIR="${ALL_PROJECTS}"
  echo "Found: ${PROJECT_DIR}"
fi

DEST="${PROJECT_DIR}/gfp"

# rsync only creates the final directory, and the remote rsync here predates
# --mkpath, so make the tree first. This also fails loudly with a useful
# message if the project path is wrong, instead of a cryptic rsync IO error.
echo "Creating ${DEST} ..."
if ! ssh "${USER_NAME}@${DTN}" "mkdir -p '${DEST}'"; then
  echo "ERROR: could not create ${DEST}" >&2
  echo "The project directory is probably wrong. Check with:" >&2
  echo "  ssh ${USER_NAME}@bridges2.psc.edu 'echo \$PROJECT; ls -d /ocean/projects/*/\$USER'" >&2
  exit 1
fi

echo "Deploying to ${USER_NAME}@${DTN}:${DEST}"

# -r recursive, -l links, -t times, -p perms, -v verbose, -P resume+progress.
# --relative keeps the paths below intact under $DEST.
rsync -rltpvP --relative \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -e ssh \
  ./assignment/dataset.py \
  ./assignment/inference.py \
  ./assignment/model.py \
  ./assignment/progen_vocab.py \
  ./assignment/train.py \
  ./assignment/data \
  ./af3_eval \
  ./psc \
  ./alphafold3/fetch_databases.sh \
  "${USER_NAME}@${DTN}:${DEST}/"

cat <<EOF

Done. Next, on Bridges-2:

  ssh ${USER_NAME}@bridges2.psc.edu
  cd ${DEST}
  bash psc/setup_env.sh          # conda env + ProGen2-small, ~10 min
  sbatch psc/train_progen.job    # GPU-small partition, ~2.9 h typical queue
  squeue -u \$USER

EOF
