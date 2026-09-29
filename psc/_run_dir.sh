#!/usr/bin/env bash
# Sourced by the job scripts - not executed directly.
#
# WHY THIS EXISTS
#
# The assignment's Python hardcodes every output path with no run identifier:
#
#     train.py:25,79      models/best_checkpoint.pt
#     inference.py:17-19  design/design.txt
#
# So two jobs sharing a working directory silently destroy each other's work.
# Training is the obvious case (interleaved torch.save() into one file), but
# inference is worse: infer_progen.job rewrites DECODING_STRATEGY with sed and
# then copies design.txt to a strategy-named file. Two overlapping inference
# jobs produce design_greedy.txt and design_topp.txt that may both hold the
# SAME decoder's output, with no error anywhere. A mislabelled result is far
# more damaging than a crash, because you write it up as fact.
#
# Rather than edit the graded assignment files, give each job its own working
# directory and let the hardcoded relative paths resolve inside it. The .py
# files are copied (so `sed -i` mutates only this job's copy), while the large
# read-only inputs are symlinked (so we do not duplicate 600 MB of weights and
# 48k sequences per run).
#
#     runs/train_47264004/            <- one Slurm job
#       train.py, inference.py, ...   copies, safe to sed
#       data -> ../../assignment/data                 symlink, shared
#       pretrained_model -> ../../assignment/...      symlink, shared
#       models/best_checkpoint.pt     this run's output, nothing else writes it
#       design/                       this run's designs
#       run.env                       provenance: job id, host, GPU, settings
#     runs/latest-train -> train_47264004
#
# Consequence worth having: concurrent runs are now safe, so you can queue
# several training variants or both decoders at once. That is the thing PSC is
# actually better at than a laptop.

# The five files deploy.sh sends. Keep in step with its rsync list.
RUN_DIR_PY_FILES="dataset.py inference.py model.py progen_vocab.py train.py"

# make_run_dir <project-root> <kind> [tag]
# Prints the created directory. Caller cd's into it.
make_run_dir() {
  local root="$1" kind="$2" tag="${3:-}"

  # Outside Slurm (a login-node smoke test) fall back to the shell PID so the
  # directory is still unique and still never collides with a real job.
  local jid="${SLURM_JOB_ID:-local$$}"
  local run="${root}/runs/${kind}_${jid}${tag:+_${tag}}"

  mkdir -p "${run}/models" "${run}/design"

  local f
  for f in ${RUN_DIR_PY_FILES}; do
    if [ ! -f "${root}/assignment/${f}" ]; then
      echo "ERROR: ${root}/assignment/${f} missing - did deploy.sh run?" >&2
      return 1
    fi
    cp "${root}/assignment/${f}" "${run}/${f}"
  done

  relink "${root}/assignment/data" "${run}/data"
  relink "${root}/assignment/pretrained_model" "${run}/pretrained_model"

  printf '%s\n' "${run}"
}

# relink <target> <link>
# `ln -sfn` is NOT reliable here. When <link> already exists as a symlink to a
# directory, some coreutils versions descend into it and create the new link
# *inside* the target directory instead of replacing it. Delete, then create.
relink() {
  local target="$1" link="$2"
  # -f on a symlink removes the link, never what it points at. If someone has
  # replaced it with a real directory, this fails loudly, which is correct.
  rm -f "${link}"
  ln -s "${target}" "${link}"
}

# write_run_env <run-dir> [extra KEY=VALUE ...]
# A flat provenance file. Months later this is the only thing that tells you
# which checkpoint came from which settings.
write_run_env() {
  local run="$1"; shift
  {
    echo "job_id=${SLURM_JOB_ID:-none}"
    echo "job_name=${SLURM_JOB_NAME:-none}"
    echo "partition=${SLURM_JOB_PARTITION:-none}"
    echo "submitted_from=${SLURM_SUBMIT_DIR:-$PWD}"
    echo "host=$(hostname)"
    echo "started=$(date -Is)"
    echo "gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    local kv
    for kv in "$@"; do echo "${kv}"; done
  } > "${run}/run.env"
}

# point_latest <project-root> <kind> <run-dir>
# Updates runs/latest-<kind>. Relative target so the tree survives being moved
# or rsync'd home.
point_latest() {
  local root="$1" kind="$2" run="$3"
  relink "$(basename "${run}")" "${root}/runs/latest-${kind}"
}

# resolve_checkpoint <project-root>
# Where inference should read the model from. Precedence:
#   CKPT=/abs/or/relative/path   explicit file, wins outright
#   TRAIN_RUN=train_47264004     a directory name under runs/
#   (default)                    runs/latest-train, set by the last good train
resolve_checkpoint() {
  local root="$1" ckpt=""

  if [ -n "${CKPT:-}" ]; then
    ckpt="${CKPT}"
  elif [ -n "${TRAIN_RUN:-}" ]; then
    ckpt="${root}/runs/${TRAIN_RUN}/models/best_checkpoint.pt"
  else
    ckpt="${root}/runs/latest-train/models/best_checkpoint.pt"
  fi

  if [ ! -f "${ckpt}" ]; then
    echo "ERROR: no checkpoint at ${ckpt}" >&2
    echo >&2
    echo "Run training first, or name a run explicitly:" >&2
    echo "  sbatch --export=ALL,TRAIN_RUN=train_<jobid> psc/infer_progen.job" >&2
    echo "  sbatch --export=ALL,CKPT=/path/to/best_checkpoint.pt psc/infer_progen.job" >&2
    echo >&2
    echo "Available training runs:" >&2
    ls -1d "${root}"/runs/train_* 2>/dev/null >&2 || echo "  (none)" >&2
    return 1
  fi

  # Absolute, because the caller cd's into the run directory afterwards.
  readlink -f "${ckpt}"
}
