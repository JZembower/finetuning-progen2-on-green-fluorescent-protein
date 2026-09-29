# Running this assignment on PSC Bridges-2

Scripts and a walkthrough for doing the ProGen2 finetuning and/or the AlphaFold 3 folding on Bridges-2 instead of locally.

## Does PSC actually speed up training?

**The compute: yes, roughly 2×. The wall-clock time to a finished model: roughly a wash — but it frees your laptop, and it wins decisively as soon as you need more than one run.**

The laptop is memory-bound, not compute-bound. 8 GB forces `BATCH_SIZE = 8`; a v100-32 has 32 GB and runs the assignment's original 32, with far more memory bandwidth (900 GB/s vs ~272 GB/s).

| | Laptop (RTX 4060, 8 GB) | Bridges-2 GPU-small (v100-32) |
| --- | --- | --- |
| Batch size | 8 (16 thrashes, 32 OOMs) | 32 |
| Updates for 10 epochs | 59,820 | 14,960 |
| Compute time | **5.8 h (measured)** | **~2.5–3 h (estimated)** |
| Queue wait | none | **~2.9 h average** (range 0–6 h) |
| Wall clock to result | ~5.8 h | ~5.5–9 h |
| Your laptop during it | unusable for GPU work | free |

So for **one** run, PSC is about break-even on wall clock. What actually makes it worth it:

- **Your laptop is free** for the ~6 h, which is the main practical win.
- **Parallelism.** You can queue several jobs at once — greedy and top-p inference together, or multiple training variants — and they run concurrently. Locally they serialise. This is where PSC genuinely beats the laptop.
- **Headroom.** progen2-medium/large/xl do not fit in 8 GB at all.

**Pick the partition carefully — it matters more than the GPU.** Same v100-32 hardware, very different queues (2026-09-28):

| Partition | Average wait | Notes |
| --- | --- | --- |
| **GPU-small** | **2.9 h** (0–6 h) | Single v100-32 node, batch or interactive, 48 h max. **Start here** — but because it is one node, a drained machine blocks it entirely. Check `sinfo -p GPU-small` if it will not start |
| GPU-shared | 28.5 h (0.1–96 h) | Per-GPU billing; the only route to h100-80 / l40s-48 |
| GPU | 29.3 h | Whole 8-GPU nodes; you do not need this |

Going to GPU-shared for an h100-80 saves maybe 2 h of compute and costs ~26 h of queue. Not worth it for this job. (AlphaFold 3 is the exception — it *requires* compute capability ≥ 8.0, so it cannot use V100s at all and must go through GPU-shared. That is a constraint on folding, not on training.)

The estimate above is extrapolated from measured laptop throughput; I have not benchmarked a V100 directly. Expect the first job to tell you the real number.

## 0. Get access (once)

1. Set your PSC password at <https://apr.psc.edu/> using your ACCESS username and email. **This is a separate password from your ACCESS one**, and it does not exist until you create it — trying to log in before this step gives `Permission denied`. Your **PSC username also differs from your ACCESS ID**.

   Verify it works before anything else:

   ```sh
   ssh <psc-username>@bridges2.psc.edu
   ```

2. Optional but recommended: generate a key and register the public half with the PSC SSH Key Manager.

   ```sh
   ssh-keygen -t ed25519 -C "psc"        # you have no key on this machine yet
   cat ~/.ssh/id_ed25519.pub             # paste into the PSC key form
   ```

3. Log in and get your bearings:

   ```sh
   ssh <psc-username>@bridges2.psc.edu
   projects          # allocation balance, group name, Ocean quota
   my_quotas         # storage usage - check this before any 630 GB plan
   echo $PROJECT     # /ocean/projects/<group>/<username>
   ```

   The deploy script needs the **group directory name** — the `<group>` in `/ocean/projects/<group>/<username>`, one lowercase path component such as `cis261594p`. That is *not* the allocation title that `projects` prints (e.g. "CIS261594: 11781 Generative AI for Biomedicine"). The surest way to read it off:

   ```sh
   echo $PROJECT
   ls -d /ocean/projects/*/$USER
   ```

**Run nothing heavy on the login node.** All compute goes through `sbatch`.

## 1. Deploy the project

From the repository root **on your laptop**, in WSL (Git Bash has no `rsync`):

```sh
bash psc/deploy.sh <psc-username>
```

Omit the group and the script resolves it from the cluster, then creates the destination before transferring. Guessing by hand is the main way this step goes wrong — the directory name usually carries a suffix letter the allocation ID does not show (`cis261594p`, not `cis261594`).

**If you belong to more than one allocation**, the script refuses to choose, because `$PROJECT` points at just one of them and the wrong group spends another project's SUs. Match titles to groups, then pass the right one:

```sh
ssh <psc-username>@bridges2.psc.edu 'projects'
bash psc/deploy.sh <psc-username> <group>
```

Sends the assignment Python, the GFP data, `af3_eval/` and these job scripts to `$PROJECT/gfp`. It deliberately skips the venv, the pretrained checkpoint (re-downloaded on PSC, which is faster than uploading 606 MB) and `lectures/`.

## 2. Finetune

```sh
ssh <psc-username>@bridges2.psc.edu
cd $PROJECT/gfp
bash psc/setup_env.sh          # conda env + ProGen2-small, ~10 min
sbatch psc/train_progen.job    # always submit from $PROJECT/gfp
squeue -u $USER                # ST: PD = pending, R = running
tail -f logs/progen_*.out
```

Submit from `$PROJECT/gfp`, not from inside `psc/` or `assignment/`. The jobs use `$SLURM_SUBMIT_DIR` to find `assignment/`, and the `logs/` directory (created by `setup_env.sh`) must exist before `sbatch` runs or Slurm cannot open the output file.

The job sets `BATCH_SIZE = 32`, since a 32 GB GPU fits the assignment's original (your laptop copy stays at 8). Override per job with `sbatch --export=ALL,BATCH_SIZE=16 psc/train_progen.job`.

### Every run is isolated — nothing overwrites anything

The assignment's Python hardcodes `models/best_checkpoint.pt` and `design/design.txt` with no run identifier, so jobs sharing a directory silently destroy each other's work. Instead of editing graded files, each job builds its own working tree under `runs/` and the hardcoded relative paths resolve inside it ([psc/_run_dir.sh](_run_dir.sh)):

```text
runs/train_47264004/          one Slurm job
  train.py, inference.py …    copies, so `sed -i` touches only this run
  data -> ../../assignment/data              symlink, shared
  pretrained_model -> ../../assignment/…     symlink, shared
  models/best_checkpoint.pt   this run's model, nothing else can write it
  design/                     this run's designs
  run.env                     job id, host, GPU, batch size, checkpoint used
runs/latest-train -> train_47264004
runs/infer_47264120_greedy/
runs/latest-infer-greedy -> infer_47264120_greedy
```

Only the five `.py` files are copied; the weights and the 48k sequences are symlinked, so a run directory costs roughly the checkpoint it produces (~600 MB for training, a few KB for inference, which links the checkpoint rather than copying it).

Two consequences:

- **You can queue variants concurrently.** Several training runs with different batch sizes, or both decoders at once, no longer collide. This is the thing PSC does better than a laptop.
- **Old results survive new runs.** `runs/latest-train` moves to the newest *successful* training run; everything earlier stays on disk under its job id. Prune by hand when Ocean gets tight (`du -sh runs/*`).

Inference defaults to `runs/latest-train`. Point it elsewhere with `TRAIN_RUN=train_<jobid>` or `CKPT=/path/to/best_checkpoint.pt`.

`setup_env.sh` deliberately builds the conda environment at `$PROJECT/gfp/envs/progen` rather than in `$HOME`. The environment alone is ~6 GB (measured from the equivalent local venv) and conda's package cache adds several more — enough to exhaust the 25 GB `$HOME` quota. The job scripts activate it by that path, so keep the two in step if you move it.

Then generate designs, ideally both ways so you can answer the decoding-strategy deliverable:

```sh
sbatch psc/infer_progen.job                              # greedy
sbatch --export=ALL,STRATEGY=topp psc/infer_progen.job   # top-p 0.8
```

Submit both at once — they run concurrently in separate `runs/infer_<jobid>_<strategy>/` trees and cannot overwrite each other. Each prints its output path and how many designs are unique, and leaves `runs/latest-infer-<strategy>` pointing at itself. If greedy collapses to near-identical sequences, that is the expected consequence of a very low-entropy training set, and it is worth reporting.

Bring results home:

```sh
rsync -aP <user>@data.bridges2.psc.edu:/ocean/projects/<group>/<user>/gfp/runs/ ./psc_runs/
```

## 3. AlphaFold 3 (the part worth doing here)

### 3a. Get a container — check for one first

```sh
module spider alphafold
ls /ocean/containers/
```

If PSC already provides AlphaFold 3, use it and skip the build. Otherwise build on your laptop, inside WSL:

```sh
wsl -d Ubuntu
cd /mnt/c/code_projects/GenAIBiomedicineAssignmentOne
bash psc/build_af3_sif.sh
```

It installs Apptainer, publishes the existing `alphafold3` Docker image to a temporary local registry, and converts it. Then upload:

```sh
rsync -P ~/af3/alphafold3.sif <user>@data.bridges2.psc.edu:/ocean/projects/<group>/<user>/af3/
```

Worth trying first, since it avoids the upload entirely: `apptainer build --fakeroot` directly on Bridges-2. If PSC permits fakeroot builds, you never move a multi-GB file.

### 3b. Weights and databases

```sh
mkdir -p $PROJECT/af3/{models,af_input,af_output,jax_cache}
cd $PROJECT/af3/models
wget https://storage.googleapis.com/alphafold3/af3.bin.zst   # ~973 MB, leave compressed
```

Databases, **only if you want MSAs** — 252 GB compressed, ~630 GB unpacked. Check `my_quotas` first, and ask `help@psc.edu` whether they are already staged somewhere shared before downloading your own copy. Run it in `tmux` so a dropped connection doesn't kill it:

```sh
tmux new -s dbs
bash $PROJECT/gfp/alphafold3/fetch_databases.sh $PROJECT/af3/public_databases
# detach with Ctrl-b then d; reattach later with: tmux attach -t dbs
```

(`deploy.sh` copies `fetch_databases.sh` across; it is self-contained and needs only `wget`, `tar` and `zstd`.)

### 3c. Build inputs and fold

```sh
cd $PROJECT/gfp
python3 af3_eval/make_af3_inputs.py runs/latest-infer-greedy/design/design_greedy.txt \
    $PROJECT/af3/af_input --with-msa
sbatch psc/fold_af3.job
```

`--with-msa` omits the empty-MSA fields so AF3's data pipeline builds real alignments — the whole point of running here.

`fold_af3.job` requests **l40s-48**. Do not switch it to a V100: AlphaFold 3 needs compute capability ≥ 8.0, and V100s (CC 7.0) return badly clashing structures unless you also set `XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`, in a mode DeepMind does not validate for accuracy. The job also stages databases to `$LOCAL` node-local scratch, which is far faster than Ocean for MSA search, and falls back to MSA-free automatically if no databases are present.

### 3d. Rank and analyse

```sh
python3 af3_eval/rank_af3_outputs.py $PROJECT/af3/af_output \
    --index $PROJECT/af3/af_input/designs_index.csv

python3 af3_eval/divergence_analysis.py \
    assignment/data/train.txt runs/latest-infer-greedy/design/design_greedy.txt \
    --ranking $PROJECT/af3/af_output/af3_ranking.csv --top 5 --out report_data/
```

Both are standard-library only, so they run in any Python 3.9+ without the conda env.

## Slurm quick reference

| Command | Purpose |
| --- | --- |
| `sbatch script.job` | Submit |
| `squeue -u $USER` | Your queue; `ST` of `PD` = pending, `R` = running |
| `scancel <jobid>` | Cancel |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | What a finished job actually used |
| `interact -p GPU-shared --gres=gpu:l40s-48:1 -t 2:00:00` | Interactive node for debugging (8 h max). Not a v100 — torch cu128 cannot run the model there |
| `sinfo -p GPU-shared -o "%20N %10T %G"` | GPU inventory and node states. Check which types are actually free *before* choosing one |
| `sprio -j <jobid>` | Why your priority is what it is. A tiny `AGE` term means waiting will not help and resubmitting costs nothing |
| `projects` / `my_quotas` | Allocation balance and storage |

**Partitions:** `GPU-small` (single v100-32 node, ~2.9 h wait, 48 h max) is the right default for everything in this assignment except AlphaFold 3. `GPU-shared` bills per GPU and is the only route to `h100-80`/`l40s-48`, but averages a 28.5 h wait. `GPU` takes whole 8-GPU nodes and you do not need it.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Permission denied, please try again.` on ssh/rsync | **Your PSC password is not your ACCESS password.** Set it at <https://apr.psc.edu/> first. Then confirm you are using your PSC *username*, not your ACCESS ID. Test with a plain `ssh <user>@bridges2.psc.edu` before retrying the deploy |
| `deploy.sh` builds a path like `/ocean/projects/CIS261594:/...` | You passed the allocation *title* ("CIS261594: 11781 Generative AI...") instead of the group directory. The script now rejects this |
| `rsync: mkdir ".../gfp" failed: No such file or directory (2)` | A **parent** directory does not exist, i.e. the group name is wrong — commonly `cis261594` where the real directory is `cis261594p`. Run `bash psc/deploy.sh <user>` with no group and let it resolve the path |
| `deploy.sh` says you belong to several allocations | Expected if you are on more than one project. `$PROJECT` points at only one, and using the wrong group spends another project's SUs. Run `ssh <user>@bridges2.psc.edu 'projects'` to match allocation titles to group names, then pass the right group explicitly |
| Password prompt appears repeatedly | Register an SSH key with the PSC SSH Key Manager, or you will retype it on every transfer |
| Job sits in `PD` for many hours | You are probably on `GPU-shared`. Switch to `GPU-small` and request less walltime |
| `PD` with `(Nodes required for job are DOWN, DRAINED or reserved...)` | GPU-small is a **single node**, so if it is down or drained you wait for that one machine rather than for your turn. Check `sinfo -p GPU-small`: a `down*` state means unreachable, and `StartTime=Unknown` in `scontrol show job <id>` means Slurm cannot even estimate a start. Hedge with `sbatch -p GPU-shared psc/train_progen.job` |
| Hedged across two partitions | Now safe, and encouraged. Each job writes its own `runs/<kind>_<jobid>/`, so concurrent runs cannot corrupt a shared checkpoint. `scancel` the loser once one starts, to stop burning SUs — the surviving run's outputs are unaffected either way |
| Which run produced this checkpoint? | `cat runs/<run>/run.env` — job id, host, GPU, batch size, and (for inference) the exact checkpoint path it read |
| `runs/latest-train` missing or stale | It is only repointed when training exits 0, so a failed or timed-out job deliberately leaves it alone. Name a run explicitly: `sbatch --export=ALL,TRAIN_RUN=train_<jobid> psc/infer_progen.job` |
| `conda: command not found` in a job | The job must `module load anaconda3/2024.10-1` and run `eval "$(conda shell.bash hook)"` before `conda activate` |
| `RuntimeError: CUDA error: no kernel image is available for execution on the device` | The GPU's architecture has no kernels in this PyTorch build. **torch 2.7.0+cu128 (the version the assignment pins) omits Volta (`sm_70`) on Linux, so every v100 node fails.** Request `l40s-48` (sm_89) or `h100-80` (sm_90) instead — both job scripts now do. To use a V100 anyway, reinstall with `--index-url https://download.pytorch.org/whl/cu126`, which still ships `sm_70`. Confirm with `python -c "import torch; print(torch.cuda.get_arch_list())"` |
| CUDA OOM at batch 32 | Lower `BATCH_SIZE`, or request a GPU with more memory |
| AF3 structures full of clashes, ranking score ≈ −99 | You ran on a V100 (CC 7.0). Switch to `l40s-48` or `h100-80` |
| Disk quota exceeded | `$HOME` is only 25 GB. `setup_env.sh` already puts the environment and conda cache on Ocean; check `~/.conda`, `~/.cache/pip` and `~/.local` if you installed anything by hand |
| `conda activate` fails in a job with "environment not found" | The environment is a *path*, not a name: `conda activate $SLURM_SUBMIT_DIR/envs/progen`. Re-run `setup_env.sh` if it is missing |
| MSA search killed | Needs lots of RAM; keep databases on `$LOCAL` and request more cores |
