# Setup: running AlphaFold 3 for Assignment 1

What you need to fold the ProGen2 GFP designs with AlphaFold 3 (AF3), rank them by pLDDT, and keep those with pTM > 0.8.
The ProGen2 side (finetuning and design) is covered briefly at the end. See [Index.md](Index.md) for where code lives.

## Pick a path first

The guidelines allow either the AF3 **web server** or the **GitHub code**. On this laptop the full local install is not possible, because the genetic databases need about 630 GB and C: has about 330 GB free. That leaves three workable options:

| Path | MSA / templates | Where it runs | Limits | Use it for |
| --- | --- | --- | --- | --- |
| **A. AlphaFold Server** | Yes (server-side) | Browser | 30 jobs/day per Google account | The numbers and pictures in the final report (what the pTM > 0.8 bar assumes) |
| **B. Local Docker, MSA-free** | No | This laptop (WSL2 + RTX 4060, 8 GB) | pLDDT/pTM run lower without an MSA | Screening all ~92 designs quickly, with no daily cap |
| **C. PSC Bridges-2** | Yes, if the databases are available there | HPC GPU node | ~2.9 h queue on GPU-small for training; ~28 h on GPU-shared, which AF3 requires | Faster training at batch 32, concurrent runs, a free laptop |

**Recommended:** use B to screen every design, then send the top 10–15 to A (about one day of server quota) and report the server's pLDDT, pTM and structure images. Scores from B and A are **not comparable** with each other: rank inside one method, and report numbers from one method.

**On PSC specifically:** for *training*, submit to the **GPU-small** partition (~2.9 h queue), not GPU-shared (~28.5 h). A v100-32 runs batch 32 against the laptop's forced batch 8, so compute is roughly 2× faster and wall clock comes out about even — the wins are a free laptop and the ability to run jobs concurrently. See [psc/README.md](psc/README.md).

---

## What this machine has (checked 2026-09-28)

| Item | Value | AF3 requirement | Status |
| --- | --- | --- | --- |
| OS | Windows 11 + WSL2 Ubuntu 24.04 | Linux | OK through WSL2 / Docker Desktop |
| GPU | RTX 4060 Laptop, 8 GB, compute capability 8.9 | NVIDIA, CC ≥ 8.0 | OK. Needs unified memory (below) |
| NVIDIA driver | 592.00 | Host CUDA ≥ 12.6 | OK. `docker run --gpus all … nvidia-smi` works |
| Docker | Docker Desktop 28.5.1, WSL2 backend | Docker with GPU support | OK |
| RAM visible to WSL | 12 GB + 16 GB swap (raised from the ~7.7 GB default; step 1) | 64 GB recommended for MSA search | OK for building the image and MSA-free inference |
| Disk (C:) | ~330 GB free | ~630 GB databases + ~1 GB weights | Enough for weights, not for databases |

GFP designs are ~237 residues, which is 237 tokens. That falls into AF3's smallest compile bucket (256), which is far below the size where memory becomes a problem.

---

## Path B: local AF3 in Docker (MSA-free)

Run every command from this point on in the **WSL Ubuntu** shell unless the step says PowerShell. Keep the AF3 working files in the Linux filesystem (`~/af3`), because I/O over `/mnt/c` is slow.

### 1. Give WSL more memory (one time, PowerShell; done on 2026-09-28)

**Required for the build as well as for folding:** the image's final `build_data` step fails with `cannot allocate memory` under WSL's default ~7.7 GB. AF3 also uses unified memory here, which can spill GPU memory into host RAM. Create `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=16GB
```

Then restart WSL with `wsl --shutdown` in PowerShell and reopen Docker Desktop.

### 2. Build the AF3 Docker image (done on 2026-09-28)

From the repository root (PowerShell or WSL):

```sh
docker build -t alphafold3 -f alphafold3/docker/Dockerfile alphafold3
```

This compiles HMMER and AF3's C++ extensions and installs JAX 0.10.2 with CUDA 12. It takes about 15–20 minutes, and the image is 14.2 GB. Check that JAX sees the GPU:

```sh
docker run --rm --gpus all alphafold3 python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
```

`.gitattributes` keeps `alphafold3/` files LF-only. A CRLF copy of `docker/jackhmmer_seq_limit.patch` would break the build.

### 3. Download the model weights (done on 2026-09-28 — you are bound by the terms)

The weights are free to download, but using them binds **you** to the [weights terms of use](alphafold3/WEIGHTS_TERMS_OF_USE.md) and the [prohibited use policy](alphafold3/WEIGHTS_PROHIBITED_USE_POLICY.md) (non-commercial use; outputs fall under [OUTPUT_TERMS_OF_USE](alphafold3/OUTPUT_TERMS_OF_USE.md)). Read those files, then:

```sh
mkdir -p ~/af3/models ~/af3/af_input ~/af3/af_output ~/af3/jax_cache
wget -P ~/af3/models https://storage.googleapis.com/alphafold3/af3.bin.zst
```

Leave the file compressed. AF3 reads `af3.bin.zst` directly. Do not put weights inside the repo: `.gitignore` excludes them, and the Docker build would copy them into the image.

Present state: `~/af3/models/af3.bin.zst`, 1,020,545,840 bytes (973 MiB), inside WSL. The input/output/cache directories listed above already exist.

### 4. Make the input JSONs from your designs

After `assignment/inference.py` has written `assignment/design/design.txt` (one sequence per line):

```sh
cd /mnt/c/code_projects/GenAIBiomedicineAssignmentOne
python3 af3_eval/make_af3_inputs.py assignment/design/design.txt ~/af3/af_input
```

This writes one `design_NNN.json` per line (seed 1, empty MSAs, no templates) and `designs_index.csv` (name → sequence).
It removes ProGen control characters (`1`/`2` = BOS/EOS) and anything that isn't one of the 20 standard amino acids, and prints a warning when it does.

### 5. Fold

```sh
docker run --rm --gpus all \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e TF_FORCE_UNIFIED_MEMORY=true \
  -e XLA_CLIENT_MEM_FRACTION=3.2 \
  -v ~/af3/af_input:/root/af_input \
  -v ~/af3/af_output:/root/af_output \
  -v ~/af3/models:/root/models \
  -v ~/af3/jax_cache:/root/jax_cache \
  alphafold3 \
  python run_alphafold.py \
    --input_dir=/root/af_input \
    --output_dir=/root/af_output \
    --model_dir=/root/models \
    --norun_data_pipeline \
    --jax_compilation_cache_dir=/root/jax_cache
```

What the flags do:

- The three `-e` variables switch on unified memory, which the [AF3 docs](alphafold3/docs/performance.md) recommend for GPUs with less than 80 GB. They override the image's A100 defaults.
- `--norun_data_pipeline` skips the MSA and template search, which is why the empty MSA fields from step 4 are required.
- `--jax_compilation_cache_dir` saves the compiled model. The first job compiles (~2 min); after that, each design takes seconds of GPU time.
- The first time, test a single file: replace `--input_dir=…` with `--json_path=/root/af_input/design_001.json`.

**GPU contention:** folding and finetuning cannot share this 8 GB card. Training peaks at ~4.5 GiB and AF3 needs roughly 3–4 GiB for a 237-token input, so run them one after the other, not at once.

### 6. Rank

```sh
python3 af3_eval/rank_af3_outputs.py ~/af3/af_output --index ~/af3/af_input/designs_index.csv
```

This writes `~/af3/af_output/af3_ranking.csv` and prints the top 5 by **mean pLDDT** that also have **pTM > 0.8**. Feed that CSV to the divergence analysis for deliverable 4:

```sh
python3 af3_eval/divergence_analysis.py assignment/data/train.txt assignment/design/design.txt \
    --ranking ~/af3/af_output/af3_ranking.csv --top 5 --out report_data/
```

pLDDT is averaged over all atoms in `atom_plddts`, which matches the server's coloring. The ranker also reads extracted **AlphaFold Server** zips (`fold_<job>_summary_confidences_0.json` + `fold_<job>_full_data_0.json`), so the same command ranks Path A results.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| Build fails at `uv run build_data` with `cannot allocate memory` | Do step 1, run `wsl --shutdown`, restart Docker Desktop, and rebuild (earlier layers are cached) |
| `could not select device driver "" with capabilities: [[gpu]]` | Docker Desktop is not running, or WSL integration is off (Settings → Resources → WSL integration) |
| Out of memory / `RESOURCE_EXHAUSTED` | Check that the unified-memory `-e` flags are set and step 1 is done. Close other GPU apps |
| Triton / flash-attention error | Add `--flash_attention_implementation=xla` (slower, but works everywhere) |
| Very slow first job | Normal: XLA compilation. Keep `jax_cache` mounted |
| No designs pass pTM > 0.8 in MSA-free mode | Expected to some degree, since single-sequence AF3 is less confident. Confirm your top designs on the server (Path A) before you decide finetuning failed |

---

## Path A: AlphaFold Server

1. Sign in at <https://alphafoldserver.com> with a Google account. Each account gets 30 jobs/day, and the guidelines suggest about 3 accounts for all designs.
2. Add a protein entity with 1 copy and paste the sequence. Name the job `design_NNN` (matching `designs_index.csv`), then choose **Continue and preview job** → **Confirm and submit job**.
3. Download each result zip, extract all of them under one folder, and run `rank_af3_outputs.py` on that folder.
4. For the report: take screenshots of the structure (colored by pLDDT) and the PAE plot from the server page. Record pTM and mean pLDDT.

## Path C: PSC Bridges-2

> Ready-to-run scripts for this path live in [psc/](psc/README.md): `deploy.sh`, `setup_env.sh`, `build_af3_sif.sh` and Slurm jobs for training, inference and folding. The section below is the reasoning; that directory is the implementation.

### Do you actually need PSC for this assignment?

**Short answer: no.** Both halves of the assignment fit on this laptop, and the measured numbers say so:

| Stage | This laptop (measured) | Bridges-2 |
| --- | --- | --- |
| ProGen2 finetune, 10 epochs | **5.8 h**, batch 8, peak 4.5 GiB | ~2.5–3 h on v100-32 at batch 32, plus **~2.9 h queue on GPU-small** |
| AF3 folding, 92 designs, MSA-free | minutes after the first compile | same, plus queue |
| AF3 folding with MSAs | not possible (needs 630 GB of databases) | possible, but you must stage 630 GB into `$PROJECT` first |

For training, the partition matters more than the GPU. **GPU-small** averages a 2.9 h wait and has the same v100-32 hardware as GPU-shared, which averages 28.5 h ([ACCESS figures](https://support.access-ci.org/documentation/resources/psc-bridges-2-gpu-bridges-2-gpu), 2026-09-28). On GPU-small the finetune is roughly wall-clock neutral against the laptop — the real gains are that your laptop stays free and that you can run several jobs concurrently. AlphaFold 3 is the exception: it needs compute capability ≥ 8.0, so it cannot run on V100s and must go through GPU-shared for an l40s-48 or h100-80.

**The MSA question is the one real reason to consider PSC** — and even there, the AlphaFold **Server** (Path A) gives you MSA-based folding with no queue and no 630 GB download. That is why the recommendation is local screening plus server confirmation.

**Use PSC when:** the laptop is needed for other work during the ~6 h finetune; you want to finetune a bigger checkpoint (progen2-medium/large/xl will not fit in 8 GB); you want MSA-based AF3 on more sequences than the server's 30/day allows; or your local GPU is unavailable. Start the queue job *early* — submit, then keep working locally while it waits.

### Access and orientation

```sh
ssh <psc-username>@bridges2.psc.edu      # PSC username, not your ACCESS ID
projects                                  # allocation balance + Ocean quota
my_quotas                                 # storage usage
module spider alphafold                   # check whether PSC already provides AF3
```

File spaces — this matters, because AF3 is large:

| Space | Path | Size | Notes |
| --- | --- | --- | --- |
| `$HOME` | `/jet/home/<user>` | 25 GB, backed up | Too small for weights + databases. Code only |
| `$PROJECT` | `/ocean/projects/<group>/<user>` | allocation-dependent, **not** backed up | Put the weights, databases, `.sif` and outputs here |
| `$LOCAL` | node-local scratch | fast, **deleted when the job ends** | Copy databases here inside the job for fast MSA search |

Transfer files through the data transfer node, not the login node:

```sh
rsync -rltpDvP -e ssh assignment/ <user>@data.bridges2.psc.edu:/ocean/projects/<group>/<user>/assignment/
```

### C1. ProGen2 finetuning on Bridges-2

Use the scripts in [psc/](psc/README.md) rather than hand-rolling a job file — they encode three
things that are easy to get wrong and expensive to discover from a failed queue slot.

```sh
bash psc/deploy.sh <psc-username>   # from the repo root, in WSL
ssh <psc-username>@bridges2.psc.edu
cd $PROJECT/gfp
bash psc/setup_env.sh               # conda env on Ocean + ProGen2-small, ~10 min
sbatch psc/train_progen.job
squeue -u $USER                     # ST: PD = pending, R = running
```

**Do not request a v100 for ProGen2.** The assignment pins torch 2.7.0+cu128, whose Linux build
ships no Volta (`sm_70`) kernels, so every v100 node dies inside the first forward pass with
`no kernel image is available for execution on the device`. That is why the job asks for
`l40s-48` — and why it must live on GPU-shared, since GPU-small has only v100s. The one-line
escape, if you would rather have GPU-small's short queue, is to reinstall torch from
`--index-url https://download.pytorch.org/whl/cu126`, which still includes `sm_70`; that
deviates from the pinned version, so disclose it if you use it.

**The environment goes on Ocean, not in `$HOME`.** It is ~6 GB and conda's package cache adds
several more, against a 25 GB `$HOME` quota. `setup_env.sh` builds it at `$PROJECT/gfp/envs/progen`
and the jobs activate it by path.

**Every job writes to its own `runs/<kind>_<jobid>/`.** `train.py` and `inference.py` hardcode
`models/best_checkpoint.pt` and `design/design.txt`, so jobs sharing a directory overwrite each
other's results — silently, in the inference case. Each job instead gets a private working tree
with the `.py` files copied and the data and weights symlinked. Concurrent runs are therefore
safe, which is the main reason to be on a cluster at all: queue several batch sizes, or both
decoders, at once. See [psc/README.md](psc/README.md#every-run-is-isolated--nothing-overwrites-anything).

Ask for a walltime you will actually use: shorter requests schedule sooner, because they fit
backfill gaps. Training saves on every validation improvement, so a timeout costs you epochs,
not the whole run.

### C2. AlphaFold 3 on Bridges-2

**Pick the right GPU.** AF3 requires **compute capability ≥ 8.0**. That rules out `v100-16` and `v100-32` (CC 7.0) — on CC 7.x AF3 produces badly clashing structures unless you set `XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`, and DeepMind does not validate numerical accuracy there. **Request `l40s-48` (CC 8.9) or `h100-80` (CC 9.0).**

**Build the container locally, not on PSC.** Bridges-2 gives you no root, so build the Docker image on your own machine (step 2 above), convert it, and copy the `.sif` across:

```sh
# On this laptop, after `docker build -t alphafold3 ...`:
docker run -d -p 5000:5000 --restart=always --name registry registry:2
docker tag alphafold3 localhost:5000/alphafold3
docker push localhost:5000/alphafold3
SINGULARITY_NOHTTPS=1 singularity build alphafold3.sif docker://localhost:5000/alphafold3:latest
rsync -P alphafold3.sif <user>@data.bridges2.psc.edu:/ocean/projects/<group>/<user>/af3/
```

Check `/ocean/containers/` and `module spider alphafold` first — if PSC already ships an AF3 image, skip the build entirely.

**Databases (only if you want MSAs).** 252 GB compressed, ~630 GB unpacked. Confirm your Ocean quota with `my_quotas`, and ask `help@psc.edu` whether they are already staged on shared storage before downloading. Run the fetch inside a `screen`/`tmux` on the data transfer node:

```sh
cd $PROJECT/af3 && bash fetch_databases.sh $PROJECT/af3/public_databases
```

`fold_af3.job` — MSA-based, the configuration worth using PSC for:

```bash
#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 12:00:00
#SBATCH -J af3_fold
#SBATCH -o af3_%j.out

set -euo pipefail
AF3=$PROJECT/af3

# Databases are much faster from node-local scratch than from Ocean.
rsync -a $AF3/public_databases/ $LOCAL/public_databases/

singularity exec --nv \
  --bind $AF3/af_input:/root/af_input \
  --bind $AF3/af_output:/root/af_output \
  --bind $AF3/models:/root/models \
  --bind $LOCAL/public_databases:/root/public_databases \
  $AF3/alphafold3.sif \
  python run_alphafold.py \
    --input_dir=/root/af_input \
    --output_dir=/root/af_output \
    --model_dir=/root/models \
    --db_dir=/root/public_databases \
    --jax_compilation_cache_dir=/root/af_output/jax_cache
```

Generate the inputs with MSA fields omitted so the data pipeline builds them:

```sh
python3 af3_eval/make_af3_inputs.py \
    runs/latest-infer-greedy/design/design_greedy.txt $PROJECT/af3/af_input --with-msa
```

(`runs/latest-infer-<strategy>` is a symlink the inference job repoints on success. Use an
explicit `runs/infer_<jobid>_<strategy>/` path if you are folding an older run.)

Then rank exactly as locally: `python3 af3_eval/rank_af3_outputs.py $PROJECT/af3/af_output --index .../designs_index.csv`.

MSA search is CPU- and RAM-hungry; if it is slow or gets killed, request more cores and memory, and keep the databases on `$LOCAL`.

---

## ProGen2 environment (finetuning and design)

A venv built from the guidelines exists at `progen/` in the repo root (gitignored): Python 3.10, torch 2.7.0 + cu128, transformers 4.49.0. ProGen2-small (151M parameters, 606 MB) is already downloaded to `assignment/pretrained_model/progen2-small/`, and `models/` and `design/` exist (`inference.py` does not create `design/` itself).

```powershell
.\progen\Scripts\Activate.ps1
cd assignment
python -u train.py        # -u so you can watch progress; ~5.8 h
python -u inference.py    # ~7.5 min for all 92 prompts
```

### Measured on this GPU (RTX 4060, 8 GB)

| Batch size | sec/step | Peak VRAM | 10 epochs |
| --- | --- | --- | --- |
| 32 (default) | — | OOM | — |
| 16 | 1.99 | 7.35 GiB | 16.5 h (thrashing near the 8 GB limit) |
| **8 (set)** | **0.35** | **4.5 GiB** | **5.8 h** |
| 4 | 0.20 | 3.1 GiB | 6.6 h |

`BATCH_SIZE` is the one value the assignment marks as changeable ("you might modify based on the computing resources you used"); `LEARNING_RATE`, `LR_SCHEDULER_UPDATES`, `EPOCHS` and the seed are left untouched. Note that batch size changes the number of optimizer updates, and the learning-rate schedule is a function of `num_updates` — so batch 8 also gives a different LR trajectory than batch 32 would.

`train.py` writes `models/best_checkpoint.pt` after any epoch whose validation loss improves, so a usable checkpoint exists after roughly the first 35 minutes and the run is safe to interrupt.
