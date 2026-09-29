# Index

**What this repo does:** CMU 11781/02741 Assignment 1. We finetune **ProGen2-small** on green fluorescent protein (GFP) sequences, generate new GFP designs from 64-residue prompts, fold them with **AlphaFold 3**, keep the top 5 by pLDDT (with pTM > 0.8), and compare them to the training-set consensus.

| Document | Purpose |
| --- | --- |
| [submission.md](submission.md) | **Start here to reproduce or resubmit** — the end-to-end runbook, clean checkout to zip |
| [setup.md](setup.md) | Environment setup: ProGen2 venv, AlphaFold 3 via Docker, the three folding routes |
| [psc/README.md](psc/README.md) | Running any part of this on PSC Bridges-2 |
| [report.html](report.html) | The submitted report; open and print to PDF |
| [README.md](README.md) | Repo origin and how the two upstream codebases relate |

Assignment spec: `Assignment_One_Guidelines.pdf` (gitignored) or [assignment/docs/index.md](assignment/docs/index.md).

## Why this assignment is shaped this way

The pipeline here is a miniature of the standard **generate-then-filter** loop used throughout the course's research papers (`lectures/GenAI for Biomedicine/`, numbered roughly in teaching order):

- A **generative model proposes** many candidate sequences. It is cheap and high-throughput, but has no direct notion of whether a sequence folds or functions.
- A **structure predictor scores** them, and confidence metrics act as a cheap in-silico proxy for "will this actually work," so only the best candidates go on to expensive wet-lab validation.

That filter step is the point. *De novo design of luciferases* (13) screened on AlphaFold2 pLDDT > 92 before expressing just 46 designs; *Simulating 500 million years of evolution* (7, ESM3) filtered generations at pLDDT > 0.8. The assignment's "top-5 by pLDDT, pTM > 0.8" rule is the same idea at course scale. This is also why the spec says a failure to clear pTM 0.8 means *the finetuning* went wrong — the filter is diagnostic of the generator.

Where each piece sits in the course arc:

| Concept | Lecture | Role here |
| --- | --- | --- |
| Backprop, SGD, Adam, LR schedules | DL `lec3`–`lec8` | The finetuning loop in `train.py` (Adam + inverse-sqrt decay) |
| Transformers, autoregressive LMs | GenAI `Transformer (2)` | ProGen2's architecture; next-token prediction over amino acids |
| Protein language models | `ProGen2` (7), `ESM` (7) | Sequence-only generative models; finetuning specialises a universal model to one family |
| Structure prediction | `AlphaFold` (11), `AlphaFold 3` (14) | The scoring/filter stage; Evoformer trunk + diffusion head |
| Inverse folding / design | `ProteinMPNN` (9), `RFdiffusion` (12) | The complementary structure-first route this assignment does *not* take |

**Sequence-first vs. structure-first.** ProGen2 is a pure sequence model: it learns the distribution of evolutionarily observed proteins and never sees 3D coordinates. Structure only re-enters at evaluation, through AlphaFold 3. That contrast — ProGen2/ESM generating sequences versus RFdiffusion/ProteinMPNN building backbones then designing sequences onto them — is the main axis the protein-design lectures organise around.

**What finetuning is doing.** ProGen2-small is pretrained on ~a billion protein sequences across all families. Finetuning on 47,854 GFP variants shifts the distribution onto one narrow family, which the ProGen2 paper shows raises structural similarity to natural proteins and improves generation quality within the family. The 64-residue prompt then pins each design to the GFP scaffold's start, so the model completes rather than invents.

**Why GFP.** All 47,854 training sequences are exactly 237 residues — a fixed-length, aligned mutational scan of avGFP, not a diverse homolog family. So position *i* means the same thing in every sequence, which is what makes the consensus-and-substitution divergence analysis well defined. Fluorescence also depends on a chromophore formed by a specific residue triad in a rigid β-barrel, so "does it fold" is a genuinely meaningful proxy for "does it work."

## Measured facts about the dataset

Computed from `assignment/data/train.txt` (see `af3_eval/divergence_analysis.py`):

| Fact | Value | Why it matters |
| --- | --- | --- |
| Sequences × length | 47,854 × exactly 237 | Fixed-length and aligned, so position *i* is comparable across all sequences |
| Consensus identity | avGFP with the initial Met removed | Consensus position *i* = avGFP position *i*+1 |
| Chromophore | **SYG at positions 64–66** | The Ser-Tyr-Gly triad that autocatalytically forms the fluorophore. Substitutions here are functionally fatal |
| Prompt coverage | positions 1–64 | The 64-token prompt ends at S64, so **the model must generate the chromophore Tyr65 itself** |
| Conservation | mean 0.984; minimum **0.959** at any position | No position varies by more than ~4%. This is a point-mutation scan of one protein, not a diverse homolog family |
| Fully invariant | positions 1, 87, 93, 118, 119, 237 | Never vary in training |

**The practical consequence:** because the data is so low-entropy, finetuning loss collapses very fast (5.42 → ~0.17 within a few hundred steps), and a **greedy**-decoded model will tend to reproduce the consensus continuation for nearly every prompt. If all 92 designs come out near-identical, ranking them by pLDDT is close to meaningless. This makes deliverable 3 ("which decoding strategy, and its parameters") substantive rather than bookkeeping: `inference.py` ships with `DECODING_STRATEGY = "greedy"`, and `forward_inference` also implements top-p with a hard-coded threshold of 0.8 ([model.py:760](assignment/model.py#L760)). Generating with both and comparing diversity is the informative experiment.

**A caveat worth stating in the report:** pLDDT and pTM measure *predicted structural confidence*, not fluorescence. A design can fold into a convincing β-barrel and still be dark if the chromophore environment is disrupted. AlphaFold 3 is also largely insensitive to the point mutations that determine GFP brightness. These metrics filter out designs that clearly fail to fold; they do not confirm function.

## Pipeline → where the code lives

| Step | What happens | Code | Output |
| --- | --- | --- | --- |
| 1. Data | GFP sequences, one per line (train ~47.9k, valid 1k, test 92 prompts) | [assignment/data/](assignment/data/), loaded by [assignment/dataset.py](assignment/dataset.py) | — |
| 2. Tokenize | Amino acids ↔ ids; `1`/`2` are BOS/EOS | [assignment/progen_vocab.py](assignment/progen_vocab.py) | — |
| 3. Model | ProGen2 causal transformer; greedy / top-p decoding in `forward_inference` | [assignment/model.py](assignment/model.py) | — |
| 4. Finetune | Adam training loop, fixed seed, saves the best-validation checkpoint | [assignment/train.py](assignment/train.py) | `assignment/models/best_checkpoint.pt` |
| 5. Design | Prompt with 64 residues, generate full proteins | [assignment/inference.py](assignment/inference.py) | `assignment/design/design.txt` |
| 6. AF3 inputs | design.txt → one AF3 JSON per design (MSA-free by default) | [af3_eval/make_af3_inputs.py](af3_eval/make_af3_inputs.py) | `~/af3/af_input/` |
| 7. Fold | AF3 inference in Docker, or AlphaFold Server | [alphafold3/run_alphafold.py](alphafold3/run_alphafold.py) | `~/af3/af_output/` |
| 8. Rank | Mean pLDDT, pTM > 0.8 filter, top 5 | [af3_eval/rank_af3_outputs.py](af3_eval/rank_af3_outputs.py) | `af3_ranking.csv` |
| 9. Divergence | Consensus of train.txt vs. top 5: substitutions, positions, decoding, interpretation | [af3_eval/divergence_analysis.py](af3_eval/divergence_analysis.py) | `report_data/substitutions.csv`, `consensus.txt` |
| 10. Report | Top-5 write-up: images, metrics, divergence, interpretation | [report.html](report.html) → print to PDF | submission PDF |

The output paths above are the **local** ones, and they are fixed names — re-running step 4 or 5 overwrites the previous result. On PSC each Slurm job works inside its own `runs/<kind>_<jobid>/` instead, so runs accumulate rather than replace one another and several can run at once ([psc/README.md](psc/README.md#every-run-is-isolated--nothing-overwrites-anything)).

**TODO status — all five are now filled in** (the guidelines' seed and noted hyperparameters are untouched; only `BATCH_SIZE`, which the file marks as changeable, was lowered to 8 for an 8 GB GPU):

| File | What was needed | Implementation |
| --- | --- | --- |
| [model.py:710](assignment/model.py#L710) | Map hidden states to a token distribution | `return F.softmax(self.lm_head(hidden_states), dim=-1)` — returns *probabilities*, because `train.py`'s loss takes `-log` of gathered values itself. Mirrors line 754 in `forward_inference` |
| [train.py:30](assignment/train.py#L30) | Adam optimizer | `torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)` |
| [train.py:47](assignment/train.py#L47) | Training forward pass | `outputs = model(inputs)` |
| [train.py:71](assignment/train.py#L71) | Validation forward pass | same, wrapped in `torch.no_grad()` to save memory |
| [inference.py:30](assignment/inference.py#L30) | Load best checkpoint | `model.load_state_dict(torch.load(...))` + `model.eval()` |
| [inference.py:38](assignment/inference.py#L38) | Generate | `model.forward_inference(prefix_seqs, eos_idx, MAX_SEQ_LENGTH, decoding_strategy=DECODING_STRATEGY)` under `no_grad` |

Since this is graded work, verify you can explain each line — particularly *why* `forward` returns softmax probabilities rather than logits, which is unusual and is dictated by the loss expression in `train.py:51`.

## Folders

| Path | Origin | Notes |
| --- | --- | --- |
| [assignment/](assignment/) | [GenAIBioMed/GenAIBioMedAssignment1](https://github.com/GenAIBioMed/GenAIBioMedAssignment1) snapshot | PyTorch code plus the MkDocs course page (`docs/`, `mkdocs.yml`) |
| [alphafold3/](alphafold3/) | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) snapshot | Inference only; JAX; Linux/Docker. Leave unmodified |
| [af3_eval/](af3_eval/) | Ours | Stdlib-only glue between ProGen2 output and AF3 |
| [psc/](psc/README.md) | Ours | Deploy scripts and Slurm jobs for running any part of this on PSC Bridges-2. `_run_dir.sh` isolates each job's outputs under `runs/<kind>_<jobid>/` |
| [lectures/](lectures/) | Course material | `Deep Learning/` (lec1–10: nets → backprop → optimizers → CNNs) and `GenAI for Biomedicine/` (numbered lectures, then research papers). Note `Deep Learning/Lecture/` duplicates its parent byte-for-byte |
| `progen/` | Local venv (gitignored) | Python 3.10, torch 2.7.0 cu128, transformers 4.49.0 |

## Inside `alphafold3/` (read-only reference)

| Need | Look at |
| --- | --- |
| CLI flags, overall run loop | [run_alphafold.py](alphafold3/run_alphafold.py) |
| Input JSON format (MSA-free fields, seeds) | [docs/input.md](alphafold3/docs/input.md) |
| Output files; pLDDT / pTM definitions | [docs/output.md](alphafold3/docs/output.md) |
| Small-GPU memory settings, compile cache | [docs/performance.md](alphafold3/docs/performance.md) |
| Install, Docker, Singularity | [docs/installation.md](alphafold3/docs/installation.md), [docker/Dockerfile](alphafold3/docker/Dockerfile) |
| MSA / template search (the CPU data pipeline) | [src/alphafold3/data/](alphafold3/src/alphafold3/data/) (`pipeline.py`, `tools/jackhmmer.py`) |
| Network: Evoformer trunk, diffusion head | [src/alphafold3/model/network/](alphafold3/src/alphafold3/model/network/) (`evoformer.py`, `diffusion_head.py`) |
| Confidence heads (pLDDT, PAE, pTM) | [model/network/confidence_head.py](alphafold3/src/alphafold3/model/network/confidence_head.py), [model/confidences.py](alphafold3/src/alphafold3/model/confidences.py) |
| Weight loading (`af3.bin.zst`) | [src/alphafold3/model/params.py](alphafold3/src/alphafold3/model/params.py) |
| mmCIF / structure handling | [src/alphafold3/structure/](alphafold3/src/alphafold3/structure/) |
| Terms for weights and outputs | [WEIGHTS_TERMS_OF_USE.md](alphafold3/WEIGHTS_TERMS_OF_USE.md), [OUTPUT_TERMS_OF_USE.md](alphafold3/OUTPUT_TERMS_OF_USE.md) |

## Results (this run)

| | |
| --- | --- |
| Finetuning | 5 of 10 epochs, `BATCH_SIZE = 8`, validation loss 0.1438 → **0.0915** |
| Designs | 92 per decoder, all exactly 237 residues, all unique |
| Folded | 20 on the AlphaFold Server — **20/20 with pTM > 0.8** |
| Top 5 | design_018, 007, 008, 010, 006 — pLDDT 92.30 → 91.81, pTM 0.91–0.92 |
| **Greedy contribution** | **0** substitutions beyond position 64, across all 92 designs |
| **Top-p (p=0.8) contribution** | **3** substitutions beyond position 64, in 3 of 92 designs |

The central finding: **188 of 191 substitutions were inherited from the 64-residue prompts, not authored by the model.** The designs look diverse only because their prompts differ. See the interpretation in [report.html](report.html).

## Things worth knowing before re-running

Hard-won during this run; each cost real time.

| Gotcha | Detail |
| --- | --- |
| **Prompt vs. model attribution** | The prompt is positions 1–64. Substitutions there were *given* to the model. Always split at 64 before claiming the model contributed something |
| **`python -u`** | Without it, stdout buffers in 8 KB blocks and a 6-hour run shows no progress for ~30 min at a time |
| **VRAM oversubscription** | High `utilization.gpu` + low `utilization.memory` + near-zero `memory.free` means Windows WDDM is paging VRAM to system RAM. Throughput drops ~5× with no throttle flags set. Check `memory.free` before blaming thermals |
| **Laptop power profiles** | A vendor "Silent" scheme caps CPU/GPU even on AC. `powercfg /list` then `/setactive`. Separate from, and additive to, the VRAM issue |
| **Batch size changes the LR schedule** | The schedule is `num_updates^-0.5`, and batch size sets updates/epoch (5,982 at batch 8 vs 1,496 at 32). Not a free knob — disclose it |
| **torch cu128 has no Volta kernels on Linux** | V100s fail with `no kernel image is available for execution on the device`. The *Windows* build does include `sm_70`, so this cannot be reproduced locally. Request `l40s-48`/`h100-80`, or install a cu126 build |
| **CUDA minor-version compatibility** | `sm_89` (RTX 4060) is absent from the compiled arch list yet works, because an `sm_86` cubin runs on any `sm_8x` device. No such fallback across major versions |
| **Slurm `REASON` is a catch-all** | The same DOWN/DRAINED string appears whether a node is dead or merely reserved. Read `SCHEDNODES` and `START_TIME` instead; `Unknown` means blocked, `N/A` means not yet scheduled |
| **GPU-small is one node** | Its short queue (~2.9 h vs ~28.5 h) comes from being a single dedicated node — which also means one drained machine blocks the whole partition |
| **Count the GPUs before picking a type** | `sinfo -p GPU-shared -o "%20N %10T %G"`. GPU-shared held **three** l40s nodes, two of them drained, against **ten** h100 nodes mostly free. Requesting the scarcer GPU cost ~24 h of queue for no gain. Partition choice is not the whole story — inventory within the partition is |
| **Slurm's `StartTime` is a worst case** | It assumes every job ahead runs to its full `TimeLimit`, which most do not. Treat it as a soft ceiling, not a countdown. Having one at all beats `N/A`: it means you hold a reservation |
| **`sprio` tells you whether waiting will help** | Ours was 94% fairshare and 0.03% age, so queueing longer earned almost nothing — and resubmitting with better flags forfeited almost nothing. Check before deciding to sit tight |
| **Hardcoded output paths** | `train.py` and `inference.py` write `models/best_checkpoint.pt` and `design/design.txt` with no run identifier, so a second run silently replaces the first. Worse for inference, where the job script's `sed` + copy can mislabel one decoder's output as the other's. `psc/_run_dir.sh` gives every Slurm job its own tree |
| **`ln -sfn` does not reliably replace a symlink** | When the link already points at a directory, some coreutils create the new link *inside* it instead. `rm -f` then `ln -s`. Also: Git Bash fakes symlinks as real directories, so symlink logic cannot be tested there — use WSL |
| **`.job` files need an LF rule** | `.gitattributes` covered `*.sh` but not `*.job`. `deploy.sh` rsyncs the working copy verbatim, so a CRLF checkout reaches Bridges-2 and `sbatch` fails on the carriage return |
| **PowerShell ≠ bash** | Heredocs, `$VAR`, and single-quote quoting don't transfer. Use `python -c "..."` |
| **Server results aren't files** | Viewing AlphaFold Server results in the browser downloads nothing. Click Download, then extract the zips |
| **pLDDT can't rank near-identical sequences** | The top-5 spread was 0.49 points for sequences differing by 1–3 residues — within AlphaFold's own run-to-run variation |

## Not in git (runtime artifacts)

`assignment/pretrained_model/`, `assignment/models/`, `assignment/design/`, `runs/` (per-job PSC working trees), `progen/`, `af3_work/`, `af3_results/` (~100 MB, and outputs carry [their own terms](alphafold3/OUTPUT_TERMS_OF_USE.md)), `lectures/` (637 MB), AF3 weights (`*.bin.zst`), databases, `submission_code.zip`, `*.log`. See [.gitignore](.gitignore).

The distilled numbers — `report_data/` and `af3_results/af3_ranking.csv` — **are** tracked, so the report's claims stay verifiable without the bulk.

**No credentials exist in this repo.** The `.gitignore` carries defensive patterns (`.env`, `*.pem`, `id_rsa*`, `credentials.json`, …) so a stray file can't be committed later, and the `psc/` scripts use `<psc-username>` placeholders rather than real identifiers.

## Deliverables (from the guidelines)

1. A zip of the completed Python files (no data).
2. A PDF report on the top 5 designs: AF3 visualization, sequence, pTM, pLDDT, an explanation of the metrics, and the divergence analysis (substitution table in `A36V` format, positional distribution over 237 residues, decoding strategy and its parameters, and a one-paragraph interpretation).
