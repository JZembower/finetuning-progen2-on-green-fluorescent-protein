# Submission runbook

Everything needed to go from a clean checkout to a submittable zip. Written so it can be
re-run from scratch — for a resubmission, a different protein family, or a later assignment
built on the same pipeline.

The spec requires exactly two deliverables:

1. **The completed Python files** — no data, only `.py`
2. **A PDF report on the top-5 candidates** — AlphaFold 3 visualisation, sequences, pTM, pLDDT,
   an explanation of those metrics, and the divergence analysis

Background on the repo layout is in [Index.md](Index.md); environment setup is in
[setup.md](setup.md); the PSC path is in [psc/README.md](psc/README.md).

---

## 0. Environment (once)

```powershell
py -3.10 -m venv progen
.\progen\Scripts\Activate.ps1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.49.0
```

Download the base model into `assignment/pretrained_model/progen2-small/`:

```powershell
cd assignment
mkdir pretrained_model\progen2-small, models, design -Force
curl.exe -L -o pretrained_model\progen2-small\progen2-small.tar.gz https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz
tar -xzf pretrained_model\progen2-small\progen2-small.tar.gz -C pretrained_model\progen2-small
```

`inference.py` does not create `design/`, so make it yourself.

> **Windows note.** Everything below is PowerShell. Bash heredocs (`<<'EOF'`), `$VAR`, and
> single-quote quoting do not work there — use `python -c "..."` for one-liners.

---

## 1. Finetune

```powershell
cd C:\code_projects\GenAIBiomedicineAssignmentOne\assignment
..\progen\Scripts\python.exe -u train.py | Tee-Object -FilePath train.log
```

`-u` matters: without it Python buffers stdout in 8 KB blocks and a multi-hour run shows no
progress.

Writes `models/best_checkpoint.pt` after any epoch whose validation loss improves, so the run
is safe to interrupt and a usable checkpoint exists after epoch 1.

**Set `BATCH_SIZE` to fit your GPU.** Measured on an 8 GB RTX 4060:

| Batch | sec/step | Peak VRAM | 10 epochs |
| --- | --- | --- | --- |
| 32 (default) | — | OOM | — |
| 16 | 1.99 | 7.35 GiB | 16.5 h |
| **8** | **0.35** | **4.5 GiB** | **5.8 h** |
| 4 | 0.20 | 3.1 GiB | 6.6 h |

Batch size changes the optimizer update count and therefore the LR trajectory, since the
schedule is `decay_factor × num_updates^-0.5`. **Record whatever you use — it belongs in the
report.** Leave `LEARNING_RATE`, `LR_SCHEDULER_UPDATES`, `EPOCHS` and the seed alone.

**Watch for throughput collapse.** If steps slow by several times mid-run, check VRAM
headroom rather than assuming thermal throttling:

```powershell
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory --format=csv
```

High `utilization.gpu` with low `utilization.memory` and almost no `memory.free` means Windows
WDDM is paging VRAM to system RAM. Close other GPU consumers (Electron apps are the usual
culprits) or lower the batch size. A laptop "Silent" power profile is a *separate*, additive
problem — `powercfg /list` then `powercfg /setactive <performance-guid>`.

**Stopping early is defensible.** Watch for the validation loss plateau; continuing past it
sharpens the output distribution, which *reduces* generation diversity. Note where you
stopped and why.

---

## 2. Generate designs

Run **both** decoders — the comparison is required by the report and is the most informative
result you will get.

```powershell
# Greedy
# edit inference.py: DECODING_STRATEGY = "greedy"
..\progen\Scripts\python.exe -u inference.py
Copy-Item design\design.txt design\design_greedy.txt

# Top-p
# edit inference.py: DECODING_STRATEGY = "topp"
..\progen\Scripts\python.exe -u inference.py
Copy-Item design\design.txt design\design_topp.txt
```

`inference.py` always writes `design/design.txt`, so copy to a distinct name after each run or
the second overwrites the first.

Sanity check — expect 92 sequences, all length 237:

```powershell
cd ..
python -c "d=[l.strip() for l in open('assignment/design/design_topp.txt') if l.strip()]; import collections; print('count',len(d),'unique',len(set(d)),'lengths',collections.Counter(len(s) for s in d))"
```

### The check that matters most

The prompt is the first **64** residues, so anything at position ≤64 was *given* to the model.
Only positions 65–237 are the model's own work. Split them:

```powershell
python -c "cons=open('report_data/consensus.txt').read().strip(); d=[l.strip() for l in open('assignment/design/design_topp.txt') if l.strip()]; subs=[[i+1 for i,(c,x) in enumerate(zip(cons,s)) if c!=x] for s in d]; inp=sum(sum(1 for p in s if p<=64) for s in subs); gen=sum(sum(1 for p in s if p>64) for s in subs); nd=sum(1 for s in subs if any(p>64 for p in s)); print('prompt-region subs:',inp); print('model-generated subs:',gen); print('designs with >=1 generated sub:',str(nd)+'/'+str(len(d)))"
```

(`report_data/consensus.txt` is produced in step 5; run that first, or once without
`--ranking` to generate it.)

Designs can look diverse while the model contributed nothing — because the prompts differ. This
split is what distinguishes the two.

---

## 3. Build AlphaFold 3 inputs

```powershell
python af3_eval\make_af3_inputs.py assignment\design\design_topp.txt af3_work\af_input_topp
```

Writes one JSON per design plus `designs_index.csv` (name → sequence). Strips ProGen control
characters (`1`/`2`) and non-standard residues, warning when it does.

Add `--with-msa` only when folding somewhere with the genetic databases (PSC); it omits the
empty-MSA fields so the data pipeline builds real alignments.

---

## 4. Fold

**Use the web server.** You need its rendered structure images for the report, and it runs
real MSAs, so pTM > 0.8 means what the assignment intends.

1. <https://alphafoldserver.com>, sign in with Google
2. Protein entity, 1 copy, paste sequence
3. Name the job `design_NNN` to match `designs_index.csv`
4. Continue and preview → Confirm and submit

**Submit ~20, not all 92** — you need only 5 finalists, and near-identical sequences add
nothing. 30 jobs/day/account.

Submit them all first, then come back and download; they run in parallel.

```powershell
Import-Csv af3_work\af_input_topp\designs_index.csv | Select-Object -First 20 | Format-Table name,sequence -Wrap
```

Extract every downloaded zip into one folder:

```powershell
New-Item -ItemType Directory -Force af3_results | Out-Null
Get-ChildItem "$env:USERPROFILE\Downloads\fold_*.zip" | ForEach-Object { Expand-Archive $_.FullName -DestinationPath af3_results -Force }
Get-ChildItem af3_results -Recurse -Filter "*summary_confidences*.json" | Select-Object Name
```

That last line must list one file per design, or step 5 reports "No AlphaFold 3 predictions
found". Viewing results in the browser downloads nothing.

<details>
<summary>Folding locally instead (Docker, no MSAs)</summary>

Only viable for screening — scores run lower without MSAs, and you get no report images.
Requires the AF3 weights and Docker image from [setup.md](setup.md).

```sh
docker run --rm --gpus all \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TF_FORCE_UNIFIED_MEMORY=true -e XLA_CLIENT_MEM_FRACTION=3.2 \
  -v ~/af3/af_input:/root/af_input -v ~/af3/af_output:/root/af_output \
  -v ~/af3/models:/root/models -v ~/af3/jax_cache:/root/jax_cache \
  alphafold3 python run_alphafold.py --input_dir=/root/af_input --output_dir=/root/af_output \
  --model_dir=/root/models --norun_data_pipeline --jax_compilation_cache_dir=/root/jax_cache
```

Folding and finetuning cannot share an 8 GB GPU — run them sequentially.
</details>

---

## 5. Rank and analyse

```powershell
python af3_eval\rank_af3_outputs.py af3_results --index af3_work\af_input_topp\designs_index.csv
```

Top 5 by mean pLDDT with pTM > 0.8; writes `af3_results\af3_ranking.csv`. Reads both local
AF3 output and extracted AlphaFold Server zips.

```powershell
python af3_eval\divergence_analysis.py assignment\data\train.txt assignment\design\design_topp.txt --ranking af3_results\af3_ranking.csv --top 5 --out report_data\
```

Prints the `A36V` substitution table, the positional distribution, and flags substitutions at
positions the training set never varies. Writes `report_data\substitutions.csv` and
`consensus.txt`.

**Check the pLDDT spread across your top 5.** If it is under ~1 point, the ranking is within
AlphaFold's own run-to-run noise and cannot meaningfully order the candidates. Say so in the
report rather than implying #1 beat #5 on merit.

**Check for substitutions near the chromophore.** In this 237-residue numbering the `SYG`
triad is at **64–66**. A design can score pLDDT > 90 and still be non-fluorescent if that
environment is disturbed.

---

## 6. Capture images

Only for the five finalists named by step 5. For each, screenshot the full server result view:
pLDDT-coloured structure, PAE plot, and the pTM readout. Save as
`af3_results\images\design_NNN.png` — the filenames `report.html` expects.

---

## 7. Build the PDF

[report.html](report.html) is the template, already wired to `af3_results/images/*.png`.

1. Fill the `<!-- YOUR NAME -->` and `<!-- DATE -->` placeholders
2. Update every number if you re-ran anything
3. `start report.html`, then **Ctrl+P → Save as PDF**, with **Background graphics enabled**

Print CSS keeps candidate cards from splitting across pages.

---

## 8. Package

```powershell
Compress-Archive -Path assignment\*.py -DestinationPath submission_code.zip -Force
```

Expect five files: `dataset.py`, `inference.py`, `model.py`, `progen_vocab.py`, `train.py`.
**No data, no checkpoints** — the spec is explicit.

Verify before submitting:

```powershell
Expand-Archive submission_code.zip -DestinationPath $env:TEMP\subcheck -Force
Get-ChildItem $env:TEMP\subcheck
Select-String -Path $env:TEMP\subcheck\*.py -Pattern "NotImplementedError" | Select-Object Filename,LineNumber
```

The second command must return **nothing** — any remaining `NotImplementedError` means a TODO
was left unfilled.

Submit `submission_code.zip` together with the PDF.

---

## Final checklist

| Item | Where |
| --- | --- |
| All five TODOs implemented | `model.py`, `train.py`, `inference.py` |
| No `NotImplementedError` remains | verified in step 8 |
| Both decoders run and compared | `design_greedy.txt`, `design_topp.txt` |
| ≥20 designs folded, pTM > 0.8 recorded | `af3_ranking.csv` |
| Images for exactly the top 5 | `af3_results\images\` |
| Substitution table + positional distribution | `report_data\substitutions.csv` |
| Decoding strategy and parameters stated | report §5.4 |
| Metrics explained, with the function caveat | report §3 |
| Deviations disclosed (batch size, epochs) | report §2.2 |
| PDF exported with images rendering | — |
| Zip contains only `.py` | — |
