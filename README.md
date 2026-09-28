# GenAI Biomedicine workspace

Independent local copy of two separate codebases used together for CMU 11781/02741 Assignment 1:

| Folder | Source | What it is |
| --- | --- | --- |
| `assignment/` | [GenAIBioMed/GenAIBioMedAssignment1](https://github.com/GenAIBioMed/GenAIBioMedAssignment1.git) | Finetune ProGen2 on green fluorescent protein (GFP), then generate candidate sequences |
| `alphafold3/` | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3.git) | AlphaFold 3 **inference** pipeline: fold those sequences and score them (pLDDT, pTM) |

This repository is **not** a fork of either project. Git history and remotes from the originals were stripped; only a snapshot of the source is kept here.

## How the two pieces fit together

The assignment is a protein-design loop:

1. Start from Salesforce **ProGen2-small**, a causal language model over amino-acid sequences.
2. Finetune it on GFP sequences in `assignment/data/`.
3. Prompt the finetuned model with the first 64 tokens of each test sequence and autoregressively generate full proteins (`assignment/inference.py`).
4. Fold the generated sequences with **AlphaFold 3** and rank by structure-confidence metrics (pLDDT, with pTM > 0.8).
5. Compare the top-5 designs to the training-set consensus sequence (divergence analysis).

You do **not** need to merge the two Python packages. Keep them in separate folders and run each in its own environment. AlphaFold 3 is JAX/Docker/Linux + GPU; the assignment is PyTorch + CUDA.

## `assignment/` — ProGen2 GFP finetuning

Course writeup: `assignment/docs/index.md` (also published at [genaibiomed.github.io/GenAIBioMedAssignment1](https://genaibiomed.github.io/GenAIBioMedAssignment1/)).

| File | Role |
| --- | --- |
| `dataset.py` | Loads `data/{train,valid,test}.txt`, tokenizes with `progen_vocab.Alphabet`, returns next-token pairs |
| `progen_vocab.py` | Amino-acid tokenizer (BOS/EOS/PAD + residue letters) |
| `model.py` | ProGen2 transformer (Hugging Face-style). TODOs: softmax LM head in `forward`; `forward_inference` already implements greedy / top-p decoding |
| `train.py` | Finetuning loop. TODOs: Adam optimizer and calling `model(...)` |
| `inference.py` | Loads the best checkpoint and writes designs. TODOs: `load_state_dict` and `forward_inference` |
| `data/` | GFP sequence splits (`train.txt` is ~11 MB) |

Starter code is intentionally incomplete (`NotImplementedError` TODOs). Do not change the given seed or the noted hyperparameters.

**Runtime artifacts (not in git):** download ProGen2-small into `assignment/pretrained_model/progen2-small/`, write checkpoints to `assignment/models/`, write generated sequences to `assignment/design/`.

## `alphafold3/` — structure prediction inference

Official docs: `alphafold3/README.md` and `alphafold3/docs/`. Paper: [Abramson et al., Nature 2024](https://doi.org/10.1038/s41586-024-07487-w).

This is **inference only** (no training). Typical path:

1. JSON input (`name`, protein/RNA/ligand sequences, seeds).
2. Optional **data pipeline** (CPU): MSA + template search against large genetic databases (`fetch_databases.sh`, up to ~1 TB).
3. **Model inference** (GPU, JAX/Haiku): Evoformer pair/MSA trunk, then a diffusion head that denoises atomic coordinates, plus confidence heads (pLDDT, pTM, distogram).
4. Outputs: mmCIF/PDB-like structures and confidence metrics.

Entry point: `alphafold3/run_alphafold.py`. Code lives under `alphafold3/src/alphafold3/` (`data/`, `model/network/`, `structure/`, C++ parsers).

**Not included here:** model weights. Download separately from Google and keep under the [weights terms of use](alphafold3/WEIGHTS_TERMS_OF_USE.md). Source is Apache 2.0 (`alphafold3/LICENSE`). Full install is Linux + NVIDIA GPU (CC ≥ 8.0) + Docker; Windows is not supported for running AF3.

For this assignment you can also skip local AF3 and use [alphafoldserver.com](https://alphafoldserver.com/) (daily job limits apply).

## License notes

- AlphaFold 3 **source** is Apache 2.0. **Parameters and outputs** have additional terms in this folder.
- Assignment starter code is from the CMU GenAI BioMed course; keep it for coursework, not as an upstream fork.
