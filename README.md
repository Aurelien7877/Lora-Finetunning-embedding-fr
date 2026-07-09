# French Embedding Experiments

Work in progress.

This repository is a small research workspace for French and multilingual LLM experiments, with the main active thread focused on improving French embedding quality for MTEB-style evaluation.

## Current focus

The active project is in `emebedding/`:

- fine-tuning `intfloat/multilingual-e5-base` with LoRA for French retrieval and semantic similarity;
- mixing MIRACL-style retrieval data with French STS/paraphrase/NLI supervision;
- validating that LoRA weights actually train, reload correctly, and produce different embeddings from the baseline;
- evaluating on French MTEB tasks with a safer CPU-only runner after local GPU/CUDA instability caused hard machine resets.

## Repository layout

- `emebedding/e5_Lora.ipynb` - main notebook for the E5 French LoRA pipeline.
- `emebedding/mteb_safe_worker.py` - isolated worker used to evaluate one MTEB task at a time without keeping large models in the notebook kernel.
- `emebedding/*_fr.ipynb` and `emebedding/training.ipynb` - earlier French fine-tuning experiments.
- `few-nerd/` - separate Few-NERD / Unsloth GRPO experiments.
- `requirements.txt` and `setup.bat` - local environment setup helpers.

Large trained models, adapters, checkpoints, logs, caches, and benchmark outputs are intentionally ignored by git. They can be regenerated or kept locally when needed, but they should not be committed.

## Useful local artifacts

These are intentionally kept locally but ignored by git:

- `emebedding/e5-fr-miracl-lora-adapter/` - PEFT LoRA adapter export.
- `emebedding/e5-fr-miracl-lora-merged/` - merged SentenceTransformer model used for local MTEB evaluation.
- `emebedding/mteb_fra_results/` - local benchmark outputs.
- `few-nerd/grpo_saved_lora/` - final Few-NERD LoRA export.

## Status

The embedding pipeline is still experimental. The key engineering lesson so far is that evaluation reliability matters as much as model quality: adapter reload checks, embedding-delta checks, strict no-leak validation splits, isolated workers, and memory guards are now part of the workflow.
