# GPT-2 from Scratch — DL4NLP LAB1

A complete pipeline for pre-training a GPT-2 language model from scratch, fine-tuning it on a science QA task, and evaluating on in-domain and out-of-domain benchmarks.

## Overview

| Stage | Script | Description |
|-------|--------|-------------|
| Lab 1 | `lab1_preprocess.py` | Tokenise FineWeb-Edu 10BT → sharded `.npy` files |
| Lab 2 | `lab2_train.py` | Pre-train GPT-2 Small (124M) on 10B tokens |
| Lab 3 | `lab3_finetune.py` | Fine-tune on SciQ multiple-choice QA |
| Lab 4 | `lab4_evaluate.py` | Evaluate on SciQ test + ARC-Easy test |

## Results

| Model | SciQ test | ARC-Easy test |
|-------|-----------|---------------|
| Pre-trained only | 68.6% | 43.6% |
| Pre-trained + fine-tuned | **91.1%** | **44.7%** |
| Random init + fine-tuned | 53.7% | 34.0% |
| Random baseline | 25.0% | 25.0% |

## Model

GPT-2 Small decoder-only transformer implemented from scratch in PyTorch:

- 12 layers, 12 attention heads, d_model = 768
- Context length: 1,024 tokens
- Vocabulary: 50,257 (GPT-2 BPE via `tiktoken`)
- Flash attention (`F.scaled_dot_product_attention`)
- Weight tying (input embedding = output projection)
- **124M total parameters** (85.8M non-embedding)

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `tiktoken`, `datasets`, `numpy`, `tqdm`, `wandb`

## Usage

### Lab 1 — Preprocessing

Tokenise FineWeb-Edu and write sharded `.npy` files to `data/fineweb_edu/`:

```bash
python lab1_preprocess.py --num_procs 4 --shard_size 100000000
```

This produces ~100 shards of 100M tokens each (~20 GB total). The first shard is the validation set.

### Lab 2 — Pre-training

Train GPT-2 Small on 10B tokens (single GPU):

```bash
python lab2_train.py
```

Resume from a checkpoint:

```bash
python lab2_train.py --resume checkpoints/ckpt_step_8000.pt --wandb_id <run_id>
```

Key config (set at top of script): `MICRO_BATCH=32`, `BATCH_TOKENS=524288`, `SEQ_LEN=1024`.

### Lab 3 — Fine-tuning

Fine-tune on SciQ from a pre-trained checkpoint:

```bash
python lab3_finetune.py --checkpoint checkpoints/ckpt_final.pt
```

Fine-tune from random initialisation (Experiment 2):

```bash
python lab3_finetune.py --random_init
```

### Lab 4 — Evaluation

Evaluate a checkpoint on SciQ test and ARC-Easy test:

```bash
python lab4_evaluate.py --checkpoint checkpoints/ckpt_final.pt --name base_pretrained
python lab4_evaluate.py --checkpoint checkpoints/finetuned.pt --name finetuned
```

Results are printed to stdout and saved to `results_<name>.txt`.

## SLURM (NSC Berzelius)

Slurm scripts for the NSC Berzelius HPC cluster are in `slurm/`:

```bash
sbatch slurm/preprocess.sh   # Lab 1 — CPU node, 2h
sbatch slurm/train.sh        # Lab 2 — 1x A100 40GB, 48h
sbatch slurm/finetune.sh     # Lab 3 — 1x A100 40GB, 6h
sbatch slurm/evaluate.sh     # Lab 4 — 1x A100 40GB, 1h
```

## Repository Structure

```
.
├── model.py              # GPT-2 architecture (from scratch)
├── dataloader.py         # Sharded data loader with DDP support
├── lab1_preprocess.py    # FineWeb-Edu tokenisation & sharding
├── lab2_train.py         # Pre-training loop
├── lab3_finetune.py      # SciQ fine-tuning
├── lab4_evaluate.py      # Evaluation on SciQ + ARC-Easy
├── requirements.txt
├── slurm/
│   ├── preprocess.sh
│   ├── train.sh
│   ├── finetune.sh
│   └── evaluate.sh
└── data/
    └── fineweb_edu/      # Preprocessed shards (not tracked by git)
```

## Training Details

**Pre-training** — AdamW, cosine LR schedule (peak 6e-4, min 6e-5), 715 warmup steps, gradient accumulation to 524,288 tokens/step, bfloat16, 1× A100 40GB, ~48h.

**Fine-tuning** — Causal LM loss on answer tokens only (binary mask), 5 epochs, peak LR 1e-4, batch size 16, ~30 min.

**Evaluation** — Log-likelihood scoring: predict the answer choice with the highest normalised log-probability conditioned on the context.
