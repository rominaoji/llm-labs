"""
Lab 3: Fine-tune the pre-trained GPT-2 on SciQ (science QA).

Fine-tuning approach: causal LM on answer tokens only.
Each example is formatted as:
    "{support} {question} {correct_answer}<|endoftext|>"
Loss is computed only on the answer tokens.

Usage:
    python lab3_finetune.py --checkpoint checkpoints/ckpt_final.pt
    python lab3_finetune.py --checkpoint checkpoints/ckpt_final.pt --random_init
"""

import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken
import wandb
from tqdm import tqdm

from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCHS       = 5
LR_MAX       = 1e-4
LR_MIN       = 1e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE   = 16
MAX_LEN      = 256        # max tokens per example (SciQ is short)
GRAD_CLIP    = 1.0
CKPT_DIR     = "checkpoints/finetune"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  type=str,  required=True,
                    help="Path to pre-trained checkpoint (lab2 output)")
parser.add_argument("--random_init", action="store_true",
                    help="Skip loading weights — train from random init (Experiment 2 baseline)")
args = parser.parse_args()

os.makedirs(CKPT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
enc    = tiktoken.get_encoding("gpt2")
EOT    = enc._special_tokens["<|endoftext|>"]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SciQDataset(Dataset):
    """
    Formats each SciQ example as:
        {support} {question} {correct_answer}<EOT>
    Returns (input_ids, loss_mask) where loss_mask=1 only on answer tokens.
    """
    def __init__(self, split: str, max_len: int = MAX_LEN):
        raw = load_dataset("allenai/sciq", split=split)
        self.examples = []

        for item in raw:
            support  = item["support"].strip()
            question = item["question"].strip()
            answer   = item["correct_answer"].strip()

            context_str = (f"{support} {question}" if support else question)
            answer_str  = f" {answer}"

            context_ids = enc.encode_ordinary(context_str)
            answer_ids  = enc.encode_ordinary(answer_str) + [EOT]

            all_ids = context_ids + answer_ids

            # Truncate from the left if too long
            if len(all_ids) > max_len:
                all_ids = all_ids[-max_len:]
                # Recompute where answer starts after truncation
                answer_start = max(0, len(all_ids) - len(answer_ids))
            else:
                answer_start = len(context_ids)

            # loss_mask: 1 on answer tokens only
            loss_mask = [0] * answer_start + [1] * (len(all_ids) - answer_start)

            self.examples.append({
                "input_ids":  torch.tensor(all_ids,   dtype=torch.long),
                "loss_mask":  torch.tensor(loss_mask, dtype=torch.float),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad a batch of variable-length examples to the same length."""
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)
    loss_masks = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, x in enumerate(batch):
        n = x["input_ids"].size(0)
        input_ids[i, :n]  = x["input_ids"]
        loss_masks[i, :n] = x["loss_mask"]

    return input_ids, loss_masks


# ---------------------------------------------------------------------------
# Log-likelihood accuracy (used for validation during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def lm_accuracy(model, dataset_split: str, device, max_examples: int = 500) -> float:
    """
    Evaluate accuracy using log-likelihood scoring on multiple-choice examples.
    Scores each answer choice independently and picks the highest.
    """
    model.eval()
    raw     = load_dataset("allenai/sciq", split=dataset_split)
    correct = 0
    total   = min(len(raw), max_examples)

    for item in tqdm(list(raw)[:total], desc="Val accuracy", file=sys.stdout, leave=False):
        support  = item["support"].strip()
        question = item["question"].strip()
        context  = f"{support} {question}" if support else question
        choices  = [item["correct_answer"],
                    item["distractor1"],
                    item["distractor2"],
                    item["distractor3"]]

        scores = []
        for choice in choices:
            score = score_choice(model, context, choice.strip(), device)
            scores.append(score)

        pred_idx = scores.index(max(scores))
        if pred_idx == 0:   # correct_answer is always index 0
            correct += 1

    model.train()
    return correct / total


def score_choice(model, context: str, choice: str, device) -> float:
    """Normalized log P(choice | context)."""
    ctx_ids    = enc.encode_ordinary(context)
    choice_ids = enc.encode_ordinary(" " + choice)
    all_ids    = ctx_ids + choice_ids

    x = torch.tensor(all_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
    y = torch.tensor(all_ids[1:],  dtype=torch.long).unsqueeze(0).to(device)

    # Pass y to get full-sequence logits (not just last position)
    logits, _ = model(x, y)
    log_probs  = F.log_softmax(logits, dim=-1)

    # Score only the choice tokens
    start     = len(ctx_ids) - 1
    choice_lp = log_probs[0, start:, :]
    choice_y  = y[0, start:]

    score = choice_lp[range(len(choice_y)), choice_y].sum().item()
    return score / len(choice_ids)   # normalize by answer length


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

config = GPTConfig()
model  = GPT(config).to(device)

if not args.random_init:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded pre-trained weights from {args.checkpoint}")
else:
    print("Random init — skipping pre-trained weights (Experiment 2 baseline)")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_dataset = SciQDataset("train")
val_dataset   = SciQDataset("validation")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
print(f"Train: {len(train_dataset)} examples  |  Val: {len(val_dataset)} examples")

# ---------------------------------------------------------------------------
# Optimiser & LR schedule
# ---------------------------------------------------------------------------

optimizer   = model.configure_optimizer(LR_MAX, WEIGHT_DECAY, (0.9, 0.95), device)
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(0.1 * total_steps)


def get_lr(step: int) -> float:
    if step < warmup_steps:
        return LR_MAX * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

run_name = "finetune-sciq-random-init" if args.random_init else "finetune-sciq-pretrained"
wandb.init(
    project="llm-labs-finetune",
    name=run_name,
    config={
        "epochs":       EPOCHS,
        "lr_max":       LR_MAX,
        "batch_size":   BATCH_SIZE,
        "random_init":  args.random_init,
        "checkpoint":   args.checkpoint if not args.random_init else "none",
    },
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
step = 0

for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)

    for input_ids, loss_mask in pbar:
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)

        # Update LR
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward pass — get per-token logits
        x      = input_ids[:, :-1]
        y      = input_ids[:, 1:]
        mask   = loss_mask[:, 1:]

        # Pass y as targets to get full-sequence logits (not just last position)
        # The model's internal loss is discarded — we compute our own masked loss
        logits, _ = model(x, y)

        # Compute loss only on answer tokens (where mask == 1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        )
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
        step += 1

    # Validation accuracy at end of each epoch
    val_acc = lm_accuracy(model, "validation", device)
    print(f"Epoch {epoch+1}  val_accuracy={val_acc:.4f} ({val_acc*100:.1f}%)")
    wandb.log({"val/accuracy": val_acc}, step=step)

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_epoch{epoch+1}.pt")
    torch.save({"epoch": epoch + 1, "model": model.state_dict(), "config": config}, ckpt_path)
    print(f"Saved: {ckpt_path}")

wandb.finish()
print("Fine-tuning complete.")
