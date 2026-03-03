"""
Lab 2: Pre-train a GPT-2 style LLM on FineWeb-Edu 10B.

Launch with torchrun for multi-GPU:
  torchrun --standalone --nproc_per_node=8 lab2_train.py

Or single GPU:
  python lab2_train.py
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from tqdm import tqdm

from model import GPT, GPTConfig
from dataloader import ShardedDataLoader

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Training
TOTAL_TOKENS     = 10_000_000_000
BATCH_TOKENS     = 524_288          # tokens per step (512 seq × 1024 ctx)
SEQ_LEN          = 1024
MICRO_BATCH      = 32               # sequences per GPU per micro-step (64 OOMs on A100 40GB)

# Optimiser
LR_MAX           = 6e-4
LR_MIN           = 6e-5             # cosine decay floor
WARMUP_STEPS     = 715
WEIGHT_DECAY     = 0.1
BETAS            = (0.9, 0.95)
GRAD_CLIP        = 1.0

# Evaluation & checkpointing
VAL_EVERY        = 250              # validate every N steps
VAL_STEPS        = 20              # number of val batches to average
CKPT_EVERY       = 1000
LOG_EVERY        = 10

DATA_DIR         = "data/fineweb_edu"
CKPT_DIR         = "checkpoints"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--resume",    type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--wandb_id",  type=str, default=None, help="W&B run ID to resume (keeps one continuous plot)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------

ddp          = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank       = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device         = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank       = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device         = "cuda" if torch.cuda.is_available() else "cpu"

device_type = "cuda" if "cuda" in device else "cpu"

# Compute gradient accumulation steps to hit BATCH_TOKENS target
tokens_per_micro = MICRO_BATCH * SEQ_LEN * ddp_world_size
grad_accum_steps = BATCH_TOKENS // tokens_per_micro
assert BATCH_TOKENS % tokens_per_micro == 0, \
    f"BATCH_TOKENS ({BATCH_TOKENS}) must be divisible by {tokens_per_micro}"

total_steps = TOTAL_TOKENS // BATCH_TOKENS

if master_process:
    print(f"GPUs: {ddp_world_size}  |  micro_batch: {MICRO_BATCH}  |  grad_accum: {grad_accum_steps}")
    print(f"Tokens/step: {BATCH_TOKENS:,}  |  Total steps: {total_steps:,}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    wandb.init(
        project="llm-labs-pretrain",
        name="gpt2-small-10B",
        id=args.wandb_id,                          # resume existing run if provided
        resume="must" if args.wandb_id else None,  # "must" = fail if ID not found
        config={
            "total_tokens":    TOTAL_TOKENS,
            "batch_tokens":    BATCH_TOKENS,
            "seq_len":         SEQ_LEN,
            "micro_batch":     MICRO_BATCH,
            "grad_accum":      grad_accum_steps,
            "lr_max":          LR_MAX,
            "lr_min":          LR_MIN,
            "warmup_steps":    WARMUP_STEPS,
            "weight_decay":    WEIGHT_DECAY,
            "n_layers":        12,
            "n_heads":         12,
            "d_model":         768,
            "vocab_size":      50257,
            "gpus":            ddp_world_size,
        },
    )

# ---------------------------------------------------------------------------
# torch.compile / dtype
# ---------------------------------------------------------------------------

torch.set_float32_matmul_precision("high")
dtype        = torch.bfloat16
autocast_ctx = torch.autocast(device_type=device_type, dtype=dtype)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

config = GPTConfig()   # GPT-2 Small: 12 layers, 12 heads, d_model=768
model  = GPT(config).to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model   # unwrapped model for saving

if master_process:
    print(f"Parameters: {raw_model.num_params()/1e6:.1f}M")

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

optimizer = raw_model.configure_optimizer(LR_MAX, WEIGHT_DECAY, BETAS, device_type)

# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    if step >= total_steps:
        return LR_MIN
    # Cosine decay from warmup end to total_steps
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

train_loader = ShardedDataLoader(DATA_DIR, "train", MICRO_BATCH, SEQ_LEN,
                                 process_rank=ddp_rank, num_processes=ddp_world_size)
val_loader   = ShardedDataLoader(DATA_DIR, "val",   MICRO_BATCH, SEQ_LEN,
                                 process_rank=ddp_rank, num_processes=ddp_world_size)

# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------

start_step = 0
if args.resume is not None:
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    if master_process:
        print(f"Resumed from {args.resume}  (step {start_step})")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate():
    model.eval()
    val_loader.reset()
    losses = []
    for _ in range(VAL_STEPS):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    val_loss = sum(losses) / len(losses)
    # Reduce across DDP processes
    if ddp:
        t = torch.tensor(val_loss, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        val_loss = t.item()
    return val_loss

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
t0 = time.time()

pbar = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps,
            desc="Training", disable=not master_process)

for step in pbar:

    # --- LR update ---
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # --- Validation ---
    if step % VAL_EVERY == 0:
        val_loss = evaluate()
        if master_process:
            val_ppl = math.exp(val_loss)
            pbar.write(f"[step {step:6d}]  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")
            wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)

    # --- Gradient accumulation ---
    optimizer.zero_grad()
    accum_loss = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Only sync gradients on the last micro-step
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        with autocast_ctx:
            _, loss = model(x, y)

        loss = loss / grad_accum_steps
        accum_loss += loss.item()
        loss.backward()

    # --- Gradient clipping ---
    if ddp:
        dist.all_reduce(torch.tensor(accum_loss, device=device), op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    optimizer.step()

    # --- Progress bar & logging ---
    if master_process:
        pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}")

    if step % LOG_EVERY == 0 and master_process:
        t1    = time.time()
        dt    = t1 - t0
        t0    = t1
        tps   = BATCH_TOKENS * LOG_EVERY / dt
        pbar.write(f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                   f"{tps/1e6:.2f}M tok/s | {dt/LOG_EVERY*1000:.0f}ms/step")
        wandb.log({
            "train/loss":          accum_loss,
            "train/lr":            lr,
            "perf/tokens_per_sec": tps,
            "perf/ms_per_step":    dt / LOG_EVERY * 1000,
        }, step=step)

    # --- Checkpoint ---
    if step % CKPT_EVERY == 0 and master_process:
        ckpt_path = os.path.join(CKPT_DIR, f"ckpt_step{step:06d}.pt")
        torch.save({
            "step":      step,
            "model":     raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    config,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

# --- Final checkpoint ---
if master_process:
    ckpt_path = os.path.join(CKPT_DIR, "ckpt_final.pt")
    torch.save({
        "step":      total_steps,
        "model":     raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    config,
    }, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")

if master_process:
    wandb.finish()

if ddp:
    dist.destroy_process_group()
