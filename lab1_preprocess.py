"""
Lab 1: Data Preprocessing Pipeline
Tokenizes the FineWeb-Edu 10B dataset using GPT-2 BPE tokenizer (tiktoken)
and saves sharded uint16 .npy files for fast training.
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from multiprocessing import Pool
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Preprocess FineWeb-Edu dataset")
parser.add_argument("--num_procs", type=int, default=4,
                    help="Number of worker processes (default: 4)")
parser.add_argument("--shard_size", type=int, default=100_000_000,
                    help="Tokens per shard (default: 100M)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SHARD_SIZE  = args.shard_size
DATA_DIR    = "data/fineweb_edu"
NUM_PROCS   = args.num_procs
DATASET     = "HuggingFaceFW/fineweb-edu"
DATASET_CFG = "sample-10BT"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]   # token 50256 — document separator


# ---------------------------------------------------------------------------
# Tokenization (runs in worker processes)
# ---------------------------------------------------------------------------
def tokenize(doc):
    """Tokenize a single document. Prepends EOT to separate documents."""
    tokens = [EOT]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    arr = np.array(tokens, dtype=np.uint16)
    return arr


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------
def write_shard(filename, tokens):
    np.save(filename, tokens)
    size_mb = tokens.nbytes / 1024 / 1024
    print(f"  Saved {filename}  ({len(tokens)/1e6:.1f}M tokens, {size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading dataset: {DATASET} / {DATASET_CFG}")
    dataset = load_dataset(
        DATASET,
        name=DATASET_CFG,
        split="train",
        streaming=True,       # stream to avoid downloading everything upfront
    )

    shard_idx = 0
    shard_buf = np.empty((SHARD_SIZE,), dtype=np.uint16)
    buf_pos   = 0
    is_first  = True          # first shard becomes validation set
    total_tokens = 0

    print(f"Tokenizing with {NUM_PROCS} workers, shard size = {SHARD_SIZE/1e6:.0f}M tokens")

    with Pool(NUM_PROCS) as pool:
        pbar = tqdm(unit="tokens", unit_scale=True, desc="Tokenizing")

        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            pbar.update(len(tokens))
            total_tokens += len(tokens)

            # Fill current shard, spilling remainder into next
            offset = 0
            while offset < len(tokens):
                space    = SHARD_SIZE - buf_pos
                to_write = min(space, len(tokens) - offset)

                shard_buf[buf_pos : buf_pos + to_write] = tokens[offset : offset + to_write]
                buf_pos += to_write
                offset  += to_write

                # Shard is full — flush to disk
                if buf_pos == SHARD_SIZE:
                    split = "val" if is_first else "train"
                    fname = os.path.join(DATA_DIR, f"fineweb_edu_{split}_{shard_idx:06d}.npy")
                    write_shard(fname, shard_buf)
                    shard_idx += 1
                    buf_pos    = 0
                    is_first   = False

        pbar.close()

    # Write any remaining tokens in the last partial shard
    if buf_pos > 0:
        fname = os.path.join(DATA_DIR, f"fineweb_edu_train_{shard_idx:06d}.npy")
        write_shard(fname, shard_buf[:buf_pos])
        shard_idx += 1

    print(f"\nDone. Total tokens: {total_tokens/1e9:.2f}B")
    print(f"Shards written: {shard_idx}  (1 val + {shard_idx-1} train)")
    print(f"Output directory: {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
