"""
DataLoader for pre-tokenized sharded .npy files produced by lab1_preprocess.py.
Supports multi-GPU training via DDP (each process reads a different slice).
"""

import os
import glob
import numpy as np
import torch


class ShardedDataLoader:
    """
    Reads sharded uint16 .npy token files sequentially.

    Each DDP process reads a non-overlapping slice of every batch so that
    together they consume B * T * num_processes tokens per step.
    """

    def __init__(
        self,
        data_dir:      str,
        split:         str,   # "train" or "val"
        batch_size:    int,   # sequences per process per step
        seq_len:       int,   # tokens per sequence
        process_rank:  int = 0,
        num_processes: int = 1,
    ):
        self.batch_size    = batch_size
        self.seq_len       = seq_len
        self.process_rank  = process_rank
        self.num_processes = num_processes

        pattern = os.path.join(data_dir, f"fineweb_edu_{split}_*.npy")
        shards  = sorted(glob.glob(pattern))
        assert len(shards) > 0, f"No shards found matching: {pattern}"
        self.shards = shards

        self.reset()

    # ------------------------------------------------------------------

    def reset(self):
        """Rewind to the beginning (call between epochs or for val eval)."""
        self.shard_idx = 0
        self.tokens    = self._load_shard(self.shards[0])
        # Each process starts at its own offset within the first shard
        self.pos = self.batch_size * self.seq_len * self.process_rank

    def _load_shard(self, path: str) -> torch.Tensor:
        data = np.load(path).astype(np.int64)
        return torch.from_numpy(data)

    # ------------------------------------------------------------------

    def next_batch(self):
        """
        Returns (x, y) tensors of shape (batch_size, seq_len).
        x = input tokens, y = next-token targets (x shifted by 1).
        """
        B, T = self.batch_size, self.seq_len

        buf = self.tokens[self.pos : self.pos + B * T + 1]
        x   = buf[:-1].view(B, T)
        y   = buf[1:].view(B, T)

        # Advance position past all processes' slices
        self.pos += B * T * self.num_processes

        # Load next shard if current one is exhausted
        if self.pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self.tokens    = self._load_shard(self.shards[self.shard_idx])
            self.pos       = B * T * self.process_rank

        return x, y
