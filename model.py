"""
GPT-2 style language model implemented from scratch.
Architecture: decoder-only transformer with causal self-attention.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50257   # GPT-2 BPE vocabulary
    max_ctx:    int = 1024    # maximum sequence length
    n_layers:   int = 12      # number of transformer blocks
    n_heads:    int = 12      # number of attention heads
    d_model:    int = 768     # embedding dimension
    dropout:    float = 0.0   # dropout (0 for pre-training at scale)
    bias:       bool = True   # bias in linear layers and layer norms


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.c_attn  = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj  = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.c_proj.SCALE_INIT = 1   # flag: scaled init applied in GPT._init_weights

        self.n_heads    = config.n_heads
        self.d_model    = config.d_model
        self.head_dim   = config.d_model // config.n_heads
        self.dropout    = config.dropout
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project to Q, K, V and split heads
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash attention (fused CUDA kernel — much faster than manual attention)
        attn_dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_dropout)

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.c_proj.SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # pre-norm residual
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.d_model),   # token embeddings
            'wpe':  nn.Embedding(config.max_ctx,    config.d_model),   # position embeddings
            'drop': nn.Dropout(config.dropout),
            'h':    nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            'ln_f': nn.LayerNorm(config.d_model, bias=config.bias),    # final layer norm
        })
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: input embeddings == output projection (saves ~38M params)
        self.transformer['wte'].weight = self.lm_head.weight

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) as in GPT-2 paper
        for pn, p in self.named_parameters():
            if hasattr(p, 'SCALE_INIT'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        _, T = idx.size()
        assert T <= self.config.max_ctx, f"Sequence length {T} exceeds max context {self.config.max_ctx}"

        pos = torch.arange(T, device=idx.device)
        x   = self.transformer['drop'](
            self.transformer['wte'](idx) + self.transformer['wpe'](pos)
        )
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def num_params(self) -> int:
        """Parameter count excluding tied embedding weights (counted once)."""
        n = sum(p.numel() for p in self.parameters())
        n -= self.transformer['wte'].weight.numel()   # tied — don't double count
        return n

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple, device_type: str):
        """AdamW with weight decay on 2D+ params only (no decay on biases/norms)."""
        decay_params   = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]

        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        # Use fused AdamW kernel if available (faster on CUDA)
        use_fused = device_type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Autoregressive generation for qualitative evaluation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_ctx else idx[:, -self.config.max_ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
