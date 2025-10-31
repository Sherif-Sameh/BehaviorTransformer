"""
Contains all submodules needed for the GPT architecture.

Largely adapted from Andrej Karpathy's implementation of minGPT available at: 

<https://github.com/karpathy/minGPT/tree/master>
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    """Minimal implementation of multi-head causal self-attention.
    
    Args:
        embd_dim: Embedding dimension for input sequences.
        n_heads: Number of parllel attention heads.
        seq_len: Expected sequence length for input sequences.
        attn_pdrop: Dropout probability for attention weights.
        proj_pdrop: Dropout probability for output projection.
    """

    def __init__(
        self,
        embd_dim: int,
        n_heads: int,
        seq_len: int,
        attn_pdrop: float,
        proj_pdrop: float,
    ):
        super().__init__()
        assert (
            embd_dim % n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.embd_dim = embd_dim
        self.n_heads = n_heads
        self.seq_len = seq_len

        # Combine Key, Query, Value projections for all heads
        self.qkv_proj = nn.Linear(embd_dim, 3 * embd_dim)
        # Output projection
        self.out_proj = nn.Linear(embd_dim, embd_dim)

        # Dropouts for attention weights and output projection
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.out_dropout = nn.Dropout(proj_pdrop)

        # Mask for causal attention
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """"Forward pass for causal self-attention layer.
        
        Args:
            x: (B, T, D) Tensor of input embeddings.
        
        Returns:
            (B, T, D) Tensor of output embeddings after self-attention and output projection.
        """
        assert x.ndim == 3, f"Input must be a 3-dim tensor, got {x.ndim} dims."
        assert x.shape[1:] == (self.seq_len, self.embd_dim), \
            f"Input sample shape must be ({self.seq_len}, {self.embd_dim}), got {x.shape[1:]}."
        B, T, D = x.shape

        # Compute Q, K, V for all heads and split them
        q, k, v = self.qkv_proj(x).split(self.embd_dim, dim=2)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, T, h_embd_dim)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, T, h_embd_dim)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, T, h_embd_dim)

        # Compute masked attention matrix (B, nh, T, h_embd_dim) x (B, nh, h_embd_dim, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Compute output after self-attention and output projection
        out = att @ v  # (B, nh, T, T) x (B, nh, T, h_embd_dim) -> (B, nh, T, h_embd_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_dropout(self.out_proj(out))
        return out
    

class TransformerBlock(nn.Module):
    """Transformer block consisting of multi-head self-attention and FFN.
    
    Args:
        embd_dim: Embedding dimension for input sequences.
        n_heads: Number of parllel attention heads.
        seq_len: Expected sequence length for input sequences.
        attn_pdrop: Dropout probability for attention weights.
        proj_pdrop: Dropout probability for output projection.
    """

    def __init__(
        self,
        embd_dim: int,
        n_heads: int,
        seq_len: int,
        attn_pdrop: float,
        proj_pdrop: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embd_dim)
        self.attn = CausalSelfAttention(
            embd_dim, n_heads, seq_len, attn_pdrop, proj_pdrop
        )
        self.ln_2 = nn.LayerNorm(embd_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * embd_dim, embd_dim),
            nn.Dropout(proj_pdrop),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for transformer block.
        
        Args:
            x: (B, T, D) Tensor of input embeddings.
        
        Returns:
            (B, T, D) Tensor of output embeddings after self-attention and FFN with layer norm and
                residual connections.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x