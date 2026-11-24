"""Layers of the looped LM: RMSNorm with a learnable gain, rotary embeddings, causal attention,
SwiGLU, and the pre-norm block. No biases, no dropout.

The rotary functions are copied from the TRM code (models/layers.py: rotate_half,
apply_rotary_pos_emb, RotaryEmbedding), so that looped_lm doesn't depend on it.
"""
import torch
import torch.nn.functional as F
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: [B, S, H, D]; cos, sin: [S, D] (float32). Computed in float32, returned in q's dtype."""
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """cos and sin tables for positions 0 .. max_position_embeddings - 1 (not saved in checkpoints)."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class RMSNorm(nn.Module):
    """x / rms(x) * gain, computed in float32; the gain starts at 1."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.qkv_proj = nn.Linear(width, 3 * num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q, k, v = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim).unbind(2)  # [B, S, H, D]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, S, D]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(width, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Block(nn.Module):
    """Pre-norm: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, width: int, num_heads: int, head_dim: int, mlp_hidden: int, eps: float):
        super().__init__()
        self.attn_norm = RMSNorm(width, eps)
        self.attn = CausalSelfAttention(width, num_heads, head_dim)
        self.mlp_norm = RMSNorm(width, eps)
        self.mlp = SwiGLU(width, mlp_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))
