"""Portable PyTorch implementations of the transformer layers."""
# Reference RoPE and RMSNorm files carry Copyright (c) 2022, 2023 Tri Dao.
# This adaptation expresses the operations with native PyTorch tensors.
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotary_cache(length, head_width, device):
    """Build the split-half rotary angles on the patch sequence axis."""
    frequencies = 1.0 / (10000 ** (torch.arange(0, head_width, 2, device=device) / head_width))
    positions = torch.arange(length, device=device) / 1
    phase = torch.outer(positions, frequencies)
    return phase.cos().bfloat16(), phase.sin().bfloat16()


def rotate(x, cache):
    cos, sin = (angle.to(x).view(1, x.shape[1], 1, -1) for angle in cache)
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, left * sin + right * cos), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        x32 = x.float()
        normalized = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + 1e-5)
        return normalized.to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    """Non-causal SDPA with per-head interleaved QKV projection storage."""

    def __init__(self, settings):
        super().__init__()
        width = settings.n_embd
        self.n_head = settings.n_head
        self.head_size = width // settings.n_head
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x, rope=None):
        batch, length, width = x.shape
        packed = self.qkv(x).reshape(batch, length, self.n_head, 3, self.head_size)
        q, k, v = packed.unbind(dim=-2)
        if rope is not None:
            q, k = rotate(q, rope), rotate(k, rope)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=None, dropout_p=0.0, scale=1.0 / math.sqrt(self.head_size),
            is_causal=False,
        )
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.norm_1 = RMSNorm(settings.n_embd)
        self.attn = SelfAttention(settings)
        self.norm_2 = RMSNorm(settings.n_embd)
        self.mlp = SwiGLU(settings.n_embd, settings.intermediate_size)

    def forward(self, x, rope=None):
        x = x + self.attn(self.norm_1(x), rope)
        return x + self.mlp(self.norm_2(x))
