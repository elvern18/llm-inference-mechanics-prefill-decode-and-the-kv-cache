"""
LLM Inference Mechanics: Prefill, Decode and the KV Cache

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - RMSNorm
import torch
import torch.nn as nn

def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        # TODO: weight parameter of ones
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps) 

def swiglu(x, w1, w3, w2):
    return w2(nn.functional.silu(w1(x)) * w3(x))

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        # TODO: w1, w3 (d -> hidden), w2 (hidden -> d), no bias
        
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)


    def forward(self, x):
        return swiglu(x, self.w1, self.w3, self.w2)

# Step 2 - rope_cache
import torch

def rope_cache(max_len, head_dim, base=10000.0):
    """Precompute cos/sin tables of shape (max_len, head_dim // 2).

    Entry [t, i] uses angle t * base ** (-2 i / head_dim):
      row t    = token position (0, 1, 2, ...)
      column i = which 2D pair ("clock hand"); i = 0 spins fastest.
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.float32)
    inv_freq = base ** (-2.0 * i / head_dim)          # (half,)  theta_i, fast -> slow
    t = torch.arange(max_len, dtype=torch.float32)    # (max_len,)
    angles = torch.outer(t, inv_freq)                 # (max_len, half): angles[t, i] = t * theta_i
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin, offset=0):
    """Rotate x of shape (B, H, T, head_dim) using split-half pairing.

    Pair i is (x[..., i], x[..., i + head_dim // 2]), rotated by angle
    (offset + t) * theta_i. `offset` is the absolute position of the first
    of the T tokens (used when decoding with a KV cache).
    """
    T, D = x.shape[-2], x.shape[-1]
    half = D // 2
    if offset + T > cos.shape[0]:
        raise ValueError(f"offset + T = {offset + T} exceeds cache length {cos.shape[0]}")
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[offset:offset + T].to(device=x.device, dtype=x.dtype)  # (T, half), broadcasts over B, H
    s = sin[offset:offset + T].to(device=x.device, dtype=x.dtype)
    # 2D rotation [[c, -s], [s, c]] applied to every pair (x1, x2) at once
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def rope_is_relative(cos, sin, head_dim, seed=0):
    """q·k should depend only on the position gap: (5, 2) vs (15, 12), gap 3 both times."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, 1, head_dim, generator=g, dtype=cos.dtype)
    k = torch.randn(1, 1, 1, head_dim, generator=g, dtype=cos.dtype)

    def dot_at(pq, pk):
        rq = apply_rope(q, cos, sin, offset=pq)
        rk = apply_rope(k, cos, sin, offset=pk)
        return (rq * rk).sum().item()

    return abs(dot_at(5, 2) - dot_at(15, 12)) < 1e-4

