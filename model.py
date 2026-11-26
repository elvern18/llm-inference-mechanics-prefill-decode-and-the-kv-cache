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

# Step 3 - GQAAttention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

class GQAAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, max_len=2048):
        super().__init__()
        # TODO: head_dim, wq, wk, wv, wo, rotary buffers cos and sin
        self.d = d 
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_len = max_len 
        self.head_dim = d // n_heads 

        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

        cos, sin = rope_cache(max_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(self, x, cache=None):
        B, T, _ = x.shape

        # -----------------------------------
        # 1. Determine past sequence length
        # -----------------------------------
        if cache is None:
            S_past = 0
        else:
            k_past, v_past = cache
            S_past = k_past.shape[2]

        # -----------------------------------
        # 2. Project Q, K, V
        # -----------------------------------
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # -----------------------------------
        # 3. Split into heads
        # -----------------------------------
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # (B, T, heads, head_dim)
        #        ↓
        # (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -----------------------------------
        # 4. Apply RoPE
        # -----------------------------------
        q = apply_rope(q, self.cos, self.sin, offset=S_past)
        k = apply_rope(k, self.cos, self.sin, offset=S_past)

        # -----------------------------------
        # 5. Append to KV cache
        # -----------------------------------
        if cache is None:
            k_all = k
            v_all = v
        else:
            k_past, v_past = cache

            k_all = torch.cat([k_past, k], dim=2)
            v_all = torch.cat([v_past, v], dim=2)

        # -----------------------------------
        # 6. Expand KV heads to Q heads
        # -----------------------------------
        repeat_factor = self.n_heads // self.n_kv_heads

        k_attn = k_all.repeat_interleave(
            repeat_factor,
            dim=1,
        )
        v_attn = v_all.repeat_interleave(
            repeat_factor,
            dim=1,
        )

        # -----------------------------------
        # 7. Attention scores
        # -----------------------------------
        scores = (
            q @ k_attn.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # -----------------------------------
        # 8. Causal mask
        # -----------------------------------
        S = k_all.shape[2]

        query_positions = torch.arange(
            S_past,
            S_past + T,
            device=x.device,
        )

        key_positions = torch.arange(
            S,
            device=x.device,
        )

        mask = (
            key_positions[None, :]
            <= query_positions[:, None]
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        # -----------------------------------
        # 9. Softmax
        # -----------------------------------
        attn = F.softmax(scores, dim=-1)

        # -----------------------------------
        # 10. Weighted sum of V
        # -----------------------------------
        out = attn @ v_attn

        # -----------------------------------
        # 11. Merge heads
        # -----------------------------------
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.d)

        # -----------------------------------
        # 12. Output projection
        # -----------------------------------
        out = self.wo(out)

        # Return UNEXPANDED KV cache
        return out, (k_all, v_all)

# Step 4 - TinyLM
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, hidden, max_len):
        super().__init__()

        self.norm1 = RMSNorm(d)
        self.attn = GQAAttention(
            d,
            n_heads,
            n_kv_heads,
            max_len=max_len,
        )

        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, hidden)

    def forward(self, x, cache=None):
        # Pre-norm attention + residual
        h = self.norm1(x)
        attn_out, new_cache = self.attn(h, cache)
        x = x + attn_out

        # Pre-norm MLP + residual
        h = self.norm2(x)
        x = x + self.mlp(h)

        return x, new_cache


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 

        self.tok = nn.Embedding(cfg["vocab"], cfg["d"])

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d=cfg["d"],
                n_heads=cfg["n_heads"],
                n_kv_heads=cfg["n_kv_heads"],
                hidden=cfg["hidden"],
                max_len=cfg["max_len"],
            )
            for _ in range(cfg["n_layers"])
        ])

        self.norm = RMSNorm(cfg["d"])

        self.head = nn.Linear(
            cfg["d"],
            cfg["vocab"],
            bias=False,
        )

    def forward(self, idx, cache=None):
        x = self.tok(idx)

        new_cache = []

        for i, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[i]

            x, layer_new_cache = block(x, layer_cache)

            new_cache.append(layer_new_cache)

        x = self.norm(x)

        logits = self.head(x)

        return logits, new_cache


def make_cfg(
    vocab=64,
    d=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    hidden=None,
    max_len=2048,
):
    if hidden is None:
        hidden = 4 * d

    return {
        "vocab": vocab,
        "d": d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden": hidden,
        "max_len": max_len,
    }

# Step 5 - prefill
import torch

@torch.no_grad()
def prefill(model, prompt):
    next_logits, cache = model(prompt.unsqueeze(0))
    next_logits = next_logits[0, -1, :]
    return next_logits, cache

@torch.no_grad()
def decode_step(model, token, cache):
    token = torch.tensor([[token]])
    next_logits, cache = model(token, cache)
    next_logits = next_logits[0, -1, :]
    return next_logits, cache

@torch.no_grad()
def generate(model, prompt, n, greedy=True, gen=None):
    # TODO: prefill once, then n decode passes; return (tokens, passes)
    logits, cache = prefill(model, prompt) 

    if greedy:
        token = torch.argmax(logits).item() 
    else:
        probs = torch.softmax(logits, dim = -1)
        token = torch.multinomial(probs, num_samples = 1, generator = gen).item() 

    tokens = [token] 

    # Generate remaining n - 1 tokens
    for _ in range(n - 1):
        next_logits, cache = decode_step(
            model,
            token,
            cache
        )

        if greedy:
            token = torch.argmax(next_logits).item()
        else:
            probs = torch.softmax(next_logits, dim=-1)
            token = torch.multinomial(
                probs,
                num_samples=1,
                generator=gen
            ).item()

        tokens.append(token)
    
    return (tokens, n) 

@torch.no_grad()
def generate_recompute(model, prompt, n):
    # TODO: greedy tokens by re-running the whole sequence each step
    tokens = []

    seq = prompt.clone()

    for _ in range(n):
        logits, _ = model(seq.unsqueeze(0))

        next_logits = logits[0, -1, :]

        token = torch.argmax(next_logits).item()

        tokens.append(token)

        seq = torch.cat([
            seq,
            torch.tensor(
                [token],
            )
        ])

    return tokens, n

# Step 6 - kv_cache_bytes
import torch

'''
        "vocab": vocab,
        "d": d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden": hidden,
        "max_len": max_len,
        '''

def kv_bytes_per_token(cfg, bytes_per_elem):
    return cfg['n_layers'] * 2 * cfg['n_kv_heads'] * cfg['d'] * bytes_per_elem // cfg['n_heads']

def kv_cache_bytes(cfg, seq_len, batch, bytes_per_elem):
    return batch * seq_len * kv_bytes_per_token(cfg, bytes_per_elem)

def measured_cache_bytes(cache):
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in layer
    )

def max_context(cfg, memory_bytes, batch, bytes_per_elem, weight_bytes):
    remaining_bytes = memory_bytes - weight_bytes
    seq_len = 0 
    while kv_cache_bytes(cfg, seq_len + 1, batch, bytes_per_elem) < remaining_bytes:
        seq_len += 1 
    return seq_len 
    
def cache_growth(model, prompt,n):
    measure_cache_bytes_at_each_step = [] 
    logits, cache = prefill(model, prompt) 
    measure_cache_bytes_at_each_step.append(measured_cache_bytes(cache)) 
    token = torch.argmax(logits)

    for _ in range(n): 
        logits, cache = decode_step(model, token.item(), cache)
        measure_cache_bytes_at_each_step.append(measured_cache_bytes(cache)) 
        token = torch.argmax(logits)
    
    return measure_cache_bytes_at_each_step

# Step 7 - flops_per_token
def param_count(cfg):
    d = cfg['d']
    vocab = cfg['vocab']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    n_kv_heads = cfg['n_kv_heads'] 
    hidden = cfg['hidden']

    return vocab * d + n_layers * (
        2 * d * d + 2 * d * n_kv_heads * d // n_heads + 3 * d * hidden + 2 * d
    ) + d + d * vocab

def matmul_params(cfg):
    d = cfg['d']
    vocab = cfg['vocab']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    n_kv_heads = cfg['n_kv_heads'] 
    hidden = cfg['hidden']

    return n_layers * (
        2 * d * d + 2 * d * n_kv_heads * d // n_heads + 3 * d * hidden 
    ) + d * vocab

def flops_per_token(cfg, context_len):
    d = cfg['d']
    vocab = cfg['vocab']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    n_kv_heads = cfg['n_kv_heads'] 
    hidden = cfg['hidden']

    return 2 * matmul_params(cfg) + 4 * n_layers * d * context_len


def prefill_flops(cfg, n):
    return sum(
        flops_per_token(cfg, context_len)
        for context_len in range(1, n + 1)
    )
    

def decode_flops(cfg, context_len):
    return flops_per_token(cfg, context_len)

