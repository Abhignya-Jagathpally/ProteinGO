"""GO-grounded Perceiver resampler: a few latent queries cross-attend to the residue states.

Latent queries start at k-means centroids of GO-definition text embeddings (the decoder's text space), so each
soft-prompt token begins as a 'functional topic' query. The latent-to-residue attention is returned for the
reverse trace.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class XAttn(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = dropout

    def forward(self, x, ctx, mask=None, need_weights=False):
        B, M, D = x.shape
        N = ctx.shape[1]
        q = self.q(x).view(B, M, self.h, self.dk).transpose(1, 2)
        k = self.k(ctx).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(ctx).view(B, N, self.h, self.dk).transpose(1, 2)
        am = None if mask is None else mask[:, None, None, :]
        if need_weights:
            att = (q @ k.transpose(-1, -2)) / self.dk ** 0.5
            if am is not None:
                att = att.masked_fill(~am, float("-inf"))
            att = att.softmax(-1)
            out = att @ v
            w = att.mean(1)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.drop if self.training else 0)
            w = None
        return self.o(out.transpose(1, 2).reshape(B, M, D)), w


class ResamplerBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln_x, self.ln_c, self.ln_s, self.ln_f = (nn.LayerNorm(d) for _ in range(4))
        self.cross = XAttn(d, heads, dropout)
        self.self_attn = XAttn(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, z, h, mask, need_weights=False):
        c, w = self.cross(self.ln_x(z), self.ln_c(h), mask, need_weights)
        z = z + c
        s, _ = self.self_attn(self.ln_s(z), self.ln_s(z))
        z = z + s
        return z + self.ff(self.ln_f(z)), w


class Resampler(nn.Module):
    def __init__(self, d_in: int, d: int = 512, n_latents: int = 16, depth: int = 2, heads: int = 8,
                 dropout: float = 0.1, init=None):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        if init is not None:
            q = torch.as_tensor(init, dtype=torch.float32)
            q = q / q.norm(dim=1, keepdim=True) * (d ** 0.5)
        else:
            q = torch.randn(n_latents, d)
        self.latents = nn.Parameter(q)
        self.blocks = nn.ModuleList(ResamplerBlock(d, heads, dropout) for _ in range(depth))
        self.ln = nn.LayerNorm(d)

    def forward(self, H, mask, need_weights=False):
        """H [B, L, d_in] residue states, mask [B, L] True on residues -> Z [B, M, d], attention list."""
        h = self.inp(H)
        z = self.latents[None].expand(H.shape[0], -1, -1)
        ws = []
        for blk in self.blocks:
            z, w = blk(z, h, mask, need_weights)
            ws.append(w)
        return self.ln(z), ws
