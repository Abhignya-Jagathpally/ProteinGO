"""Competitive (slot-attention) Perceiver resampler for Phase C1.

Copy of `tracego.models.resampler.Resampler` with one change in the latent↔residue cross-attention: the softmax
runs over latents for each residue (Locatello et al. 2020 slot attention), so the 16 soft-prompt slots compete for
each residue instead of each slot independently soft-maxing over the sequence. That is the collapse mode of the
stock resampler (latent–latent attention-map correlation 0.83–1.00). A diversity penalty on the attention maps is
added at training time (see `diversity_loss`).

Do not edit `tracego/models/resampler.py`; this module lives under analysis/ and is only used by the C1 retrain.

  from tracego.analysis.slot_resampler import CompetitiveResampler, diversity_loss, latent_corr
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompetitiveXAttn(nn.Module):
    """Multi-head cross-attention with slot-attention softmax (over queries / latents) per residue."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = dropout

    def forward(self, x, ctx, mask=None, need_weights=False):
        """x [B,M,D] latents, ctx [B,N,D] residues, mask [B,N] True on residues.
        Returns (out [B,M,D], attn [B,M,N] or None). Softmax is over M for each residue; each latent then takes a
        weighted mean of the residue values (re-normalised over N)."""
        B, M, D = x.shape
        N = ctx.shape[1]
        q = self.q(x).view(B, M, self.h, self.dk).transpose(1, 2)          # [B,H,M,dk]
        k = self.k(ctx).view(B, N, self.h, self.dk).transpose(1, 2)        # [B,H,N,dk]
        v = self.v(ctx).view(B, N, self.h, self.dk).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)              # [B,H,M,N]
        # Softmax over latents first. Do NOT put -inf on padded residues before this softmax: every latent
        # would be -inf there and softmax(-inf,...,-inf) is NaN. Zero the pad after the softmax instead.
        attn_m = logits.softmax(dim=-2)                                    # over M
        if mask is not None:
            attn_m = attn_m * mask[:, None, None, :].float()
        # weighted mean over residues (slot-attention update normalisation)
        mass = attn_m.sum(-1, keepdim=True).clamp(min=1e-9)
        w = attn_m / mass
        if self.training and self.drop > 0:
            w = F.dropout(w, self.drop)
        out = (w @ v).transpose(1, 2).reshape(B, M, D)
        weights = w.mean(1) if need_weights else None                      # [B,M,N]
        return self.o(out), weights


class CompetitiveBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln_x, self.ln_c, self.ln_s, self.ln_f = (nn.LayerNorm(d) for _ in range(4))
        self.cross = CompetitiveXAttn(d, heads, dropout)
        # latent self-attention stays standard (softmax over latents)
        from tracego.models.resampler import XAttn
        self.self_attn = XAttn(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, z, h, mask, need_weights=False):
        c, w = self.cross(self.ln_x(z), self.ln_c(h), mask, need_weights)
        z = z + c
        s, _ = self.self_attn(self.ln_s(z), self.ln_s(z))
        z = z + s
        return z + self.ff(self.ln_f(z)), w


class CompetitiveResampler(nn.Module):
    """Same interface as `Resampler`: forward(H, mask, need_weights) -> (Z, attention_list)."""

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
        self.blocks = nn.ModuleList(CompetitiveBlock(d, heads, dropout) for _ in range(depth))
        self.ln = nn.LayerNorm(d)

    def forward(self, H, mask, need_weights=False):
        h = self.inp(H)
        z = self.latents[None].expand(H.shape[0], -1, -1)
        ws = []
        for blk in self.blocks:
            z, w = blk(z, h, mask, need_weights)
            ws.append(w)
        return self.ln(z), ws


def diversity_loss(att: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine of per-latent attention maps. att [B,M,L], mask [B,L] True on residues.
    Zero when the M maps are orthogonal; ~1 under total collapse."""
    m = mask.float()
    a = att * m[:, None, :]
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    G = a @ a.transpose(-1, -2)                                            # [B,M,M]
    M = G.shape[-1]
    off = G.tril(-1).abs().sum((-1, -2)) + G.triu(1).abs().sum((-1, -2))
    return (off / (M * (M - 1))).mean()


@torch.no_grad()
def latent_corr(att: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean absolute off-diagonal Pearson correlation of attention maps (diagnostic)."""
    m = mask.float()
    a = att * m[:, None, :]
    a = a - a.sum(-1, keepdim=True) / m.sum(-1)[:, None, None].clamp(min=1)
    a = a * m[:, None, :]
    n = (a.pow(2).sum(-1) + 1e-9).sqrt()
    a = a / n[..., None]
    G = a @ a.transpose(-1, -2)
    M = G.shape[-1]
    off = G.tril(-1).abs().sum((-1, -2)) + G.triu(1).abs().sum((-1, -2))
    return float((off / (M * (M - 1))).mean())
