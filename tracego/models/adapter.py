"""Stage A: ESM2 + LoRA + GO-grounded resampler, trained for GO supervision and functional retrieval.

Outputs per protein (one encoder pass, cached afterwards):
  Z  [M, d]  soft-prompt latents for the instruction decoder
  r  [d_r]   L2-normalised retrieval embedding (neighbour search is done in this space)
  h  [480]   adapted mean-pooled residue state (kept for the before/after-adapter analyses)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracego.models.encoder import ESMEncoder
from tracego.models.resampler import Resampler


class AdaptedEncoder(nn.Module):
    def __init__(self, esm_name: str, n_terms: int, term_init=None, latent_init=None, lora_r: int = 16,
                 d: int = 512, n_latents: int = 16, depth: int = 2, d_r: int = 512, dropout: float = 0.1,
                 lora_from_layer: int = 0):
        super().__init__()
        self.enc = ESMEncoder(esm_name, lora_r=lora_r, lora_from_layer=lora_from_layer)
        H = self.enc.hidden
        self.resampler = Resampler(H, d, n_latents, depth, dropout=dropout, init=latent_init)
        self.ret = nn.Sequential(nn.Linear(H + d, d_r), nn.GELU(), nn.Linear(d_r, d_r))
        self.cls_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        te = torch.as_tensor(term_init, dtype=torch.float32) if term_init is not None else torch.randn(n_terms, d)
        te = te / te.norm(dim=1, keepdim=True)
        self.term_emb = nn.Parameter(te)
        self.term_bias = nn.Parameter(torch.full((n_terms,), -4.0))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.log_tau = nn.Parameter(torch.tensor(-2.5))

    def forward(self, input_ids, attention_mask, need_weights=False):
        out = self.enc(input_ids, attention_mask)
        Hs = out.last_hidden_state
        m = self.enc.residue_mask(input_ids, attention_mask)
        mf = m.float()[..., None]
        h = (Hs * mf).sum(1) / mf.sum(1)
        Z, ws = self.resampler(Hs, m, need_weights)
        r = F.normalize(self.ret(torch.cat([h, Z.mean(1)], -1)), dim=-1)
        return {"Z": Z, "r": r, "h": h, "att": ws, "mask": m}

    def term_logits(self, Z):
        """Per-term logits from the latents: max over latents of the scaled cosine to the term embedding."""
        q = F.normalize(self.cls_proj(Z), dim=-1)
        e = F.normalize(self.term_emb, dim=-1)
        return self.scale * (q @ e.T).amax(1) + self.term_bias


def ia_bce(logits, Y, known, w_term):
    """IA-weighted BCE over the terms of the namespaces a protein is annotated in (open-world: an unannotated
    namespace is unknown, not negative). known [B, V] bool."""
    l = F.binary_cross_entropy_with_logits(logits, Y, reduction="none") * w_term[None]
    return (l * known).sum() / known.sum().clamp(min=1)


def soft_contrastive(rq, Yq, rk, Yk, w_term, tau, self_mask=None, label_tau: float = 0.05):
    """Functional retrieval loss: cross-entropy between the label-similarity distribution and the embedding-
    similarity distribution over the keys (the batch plus a queue of earlier batches, MoCo-style). Label
    similarity = cosine of the sqrt(IA)-weighted label vectors."""
    sw = w_term[None].sqrt()
    C = F.normalize(Yq * sw, dim=-1) @ F.normalize(Yk * sw, dim=-1).T
    valid = (Yq.sum(1) > 0)[:, None] & (Yk.sum(1) > 0)[None, :]
    if self_mask is not None:
        valid = valid & ~self_mask
    ok = valid.any(1)
    q = (C / label_tau).masked_fill(~valid, float("-inf"))[ok].softmax(-1)
    lp = ((rq @ rk.T) / tau).masked_fill(~valid, float("-inf"))[ok].log_softmax(-1)
    return -(q * lp.masked_fill(q == 0, 0)).sum(-1).mean()
