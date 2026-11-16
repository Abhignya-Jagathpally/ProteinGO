"""Stage B: retrieval-conditioned instruction decoder (flan-t5-small) that writes GO terms as tokens.

Encoder input  = [16 soft-prompt latents from the adapted encoder] + instruction text + evidence tokens
                 (<nb> <sim_b> <tax_rank> GO:leaf ... per retrieved neighbour).
Decoder output = GO tokens of one namespace in a topological order, then </s>.

Two heads read every decoder state h_s:
  next-token  logits over the namespace's GO tokens + </s>; with the hierarchy constraint the softmax is restricted
              to the frontier F(S_s) (terms whose in-vocabulary ancestors are all emitted) plus </s>.
  membership  sigmoid per term: does t belong to the protein, given the committed prefix S_s? (IA-weighted BCE)
Both add a learned monotone copy bias of the neighbour vote (pointer-generator style).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SIM_BINS = 24
TAX_TOKENS = ["<tax_species>", "<tax_genus>", "<tax_family>", "<tax_order>", "<tax_class>", "<tax_phylum>",
              "<tax_kingdom>", "<tax_domain>", "<tax_none>"]
# Phase A1 homology-curriculum tokens: unmasked example, then one token per FR24-P identity bin of the cutoff
HOM_TOKENS = ["<hom_full>", "<hom_no_hit>", "<hom_lt30>", "<hom_30_50>", "<hom_50_80>", "<hom_80_95>",
              "<hom_95_99>", "<hom_100>"]


def special_tokens(terms, homology: bool = False) -> list[str]:
    # HOM tokens are appended after GO terms so an existing checkpoint's go0..goN indices stay fixed
    base = ["<nb>"] + [f"<sim_{i}>" for i in range(N_SIM_BINS)] + TAX_TOKENS + list(terms)
    return base + (HOM_TOKENS if homology else [])


class Hierarchy:
    """In-vocabulary ancestor structure of one namespace, as dense GPU tensors over its local term index."""

    def __init__(self, anc_pairs: np.ndarray, cols: np.ndarray, device):
        loc = -np.ones(int(max(cols.max(), anc_pairs.max())) + 1, np.int64)
        loc[cols] = np.arange(len(cols))
        c, p = loc[anc_pairs[:, 0]], loc[anc_pairs[:, 1]]
        ok = (c >= 0) & (p >= 0) & (c != p)
        V = len(cols)
        D = torch.zeros(V, V, dtype=torch.float16)
        D[p[ok], c[ok]] = 1  # D[a, t] = 1 when a is an ancestor of t (row a: descendants of a)
        self.V = V
        self.desc = D.to(device)
        self.n_anc = D.sum(0).to(device)  # number of in-vocabulary ancestors of t
        self.parents_np = (c[ok], p[ok])

    def frontier(self, S: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
        """S [.., V] emitted (bool), cnt [.., V] emitted-ancestor counts -> legal next terms."""
        return (~S) & (cnt >= self.n_anc - 0.5)


class CopyBias(nn.Module):
    """phi(v) = a * logit(clip(v)) + b * [v > 0], one (a, b) per namespace and head."""

    def __init__(self, mem_bias: float):
        super().__init__()
        # initialised so that membership starts exactly at the neighbour vote: sigmoid(logit(v)) for v > 0
        # (the -mem_bias offset cancels the membership bias), sigmoid(mem_bias) ~ 0 for unvoted terms
        self.a = nn.Parameter(torch.ones(3, 2))
        self.b = nn.Parameter(torch.tensor([[0.0, -mem_bias]] * 3))

    def forward(self, v, ns: int, head: int):
        lv = torch.logit(v.clamp(1e-3, 1 - 1e-3))
        return self.a[ns, head] * lv * (v > 0) + self.b[ns, head] * (v > 0)


class TraceDecoder(nn.Module):
    def __init__(self, t5_name: str, terms, aspect: np.ndarray, d_latent: int = 512, term_init_ids=None,
                 copy_bias: bool = True, homology: bool = False):
        super().__init__()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(t5_name)
        n0 = len(self.tok)
        self.homology = homology
        self.tok.add_tokens(special_tokens(terms, homology=homology), special_tokens=False)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.t5.resize_token_embeddings(len(self.tok), mean_resizing=False)
        d = self.t5.config.d_model
        self.go0 = self.tok.convert_tokens_to_ids(terms[0])
        assert self.tok.convert_tokens_to_ids(terms[-1]) == self.go0 + len(terms) - 1
        self.nb_id = self.tok.convert_tokens_to_ids("<nb>")
        self.sim0 = self.tok.convert_tokens_to_ids("<sim_0>")
        self.tax0 = self.tok.convert_tokens_to_ids(TAX_TOKENS[0])
        self.hom0 = self.tok.convert_tokens_to_ids(HOM_TOKENS[0]) if homology else self.nb_id
        self.n_terms = len(terms)
        emb = self.t5.get_input_embeddings().weight
        head = self.t5.get_output_embeddings().weight
        with torch.no_grad():
            norm_e, norm_h = emb[:n0].norm(dim=1).mean(), head[:n0].norm(dim=1).mean()
            n_new = len(self.tok) - n0
            emb[n0:] = torch.randn(n_new, d) * norm_e / d ** 0.5
            head[n0:] = torch.randn(n_new, d) * norm_h / d ** 0.5
            if term_init_ids is not None:
                # definition-text init: mean subword embedding of "name: definition" (name tokens counted twice)
                for j, ids in enumerate(term_init_ids):
                    ids = torch.as_tensor(ids, dtype=torch.long)
                    emb[self.go0 + j] = emb[ids].mean(0)
                    head[self.go0 + j] = head[ids].mean(0)
        self.soft = nn.Sequential(nn.Linear(d_latent, d), nn.LayerNorm(d))
        self.soft_scale = nn.Parameter(norm_e.detach().clone() / d ** 0.5)
        self.mem_proj = nn.Linear(d, d)
        nn.init.zeros_(self.mem_proj.weight)  # zero-init residual on top of the copy prior
        nn.init.zeros_(self.mem_proj.bias)
        self.mem_emb = nn.Parameter(head[self.go0:self.go0 + len(terms)].detach().clone())
        mb = -6.0 if copy_bias else -3.0
        self.mem_bias = nn.Parameter(torch.full((len(terms),), mb))
        self.copy = CopyBias(mb) if copy_bias else None
        self.ns_cols = [torch.as_tensor(np.where(aspect == a)[0]) for a in "FPC"]

    # ---- encoder side ----
    def encode(self, Z, src_ids, src_mask):
        """Z [B, M, d_latent] (or None), src_ids [B, L] -> encoder states and mask."""
        e = self.t5.get_input_embeddings()(src_ids)
        if Z is not None:
            s = self.soft(Z) * self.soft_scale
            e = torch.cat([s, e], 1)
            src_mask = torch.cat([torch.ones(Z.shape[:2], dtype=src_mask.dtype, device=src_mask.device), src_mask], 1)
        enc = self.t5.encoder(inputs_embeds=e, attention_mask=src_mask)
        return enc.last_hidden_state, src_mask

    def decode(self, dec_ids, enc_h, enc_mask, past=None, output_attentions=False):
        out = self.t5.decoder(input_ids=dec_ids, encoder_hidden_states=enc_h, encoder_attention_mask=enc_mask,
                              past_key_values=past, use_cache=past is not None, output_attentions=output_attentions,
                              return_dict=True)
        return out

    # ---- heads (namespace-local term index) ----
    def next_logits(self, h, ns: int, vote=None):
        cols = self.ns_cols[ns].to(h.device)
        W = self.t5.get_output_embeddings().weight
        lg = h @ W[self.go0 + cols].T
        eos = h @ W[self.tok.eos_token_id]
        if self.copy is not None and vote is not None:
            lg = lg + self.copy(vote, ns, 0)
        return lg, eos

    def member_logits(self, h, ns: int, vote=None):
        cols = self.ns_cols[ns].to(h.device)
        lg = self.mem_proj(h) @ self.mem_emb[cols].T / (h.shape[-1] ** 0.5) + self.mem_bias[cols]
        if self.copy is not None and vote is not None:
            lg = lg + self.copy(vote, ns, 1)
        return lg


def random_topological(terms_local: np.ndarray, parents: dict, rng: np.random.Generator, shuffle: bool = True):
    """Uniform-choice Kahn order of an ancestor-closed term set (local ids). parents: {t: in-vocab ancestors}.
    shuffle=False and no hierarchy is handled by the caller."""
    s = set(terms_local.tolist())
    need = {t: len(parents.get(t, ())) for t in s}
    kids: dict = {}
    for t in s:
        for p in parents.get(t, ()):
            kids.setdefault(p, []).append(t)
    avail = [t for t in s if need[t] == 0]
    out = []
    while avail:
        i = int(rng.integers(len(avail)))
        t = avail[i]
        avail[i] = avail[-1]
        avail.pop()
        out.append(t)
        for c in kids.get(t, ()):
            need[c] -= 1
            if need[c] == 0:
                avail.append(c)
    return out


def set_targets(tgt: torch.Tensor, Ymh: torch.Tensor, H: Hierarchy | None, truncated: torch.Tensor):
    """Expected-linearisation targets.
    tgt [B, T] local term ids (-1 pad); Ymh [B, V] true set (bool).
    Returns, for decoder positions s = 0..T (state before emitting tgt[:, s]):
      S     [B, T+1, V] emitted-before-s indicator
      legal [B, T+1, V] legal next terms (frontier, or all not-yet-emitted terms without hierarchy)
      avail [B, T+1, V] available true terms = legal & Y (uniform target)
      eos   [B, T+1]    </s> is the target (no available true term and the sequence was not truncated)
      valid [B, T+1]    position carries a loss"""
    B, T = tgt.shape
    V = Ymh.shape[1]
    dev = tgt.device
    oh = torch.zeros(B, T + 1, V, dtype=torch.float16, device=dev)
    m = tgt >= 0
    bi, si = torch.where(m)
    oh[bi, si + 1, tgt[bi, si]] = 1
    S = oh.cumsum(1) > 0.5
    if H is not None:
        dsc = torch.zeros(B, T + 1, V, dtype=torch.float16, device=dev)
        dsc[bi, si + 1] = H.desc[tgt[bi, si]]
        cnt = dsc.float().cumsum(1)
        legal = H.frontier(S, cnt)
        del dsc, cnt
    else:
        legal = ~S
    avail = legal & Ymh[:, None, :]
    lens = m.sum(1)
    pos = torch.arange(T + 1, device=dev)[None]
    valid = pos <= lens[:, None]
    has = avail.any(-1)
    eos = (~has) & valid & (~truncated[:, None] | (pos < lens[:, None]))
    valid = valid & (has | eos)
    return S, legal, avail, eos, valid
