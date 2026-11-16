"""Residue-trace evaluation against UniProt active / binding sites (Phase C1).

A term's residue profile is the attention-rollout of the resampler's last-block latent→residue maps, weighted by a
sharp softmax over the term-head cosine of each latent to the term embedding. Metrics: top-k precision / recall of
the profile against annotated site positions (k = #sites and k = 10), vs random, vs an amino-acid prior, and vs the
collapsed fold-0 adapter.

  from tracego.analysis.trace_eval import residue_profile, score_sites, aa_prior
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from tracego.analysis.uniprot_sites import LIGAND_GO, CATALYTIC, METAL, METALS, site_sets


def residue_profile(Z, att, term_emb, cls_proj, term_idx: int, temp: float = 0.1) -> np.ndarray:
    """[L] normalised residue mass for one term. Z [M,d], att [M,L] (last block), term_emb [V,d], cls_proj module."""
    with torch.no_grad():
        q = F.normalize(cls_proj(Z[None]), dim=-1)[0]                       # [M,d]
        e = F.normalize(term_emb[term_idx], dim=-1)
        w = (q @ e / temp).softmax(0)                                       # [M]
        res = (w[:, None] * att).sum(0)
        res = res / res.sum().clamp(min=1e-9)
    return res.float().cpu().numpy()


def topk_hits(profile: np.ndarray, sites: list[int], k: int) -> dict:
    """Precision / recall of the top-k residues of `profile` against the site set."""
    sites = [s for s in sites if 0 <= s < len(profile)]
    if not sites or k <= 0:
        return {"prec": 0.0, "rec": 0.0, "k": k, "n_sites": len(sites), "hits": 0}
    k = min(k, len(profile))
    top = set(np.argpartition(-profile, k - 1)[:k].tolist())
    hits = len(top & set(sites))
    return {"prec": hits / k, "rec": hits / len(sites), "k": k, "n_sites": len(sites), "hits": hits}


def aa_prior(seq: str, site_aa_freq: dict) -> np.ndarray:
    """Residue scores from how often each amino acid appears in annotated sites (train prior)."""
    p = np.array([site_aa_freq.get(a, 1e-6) for a in seq], np.float64)
    return p / p.sum()


def build_aa_prior(sites_json: dict) -> dict:
    c = Counter()
    for e in sites_json.values():
        ss = site_sets(e, len(e["seq"]))
        for i in ss.get("functional", []):
            c[e["seq"][i]] += 1
    tot = sum(c.values()) or 1
    return {a: n / tot for a, n in c.items()}


def term_for_sites(entry: dict, label: str, terms_set: set) -> str | None:
    """Pick an MF term to evaluate against a site label ('act', 'bind', 'functional', 'lig:X')."""
    if label == "act":
        return CATALYTIC if CATALYTIC in terms_set else None
    if label.startswith("lig:"):
        lig = label[4:]
        g = LIGAND_GO.get(lig)
        if g and g in terms_set:
            return g
        if lig in METALS and METAL in terms_set:
            return METAL
        return None
    if label in ("bind", "functional"):
        # most specific ligand term present, else metal, else catalytic for act-only proteins
        for typ, lo, hi, lig, _ in entry["features"]:
            if typ != "Binding site" or not lig:
                continue
            g = LIGAND_GO.get(lig)
            if g and g in terms_set:
                return g
        if any(f[3] in METALS for f in entry["features"]):
            return METAL if METAL in terms_set else None
        if any(f[0] == "Active site" for f in entry["features"]):
            return CATALYTIC if CATALYTIC in terms_set else None
    return None


def summarise(rows: list[dict]) -> dict:
    """Mean precision / recall over a list of per-(protein,term) metric dicts; also mean of random baselines."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k.endswith("_prec") or k.endswith("_rec")]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["n"] = len(rows)
    return out
