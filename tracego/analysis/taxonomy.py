"""Phase D2: protein evolution across species. Taxonomic distance between a TV v2 query and its annotated
neighbours, and retrieval with the query's own clade removed from the reference set.

  lca_group           lowest NCBI rank shared by two proteins (bundle lineages; same taxon id = "species")
  clade_keys          nested clade ids (taxon, species, genus, family) used to drop same-clade references
  topk_cosine_clade   `retrieval.topk_cosine` with same-clade references masked on the GPU (exact top-k after
                      removal, so no over-fetching is needed)
  rerank_union        chain-identity MaxSim re-rank of the union of several candidate lists of a query, one pass
  fold_neighbours / ensemble_vote   build_evidence.py + knn_baseline.py on the masked candidates

The decoder is not re-run here: TraceGO's TV readout is anchored to the vote, so leave-clade-out is measured on the
retrieval vote only. Driver: `python scripts/analysis_d2_taxonomy.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tracego.data.bundle import RANKS
from tracego.models.retrieval import maxsim_rerank, vote

GROUPS = (*RANKS, "none", "unmapped")
# coarse LCA groups for the 2-D tables (small deep-rank groups pooled)
COARSE = {"species": "species", "genus": "genus", "family": "family", "order": "order-class",
          "class": "order-class", "phylum": "phylum+", "kingdom": "phylum+", "superkingdom": "phylum+",
          "none": "phylum+", "unmapped": "unmapped"}
COARSE_ORDER = ("species", "genus", "family", "order-class", "phylum+", "unmapped")
ID4 = ("<50", "50-80", "80-95", "95-100")
CONDITIONS = {"none": 0, "species": 2, "genus": 3, "family": 4}  # number of nested clade levels removed


def id4_of(ident: np.ndarray) -> np.ndarray:
    x = np.asarray(ident, float)
    return np.select([x < 50, x < 80, x < 95], list(ID4[:3]), ID4[3])


def lca_group(b, q_rows: np.ndarray, n_rows: np.ndarray) -> np.ndarray:
    """Lowest shared rank of (query, neighbour) pairs. "species" also covers an identical taxon id whose species
    rank is missing; "none" = no shared rank (different domains, or the shared ranks missing on one side);
    "unmapped" = either protein has no NCBI taxon / lineage."""
    tq, tn = b.prot.ncbi_taxon.values[q_rows], b.prot.ncbi_taxon.values[n_rows]
    Lq, Ln = b.lineage(q_rows), b.lineage(n_rows)
    eq = (Lq == Ln) & (Lq > 0)
    k = np.where(eq.any(1), eq.argmax(1), len(RANKS))
    g = np.array(GROUPS)[k]
    g[(tq == tn) & (tq > 0)] = "species"
    g[(tq == 0) | (tn == 0) | ~Lq.any(1) | ~Ln.any(1)] = "unmapped"
    return g


def missing_ranks(b, q_rows: np.ndarray, n_rows: np.ndarray) -> dict:
    """Per rank: pairs with the rank missing on the query side, on the neighbour side, and on either."""
    Lq, Ln = b.lineage(q_rows), b.lineage(n_rows)
    return {r: {"query": int((Lq[:, k] == 0).sum()), "neighbour": int((Ln[:, k] == 0).sum()),
                "either": int(((Lq[:, k] == 0) | (Ln[:, k] == 0)).sum())} for k, r in enumerate(RANKS)}


def clade_keys(b, rows: np.ndarray) -> np.ndarray:
    """[4, n] nested clade ids per protein: raw taxon, species, genus, family (0 = missing, never matched)."""
    L = b.lineage(rows)
    return np.stack([b.prot.ncbi_taxon.values[rows], L[:, 0], L[:, 1], L[:, 2]]).astype(np.int64)


@torch.no_grad()
def topk_cosine_clade(Q, R, k, q_ids, r_ids, q_keys, r_keys, device="cuda", chunk=2048):
    """`topk_cosine` (fp16 cosine, same-accession references dropped) with every reference that shares a non-zero
    clade id with the query on any of the given levels (q_keys [l, nq], r_keys [l, nr]; l = 0 -> nothing) dropped."""
    Rt = torch.nn.functional.normalize(torch.as_tensor(R, dtype=torch.float16, device=device).float(), dim=1).half()
    codes = {a: i for i, a in enumerate(np.unique(np.concatenate([q_ids, r_ids])))}
    rid = torch.as_tensor(np.array([codes[a] for a in r_ids]), device=device)
    qid = np.array([codes[a] for a in q_ids])
    rk = torch.as_tensor(r_keys, device=device)
    I = np.zeros((len(Q), k), np.int64)
    S = np.zeros((len(Q), k), np.float32)
    n_drop = np.zeros(len(Q), np.int64)
    for s in range(0, len(Q), chunk):
        q = torch.nn.functional.normalize(torch.as_tensor(Q[s:s + chunk], device=device).float(), dim=1).half()
        sim = (q @ Rt.T).float()
        sim[torch.as_tensor(qid[s:s + chunk], device=device)[:, None] == rid[None, :]] = -2
        drop = torch.zeros_like(sim, dtype=torch.bool)
        for lv in range(len(q_keys)):
            qk = torch.as_tensor(q_keys[lv, s:s + chunk], device=device)[:, None]
            drop |= (qk == rk[lv][None, :]) & (qk > 0)
        sim[drop] = -2
        n_drop[s:s + chunk] = drop.sum(1).cpu().numpy()
        v, i = sim.topk(k + 4, dim=1)
        I[s:s + chunk] = i[:, :k].cpu().numpy()
        S[s:s + chunk] = v[:, :k].cpu().numpy()
    return I, S, n_drop


@torch.no_grad()
def rerank_union(codes, q_rows: np.ndarray, cands: list, mode: str = "chainaa0.0", width: int = 200,
                 log_fn=None) -> tuple[list, np.ndarray]:
    """Re-rank scores for several candidate lists ([nq, c] reference rows each) of the same queries; each distinct
    (query, reference) pair is scored once (the chain-identity score does not depend on the batch it is in)."""
    out = [np.zeros(c.shape, np.float32) for c in cands]
    sizes = np.zeros(len(q_rows), np.int64)
    for i, q in enumerate(q_rows):
        u, inv = np.unique(np.concatenate([c[i] for c in cands]), return_inverse=True)
        e = np.zeros(len(u), np.float32)
        for s in range(0, len(u), width):
            e[s:s + width] = maxsim_rerank(codes, np.array([q]), u[None, s:s + width], mode)[0]
        o = 0
        for c, r in zip(cands, out):
            r[i] = e[inv[o:o + c.shape[1]]]
            o += c.shape[1]
        sizes[i] = len(u)
        if log_fn is not None and i % 1000 == 0:
            log_fn(f"re-rank {i:,}/{len(q_rows):,}: union of {len(cands)} lists = {len(u)} candidates; "
                   f"top identity {e.max():.3f}")
    return out, sizes


def keep_top(cand: np.ndarray, score: np.ndarray, k_keep: int):
    o = np.argsort(-score, 1, kind="stable")[:, :k_keep]
    return np.take_along_axis(cand, o, 1), np.take_along_axis(score, o, 1)


def ensemble_vote(lists: list, Yo, o, T: float, k_vote: int):
    """knn_baseline.py: full-ontology vote per cross-fitting space, aspect-normalised, averaged over the spaces."""
    roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    P = None
    for I, S in lists:
        Pf = vote(I, S, Yo, T, k_vote, aspect_cols=o.ns, roots=roots)
        P = Pf if P is None else P + Pf
    return (P / len(lists)).tocsr()


# ---------------------------------------------------------------- scoring tables
def per_protein_table(ev, cur: dict, per_protein_f) -> pd.Series:
    """Per-protein F (benchmark Fmax threshold of each namespace) averaged over the namespaces the protein has
    ground truth in."""
    f = pd.concat([per_protein_f(ev, cur, a).set_index("id").f.rename(a) for a in ("MF", "BP", "CC")], axis=1)
    return f.mean(1)


def group_scores(ev, cur: dict, ids: np.ndarray, labels: np.ndarray, order, ppf: pd.Series | None = None) -> list:
    rows = []
    for g in order:
        sel = ids[labels == g]
        if not len(sel):
            continue
        f = ev.fmax(cur, subset=list(sel))
        r = {"group": g, "n": int(len(sel)), **{a: round(f[a], 4) for a in ("MF", "BP", "CC", "mean")}}
        if ppf is not None:
            r["ppF"] = round(float(ppf.reindex(sel).mean()), 4)
        rows.append(r)
    return rows


def grid(values: pd.Series, a: np.ndarray, b_: np.ndarray, order_a, order_b, stat: str = "mean") -> pd.DataFrame:
    """2-D table of a per-protein statistic (and counts) over two labelings."""
    df = pd.DataFrame({"a": a, "b": b_, "v": values.values})
    m = df.pivot_table(index="a", columns="b", values="v", aggfunc=stat).reindex(index=order_a, columns=order_b)
    n = df.pivot_table(index="a", columns="b", values="v", aggfunc="size").reindex(index=order_a, columns=order_b)
    return m.round(4), n.fillna(0).astype(int)
