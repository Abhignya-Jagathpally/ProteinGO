"""Instruction examples for the decoder: one (protein, namespace) pair each.

Source:  [soft prompt Z] "Instruction: list the molecular function GO terms of this protein. Organism: Homo
         sapiens. Evidence:" <nb> <sim_b> <tax_rank> GO:leaf GO:leaf ... <nb> ...
Target:  the protein's GO terms in that namespace in a random topological order (uniform-choice Kahn), then </s>.

Neighbour GO leaves are the most specific in-vocabulary terms of the reference's label set (ancestors are implied
by the hierarchy), ordered by IA. Only reference labels are shown; the query's own labels never enter the source.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from tracego.data.bundle import ASPECT_NAME, RANKS
from tracego.models.decoder import N_SIM_BINS, random_topological


def leaves_by_ns(Y: sp.csr_matrix, anc: np.ndarray, aspect: np.ndarray, ia: np.ndarray, cap: int) -> list:
    """For every row of Y: {ns: [vocab ids of leaf terms, by IA desc, at most cap]}; rows without labels -> {}."""
    V = Y.shape[1]
    A = sp.csr_matrix((np.ones(len(anc), np.float32), (anc[:, 0], anc[:, 1])), shape=(V, V))
    A.setdiag(0)
    A.eliminate_zeros()
    has_desc = (Y @ A.T).tocsr()  # (Y A^T)[i, a] > 0 when some labelled t has a as ancestor... transpose below
    # A[t, a] = 1: a ancestor of t.  Y @ A gives, per row, the ancestors of its terms; a label that is an ancestor of
    # another label is not a leaf.
    anc_of = (Y @ A).tocsr()
    out = []
    ns_of = np.array(["FPC".index(a) for a in aspect])
    for i in range(Y.shape[0]):
        lab = Y.indices[Y.indptr[i]:Y.indptr[i + 1]]
        if len(lab) == 0:
            out.append({})
            continue
        inner = set(anc_of.indices[anc_of.indptr[i]:anc_of.indptr[i + 1]].tolist())
        lv = np.array([t for t in lab if t not in inner], np.int64)
        d = {}
        for k in range(3):
            x = lv[ns_of[lv] == k]
            if len(x):
                d[k] = x[np.argsort(-ia[x], kind="stable")][:cap]
        out.append(d)
    del has_desc
    return out


def sim_edges(S: np.ndarray) -> np.ndarray:
    """Quantile edges of neighbour similarities (training queries) for the <sim_b> tokens."""
    return np.quantile(S.ravel(), np.linspace(0, 1, N_SIM_BINS + 1)[1:-1])


def tax_rank(lq: np.ndarray, ln: np.ndarray) -> np.ndarray:
    """Lowest shared rank index (0 species ... 7 domain, 8 none) between lineages [n, 8] and [n, k, 8]."""
    same = (lq[:, None, :] == ln) & (ln > 0)
    first = np.where(same.any(-1), same.argmax(-1), len(RANKS))
    return first


class InstructionSet:
    """Source token ids for a set of queries in one encoder space; targets built lazily.

    Optional Phase A1 homology tokens (`hom_bin`: -1 = unmasked / full, 0..6 = FR24-P cutoff bins) and Phase B
    auxiliary evidence (`aux_toks`: per-query list of already-encoded token ids inserted after the organism).
    """

    def __init__(self, tok_ids: dict, bundle, q_rows: np.ndarray, I: np.ndarray, S: np.ndarray, ref_leaves: dict,
                 ref_has: np.ndarray, edges: np.ndarray, k: int, evidence: bool, prefix_cache: dict,
                 hom_bin: np.ndarray | None = None, aux_toks: list | None = None):
        self.q_rows = q_rows
        self.src = {}
        lin_q = bundle.lineage(q_rows)
        lin_n = bundle.lineage(I[:, :k].ravel()).reshape(len(q_rows), k, -1) if evidence else None
        tr = tax_rank(lin_q, lin_n) if evidence else None
        sb = np.searchsorted(edges, S[:, :k]) if evidence else None
        # best remaining neighbour similarity as a sim-bin token (curriculum / TV-remote awareness)
        best_sb = np.searchsorted(edges, S[:, 0]) if evidence and hom_bin is not None else None
        org = bundle.prot.organism.fillna("unknown organism").str.split(" (", regex=False).str[0].values[q_rows]
        for ns in range(3):
            asp = ASPECT_NAME["FPC"[ns]]
            seqs = []
            use_hom = hom_bin is not None
            use_aux = aux_toks is not None
            for i in range(len(q_rows)):
                if not use_hom and not use_aux:
                    key = (ns, org[i])
                    if key not in prefix_cache:
                        prefix_cache[key] = tok_ids["encode"](
                            f"Instruction: list the {asp} GO terms of this protein. Organism: {org[i]}. Evidence:")
                    ids = list(prefix_cache[key])
                else:
                    key = (ns, org[i], "base")
                    if key not in prefix_cache:
                        prefix_cache[key] = tok_ids["encode"](
                            f"Instruction: list the {asp} GO terms of this protein. Organism: {org[i]}.")
                    ids = list(prefix_cache[key])
                    if use_hom:
                        # <hom_full> for unmasked (-1); else <hom_no_hit>..<hom_100>
                        ids += [tok_ids["hom0"] + (0 if hom_bin[i] < 0 else int(hom_bin[i]) + 1),
                                tok_ids["sim0"] + int(best_sb[i])]
                    if use_aux and aux_toks[i]:
                        ids += list(aux_toks[i])
                    ids += tok_ids["encode"](" Evidence:")
                if evidence:
                    for j in range(k):
                        nb = I[i, j]
                        lv = ref_leaves.get(nb, {}).get(ns)
                        if lv is None:
                            continue
                        ids += [tok_ids["nb"], tok_ids["sim0"] + int(sb[i, j]), tok_ids["tax0"] + int(tr[i, j])]
                        ids += (tok_ids["go0"] + lv).tolist()
                else:
                    ids += [tok_ids["nb"]]
                ids.append(tok_ids["eos"])
                seqs.append(np.array(ids, np.int32))
            self.src[ns] = seqs


class Collator:
    """Builds padded tensors for a namespace-homogeneous batch."""

    def __init__(self, Zmm, zpos: dict, ns_cols: list, parents: list, max_src: int, max_tgt: int,
                 constrained: bool, seed: int = 0):
        self.Z, self.zpos = Zmm, zpos
        self.ns_cols = [np.asarray(c) for c in ns_cols]
        self.parents = parents
        self.max_src, self.max_tgt = max_src, max_tgt
        self.constrained = constrained
        self.rng = np.random.default_rng(seed)

    def __call__(self, items, ns: int, Y: sp.csr_matrix | None, vote: sp.csr_matrix | None, device):
        """items: list of (InstructionSet, index). Y/vote: rows aligned with items (vocab columns)."""
        B = len(items)
        src = [s.src[ns][i][: self.max_src] for s, i in items]
        L = max(len(x) for x in src)
        ids = np.zeros((B, L), np.int64)
        am = np.zeros((B, L), np.int64)
        for b, x in enumerate(src):
            ids[b, :len(x)] = x
            am[b, :len(x)] = 1
        rows = [s.q_rows[i] for s, i in items]
        Z = None
        if self.Z is not None:
            Z = torch.as_tensor(np.stack([self.Z[self.zpos[r]] for r in rows]).astype(np.float32))
        cols = self.ns_cols[ns]
        out = {"src": torch.as_tensor(ids), "src_mask": torch.as_tensor(am), "Z": Z, "rows": np.array(rows)}
        if vote is not None:
            out["vote"] = torch.as_tensor(vote[:, cols].toarray())
        if Y is not None:
            Yn = Y[:, cols].tocsr()
            out["Y"] = torch.as_tensor(Yn.toarray() > 0)
            tg = np.full((B, self.max_tgt), -1, np.int64)
            trunc = np.zeros(B, bool)
            for b in range(B):
                lab = Yn.indices[Yn.indptr[b]:Yn.indptr[b + 1]]
                if self.constrained:
                    order = random_topological(lab, self.parents[ns], self.rng)
                else:
                    order = self.rng.permutation(lab).tolist()
                trunc[b] = len(order) > self.max_tgt
                order = order[: self.max_tgt]
                tg[b, :len(order)] = order
            T = max(1, int((tg >= 0).sum(1).max()))
            out["tgt"] = torch.as_tensor(tg[:, :T])
            out["truncated"] = torch.as_tensor(trunc)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
