"""Retrieval in the (adapted) embedding space: cosine top-k neighbours and the similarity-weighted GO vote.

No sequence-alignment tool is involved. Same-accession references are excluded, as in the repo's kNN.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def topk_cosine(Q: np.ndarray, R: np.ndarray, k: int, q_ids=None, r_ids=None, exclude_rows=None,
                device="cuda", chunk: int = 4096, exclude: sp.csr_matrix | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Q [nq, d], R [nr, d] (any float) -> (idx [nq, k] into R, sim [nq, k] float32), sorted by similarity.
    exclude_rows: optional [nq] array with a row of R to drop per query (self), or None. Same accession ids
    (q_ids[i] == r_ids[j]) are dropped as well. exclude: optional CSR [nq, nr] of (query, reference) pairs to drop
    (hidden references of the TV-remote protocol and of the homology curriculum)."""
    Rt = torch.nn.functional.normalize(torch.as_tensor(R, dtype=torch.float16, device=device).float(), dim=1).half()
    rid_hash = None
    if q_ids is not None and r_ids is not None:
        codes = {a: i for i, a in enumerate(np.unique(np.concatenate([q_ids, r_ids])))}
        rid_hash = torch.as_tensor(np.array([codes[a] for a in r_ids]), device=device)
        qid_hash = np.array([codes[a] for a in q_ids])
    I = np.zeros((len(Q), k), np.int64)
    S = np.zeros((len(Q), k), np.float32)
    kk = k + 4
    for s in range(0, len(Q), chunk):
        q = torch.nn.functional.normalize(torch.as_tensor(Q[s:s + chunk], device=device).float(), dim=1).half()
        sim = (q @ Rt.T).float()
        if rid_hash is not None:
            qh = torch.as_tensor(qid_hash[s:s + chunk], device=device)
            sim[qh[:, None] == rid_hash[None, :]] = -2
        if exclude_rows is not None:
            ex = torch.as_tensor(exclude_rows[s:s + chunk], device=device)
            ok = ex >= 0
            sim[torch.arange(len(ex), device=device)[ok], ex[ok]] = -2
        if exclude is not None:
            sub = exclude[s:s + chunk].tocoo()
            if sub.nnz:
                sim[torch.as_tensor(sub.row, device=device), torch.as_tensor(sub.col, device=device)] = -2
        v, i = sim.topk(kk, dim=1)
        I[s:s + chunk] = i[:, :k].cpu().numpy()
        S[s:s + chunk] = v[:, :k].cpu().numpy()
    return I, S


class ResidueCodes:
    """PCA-compressed, L2-normalised residue states of one encoder layer for every protein, as one flat fp16
    array with offsets; the late-interaction re-ranker reads them from GPU memory."""

    def __init__(self, path, device="cuda", seqs=None):
        z = np.load(path)
        self.off = z["off"]
        rows = z["rows"]
        self.pos = {int(r): i for i, r in enumerate(rows)}
        self.X = torch.as_tensor(z["X"], device=device)
        self.device = device
        self.aa = None
        if seqs is not None:
            # amino-acid track aligned with the codes (same truncation), for identity along the embedding alignment
            lut = np.zeros(256, np.int8)
            for k, c in enumerate("ACDEFGHIKLMNPQRSTVWY"):
                lut[ord(c)] = k + 1
            flat = np.zeros(int(self.off[-1]), np.int8)
            for i, r in enumerate(rows):
                n = self.off[i + 1] - self.off[i]
                s = np.frombuffer(seqs[r].encode(), np.uint8)[:n]
                flat[self.off[i]:self.off[i] + len(s)] = lut[s]
            self.aa = torch.as_tensor(flat, device=device)

    def get(self, row: int) -> torch.Tensor:
        i = self.pos[int(row)]
        return self.X[self.off[i]:self.off[i + 1]]

    def get_aa(self, row: int) -> torch.Tensor:
        i = self.pos[int(row)]
        return self.aa[self.off[i]:self.off[i + 1]]

    def padded(self, rows) -> tuple[torch.Tensor, torch.Tensor]:
        seg = [self.get(r) for r in rows]
        Lm = max(len(s) for s in seg)
        C = torch.zeros(len(seg), Lm, self.X.shape[1], dtype=self.X.dtype, device=self.device)
        m = torch.zeros(len(seg), Lm, dtype=torch.bool, device=self.device)
        for i, s in enumerate(seg):
            C[i, :len(s)] = s
            m[i, :len(s)] = True
        return C, m

    def padded_aa(self, rows) -> torch.Tensor:
        seg = [self.get_aa(r) for r in rows]
        A = torch.full((len(seg), max(len(s) for s in seg)), -1, dtype=self.aa.dtype, device=self.device)
        for i, s in enumerate(seg):
            A[i, :len(s)] = s
        return A


def chain_identity(sim: torch.Tensor, m: torch.Tensor, tau: float = 0.9, aq=None, A=None,
                   side_mode: str = "max") -> torch.Tensor:
    """Embedding-space identity estimate from a residue similarity tensor sim [c, Lq, Lc] (candidate mask m [c, Lc]):
    the fraction of residues whose best match exceeds tau AND lies on a diagonal run with a sequence neighbour's
    best match (j*(i+1) = j*(i) + 1), i.e. residues an ungapped local alignment would call identical. With the
    amino-acid tracks (aq [Lq], A [c, Lc]) a residue must also carry the same amino acid as its embedding match, so
    the score reads percent identity along the alignment the embeddings induce. side_mode "max": computed in both
    directions and maxed so a fragment on either side is not penalised; "q": normalised over the query length only
    (a short reference fragment no longer counts as identical to a full-length query), with the reference side as a
    1e-3 tie-break."""
    def side(v, j, valid, eq):
        ch = torch.zeros_like(v, dtype=torch.bool)
        run = j[:, 1:] == j[:, :-1] + 1
        ch[:, 1:] |= run
        ch[:, :-1] |= run
        ok = (v > tau) & ch & valid
        if eq is not None:
            ok &= eq
        return ok.float().sum(1) / valid.float().sum(1)
    vq, jq = sim.max(2)
    vc, jc = sim.max(1)
    eq_q = eq_c = None
    if aq is not None:
        eq_q = A.gather(1, jq) == aq[None]
        eq_c = aq[jc] == A
    a, c = side(vq, jq, torch.ones_like(vq, dtype=torch.bool), eq_q), side(vc, jc, m, eq_c)
    if side_mode == "q":
        return a + 1e-3 * c
    return torch.maximum(a, c)


@torch.no_grad()
def maxsim_rerank(codes: ResidueCodes, q_rows: np.ndarray, cand_rows: np.ndarray, mode: str = "q2c"):
    """Late-interaction (ColBERT MaxSim) scores between each query and its candidates.
    q2c = mean over query residues of the best-matching candidate residue; c2q the reverse; sym their mean;
    max their maximum (a local-alignment-like score that does not penalise a fragment on either side)."""
    out = np.zeros(cand_rows.shape, np.float32)
    for i, q in enumerate(q_rows):
        x = codes.get(q)
        C, m = codes.padded(cand_rows[i])
        sim = torch.einsum("id,cjd->cij", x, C).float().masked_fill(~m[:, None, :], -1)
        if mode.startswith("chainaa"):
            side_mode = "q" if mode.startswith("chainaaq") else "max"
            out[i] = chain_identity(sim, m, float(mode[7 + (side_mode == "q"):] or 0.0), codes.get_aa(q),
                                    codes.padded_aa(cand_rows[i]), side_mode).cpu().numpy()
            continue
        if mode.startswith("chain"):
            out[i] = chain_identity(sim, m, float(mode[5:] or 0.9)).cpu().numpy()
            continue
        a = sim.max(2).values.mean(1)
        if mode == "q2c":
            s = a
        else:
            b = (sim.max(1).values * m).sum(1) / m.sum(1)
            s = {"c2q": b, "sym": (a + b) / 2, "max": torch.maximum(a, b)}[mode]
        out[i] = s.cpu().numpy()
    return out


def whiten(X: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """PCA whitening fit on the reference rows (removes the anisotropy of mean-pooled ESM states)."""
    X = X.astype(np.float32)
    mu = X[fit_rows].mean(0)
    w, V = np.linalg.eigh(np.cov((X[fit_rows] - mu).T))
    return ((X - mu) @ (V / np.sqrt(w + 1e-6 * w.max()))).astype(np.float32)


def vote(I: np.ndarray, S: np.ndarray, L: sp.csr_matrix, T: float, k: int, aspect_cols: list | None = None,
         roots: list | None = None, floor: float = 0.0, ns_has=None) -> sp.csr_matrix:
    """Similarity-sharpened neighbour vote over the columns of L (references x terms):
        w_j = exp((s_j - max_i s_i) / T) * max(s_j - floor, 0),   p_t = sum_j w_j y_jt / sum_j w_j,
    normalised per aspect over the neighbours annotated in that aspect when `roots` (one root column per aspect)
    and `aspect_cols` (namespace id per column) are given. The cosine plays the role of the bitscore and the
    exponent the identity sharpening of the repo's T5_k50_g0_na kNN."""
    I, S = I[:, :k], S[:, :k].astype(np.float64)
    w = np.exp((S - S[:, :1]) / T) * np.maximum(S - floor, 1e-6)
    nq = len(I)
    W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(nq), k), I.ravel())), shape=(nq, L.shape[0]))
    P = (W @ L).tocsr()
    rows = np.repeat(np.arange(nq), np.diff(P.indptr))
    if roots is not None or ns_has is not None:
        # ns_has [n_ref, 3]: reference annotated in namespace k (used when the roots are not label columns)
        D = (W @ (L[:, roots] if ns_has is None else ns_has))
        D = D.toarray() if sp.issparse(D) else np.asarray(D)
        den = D[rows, aspect_cols[P.indices]]
    else:
        den = np.asarray(W.sum(1)).ravel()[rows]
    P.data = np.minimum(P.data / np.maximum(den, 1e-12), 1.0).astype(np.float32)
    P.eliminate_zeros()
    return P
