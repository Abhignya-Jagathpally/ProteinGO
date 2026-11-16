"""TV-remote: TV v2 queries with their close references hidden, so the identity mix matches fresh24 FR24-P.

Every TV v2 protein draws one identity cutoff c and every reference whose identity to it exceeds c is hidden. The
identity u(q, r) of a pair is DIAMOND's percent identity when the repo's TV hit list (`diamond_hits_tv.parquet`,
top 50 by bitscore) contains the pair, otherwise the embedding-alignment identity of the re-ranker (chain+aa on the
layer-8 residue codes; it tracks DIAMOND identity above ~0.4). DIAMOND is read for the protocol and the bins only,
never as model input.

The cutoffs are the upper edges of the descriptive identity bins; a protein can take a bin when hiding at that bin's
edge leaves its best remaining identity inside the bin. Bins are assigned by iterative proportional fitting over
each protein's achievable bins so the realised mix matches FR24-P (seeded, saved once).

Retrieval under the mask (`apply`): hidden DIAMOND-listed references are excluded before the cosine search, and any
re-ranked candidate whose embedding identity exceeds the cutoff (and is not a DIAMOND-listed pair below it) is
dropped, so no encoder or candidate channel can leak a hidden reference.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from tracego import paths

MASK = paths.RUNS / "phaseA" / "tv_remote.npz"
BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]
FR24P_COUNTS = {"no_hit": 2270, "<30": 415, "30-50": 2633, "50-80": 3036, "80-95": 1527, "95-99.9": 897, "100": 453}
# hide u > CUT[bin] (identity as a fraction); no_hit hides every DIAMOND-listed pair and embedding identity > NOHIT_EMB
CUT = {"no_hit": 0.0, "<30": 0.30, "30-50": 0.50, "50-80": 0.80, "80-95": 0.95, "95-99.9": 0.9995, "100": 9.0}
NOHIT_EMB = 0.25     # 90th percentile of the embedding identity of non-DIAMOND-hit candidates on TV v2 is 0.27
HIT_EMB = 0.30       # an unlisted reference counts as a hit for binning above this embedding identity


def emb_cut(cut: np.ndarray) -> np.ndarray:
    """Embedding-identity threshold above which an unlisted candidate is hidden."""
    return np.where(cut <= 0, NOHIT_EMB, cut)


def bin_index(pid: np.ndarray) -> np.ndarray:
    """Percent identity (0 = no hit) -> index into BINS."""
    x = np.asarray(pid, float)
    return np.select([x <= 0, x < 30, x < 50, x < 80, x < 95, x < 100], [0, 1, 2, 3, 4, 5], 6)


def ipf_assign(achievable: np.ndarray, target: np.ndarray, rng: np.random.Generator, iters: int = 200):
    """achievable [n, B] bool, target [B] probabilities -> bin per protein whose expected mix matches target as
    closely as the achievable sets allow (multiplicative bin weights, then one seeded draw per protein)."""
    w = np.ones(len(target))
    for _ in range(iters):
        P = achievable * (w * target)[None]
        P /= P.sum(1, keepdims=True)
        cur = P.mean(0)
        w *= np.where(cur > 0, target / np.maximum(cur, 1e-12), 1.0)
    P = achievable * (w * target)[None]
    P /= P.sum(1, keepdims=True)
    u = rng.random(len(P))[:, None]
    return (P.cumsum(1) < u).sum(1).clip(max=len(target) - 1), P.mean(0)


class RemoteMask:
    """Saved TV-remote mask: per query row its cutoff, hidden DIAMOND-listed references, DIAMOND-listed references
    kept (allowed even when their embedding identity is above the cutoff) and its realised identity bin."""

    def __init__(self, path=MASK):
        z = np.load(path, allow_pickle=False)
        self.q = z["q"]
        self.cut = z["cut"]
        # npz stores bin names; expose both the name array and a 0..6 index for curriculum / reporting
        self.bin_name = z["bin"].astype(str)
        self.bin = np.array([BINS.index(x) for x in self.bin_name], np.int8)
        self.identity = z["identity"]
        self.n = len(self.q)
        self.pos = {int(r): i for i, r in enumerate(self.q)}
        self.hid = [z["hid_rows"][z["hid_ptr"][i]:z["hid_ptr"][i + 1]] for i in range(self.n)]
        self.allow = [set(z["allow_rows"][z["allow_ptr"][i]:z["allow_ptr"][i + 1]].tolist()) for i in range(self.n)]

    def exclusion(self, q_rows: np.ndarray, ref_rows: np.ndarray) -> sp.csr_matrix:
        """CSR [len(q_rows), len(ref_rows)] of hidden (query, reference) pairs."""
        mx = max([int(ref_rows.max())] + [int(h.max()) for h in self.hid if len(h)])
        rpos = -np.ones(mx + 1, np.int64)
        rpos[ref_rows] = np.arange(len(ref_rows))
        r, c = [], []
        for i, q in enumerate(q_rows):
            h = rpos[self.hid[self.pos[int(q)]]]
            h = h[h >= 0]
            r.append(np.full(len(h), i))
            c.append(h)
        r, c = np.concatenate(r), np.concatenate(c)
        return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(q_rows), len(ref_rows)))

    def drop(self, q_rows: np.ndarray, cand_rows: np.ndarray, emb_id: np.ndarray) -> np.ndarray:
        """[nq, C] bool: re-ranked candidates hidden by the embedding-identity rule."""
        idx = np.array([self.pos[int(q)] for q in q_rows])
        over = emb_id > emb_cut(self.cut[idx])[:, None]
        for i in np.where(over.any(1))[0]:
            al = self.allow[idx[i]]
            if al:
                over[i] &= ~np.isin(cand_rows[i], list(al))
        return over


class CurriculumMask:
    """Homology-curriculum mask for training queries (Phase A1): each query draws an identity cutoff from the
    FR24-P bin mix (plus a share of fully unmasked examples); references whose embedding-alignment identity exceeds
    the cutoff are dropped. No DIAMOND list is used (training pairs are not listed), so exclusion() is empty and
    drop() is the whole protocol."""

    def __init__(self, q_rows: np.ndarray, seed: int = 0, unmasked: float = 0.25):
        rng = np.random.default_rng(seed)
        n = len(q_rows)
        # probability mass: unmasked share + FR24-P proportions on the remainder
        counts = np.array([FR24P_COUNTS[b] for b in BINS], float)
        p = counts / counts.sum()
        p = np.concatenate([[unmasked], (1.0 - unmasked) * p])  # index 0 = unmasked
        draw = rng.choice(len(p), size=n, p=p)
        self.q = np.asarray(q_rows)
        self.pos = {int(r): i for i, r in enumerate(self.q)}
        self.unmasked = draw == 0
        # bin index into BINS for masked examples; -1 for unmasked
        self.bin = np.where(self.unmasked, -1, draw - 1).astype(np.int8)
        self.cut = np.where(self.unmasked, 9.0, np.array([CUT[BINS[i]] for i in np.maximum(draw - 1, 0)])).astype(
            np.float32)
        self.n = n
        self.allow = [set() for _ in range(n)]
        log_bins = {("unmasked" if i < 0 else BINS[i]): int((self.bin == i).sum()) for i in range(-1, len(BINS))}
        self._log_bins = log_bins

    def exclusion(self, q_rows: np.ndarray, ref_rows: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((len(q_rows), len(ref_rows)), dtype=np.float32)

    def drop(self, q_rows: np.ndarray, cand_rows: np.ndarray, emb_id: np.ndarray) -> np.ndarray:
        idx = np.array([self.pos[int(q)] for q in q_rows])
        return emb_id > emb_cut(self.cut[idx])[:, None]


def masked_search(mask, q_rows, ref_rows, cosine, rerank, k_keep: int, sentinel: int, rounds: int = 4,
                  log=print):
    """Neighbours under the mask. cosine(q_sub_idx, exclude CSR) -> candidate indices into ref_rows; rerank(q_rows,
    cand_rows) -> embedding identity. Hidden candidates are dropped and excluded from the next round, which searches
    deeper for queries left with fewer than k_keep visible neighbours. Unfilled slots get `sentinel` (a row without
    labels) and score 0, so they carry neither vote weight nor evidence tokens."""
    nq = len(q_rows)
    excl = mask.exclusion(q_rows, ref_rows)
    keep = [dict() for _ in range(nq)]
    todo = np.arange(nq)
    for rd in range(rounds):
        I = cosine(todo, excl[todo])
        cand = ref_rows[I]
        e = rerank(q_rows[todo], cand)
        dr = mask.drop(q_rows[todo], cand, e)
        for a, i in enumerate(todo):
            for c, s, d in zip(cand[a], e[a], dr[a]):
                if not d:
                    keep[i][int(c)] = float(s)
        rpos = np.searchsorted(ref_rows, cand)
        add = sp.csr_matrix((np.ones(cand.size, np.float32), (np.repeat(todo, cand.shape[1]), rpos.ravel())),
                            shape=excl.shape)
        excl = ((excl + add) > 0).astype(np.float32).tocsr()
        n_vis = np.array([len(keep[i]) for i in todo])
        log(f"masked search round {rd}: {len(todo):,} queries; hidden among candidates {dr.mean():.3f}; queries with "
            f"< {k_keep} visible {int((n_vis < k_keep).sum()):,}")
        todo = todo[n_vis < k_keep]
        if not len(todo):
            break
    I = np.full((nq, k_keep), sentinel, np.int64)
    S = np.zeros((nq, k_keep), np.float32)
    for i in range(nq):
        it = sorted(keep[i].items(), key=lambda x: -x[1])[:k_keep]
        if it:
            I[i, :len(it)] = [x[0] for x in it]
            S[i, :len(it)] = [x[1] for x in it]
    return I, S
