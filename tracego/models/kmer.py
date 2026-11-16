"""Exact k-mer candidate channel for near-identical fragments (Phase A3).

Mean-pooled embeddings of a fragment and of its full-length parent can be far apart, so a fragment's identical
reference may never reach the cosine top-C. This channel finds it from sequence alone, in-house (no DIAMOND or
MMseqs2): a sorted hash index of every k-mer (default 8) over the reference sequences; a query's k-mers are looked
up, hits are grouped by (reference, diagonal = ref_pos - query_pos), and a reference is scored by the query coverage
of its best diagonal, i.e. the fraction of the query's k-mers matched in register. k-mers occurring more than
`max_occ` times in the references (low-complexity repeats) are skipped.
"""
from __future__ import annotations

import time

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
BASE = 21


def encode(seqs) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated residue codes (0..19, 20 = other / separator) and the start offset of every sequence."""
    lut = np.full(256, 20, np.int8)
    for i, c in enumerate(AA):
        lut[ord(c)] = i
    lens = np.array([len(s) for s in seqs], np.int64)
    off = np.zeros(len(seqs) + 1, np.int64)
    off[1:] = np.cumsum(lens + 1)
    flat = np.full(int(off[-1]), 20, np.int8)
    joined = np.frombuffer("".join(s + "*" for s in seqs).encode(), np.uint8)
    flat[:] = lut[joined]
    return flat, off


def kmer_codes(flat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling base-21 code of every window of length k and whether the window is all standard residues."""
    n = len(flat) - k + 1
    code = np.zeros(max(n, 0), np.int64)
    bad = np.zeros(max(n, 0), bool)
    for j in range(k):
        x = flat[j:j + n].astype(np.int64)
        code = code * BASE + x
        bad |= x == 20
    return code, ~bad


class KmerIndex:
    def __init__(self, seqs, rows: np.ndarray, k: int = 8, max_occ: int = 256, log=print):
        t = time.time()
        self.k, self.max_occ, self.rows = k, max_occ, np.asarray(rows)
        flat, off = encode(seqs)
        code, ok = kmer_codes(flat, k)
        ref = np.repeat(np.arange(len(seqs), dtype=np.int32), np.diff(off))[: len(code)]
        pos = (np.arange(len(code)) - off[ref]).astype(np.int32)
        o = np.argsort(code[ok], kind="stable")
        self.code = code[ok][o]
        self.ref = ref[ok][o]
        self.pos = pos[ok][o]
        u, cnt = np.unique(self.code, return_counts=True)
        log(f"k-mer index: k={k}; {len(seqs):,} references; {len(self.code):,} k-mers ({len(u):,} distinct); "
            f"k-mers above max_occ={max_occ}: {(cnt > max_occ).sum():,} ({cnt[cnt > max_occ].sum() / len(self.code):.3f} "
            f"of occurrences); {time.time() - t:.0f}s; {self.code.nbytes * 2 / 1e9:.2f} GB")

    def search(self, seqs, top: int = 20, min_cov: float = 0.0, q_ids=None, r_ids=None, batch: int = 256):
        """-> (rows [nq, top] bundle rows of the references or -1, cov [nq, top] best-diagonal query coverage)."""
        nq = len(seqs)
        R = np.full((nq, top), -1, np.int64)
        C = np.zeros((nq, top), np.float32)
        nr = len(self.rows)
        for s in range(0, nq, batch):
            sub = seqs[s:s + batch]
            flat, off = encode(sub)
            code, ok = kmer_codes(flat, self.k)
            qi = np.repeat(np.arange(len(sub), dtype=np.int64), np.diff(off))[: len(code)]
            qp = np.arange(len(code)) - off[qi]
            code, qi, qp = code[ok], qi[ok], qp[ok]
            nk = np.bincount(qi, minlength=len(sub))
            lo = np.searchsorted(self.code, code, "left")
            hi = np.searchsorted(self.code, code, "right")
            cnt = hi - lo
            use = (cnt > 0) & (cnt <= self.max_occ)
            lo, cnt, qi, qp = lo[use], cnt[use], qi[use], qp[use]
            if not len(lo):
                continue
            tot = int(cnt.sum())
            start = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
            j = start + np.arange(tot)
            hq = np.repeat(qi, cnt)
            hr = self.ref[j].astype(np.int64)
            diag = self.pos[j].astype(np.int64) - np.repeat(qp, cnt) + (1 << 16)
            key = (hq * nr + hr) * (1 << 17) + diag
            uk, n = np.unique(key, return_counts=True)
            pair = uk >> 17
            # best diagonal per (query, reference)
            o = np.lexsort((-n, pair))
            pair, n = pair[o], n[o]
            first = np.r_[True, pair[1:] != pair[:-1]]
            pair, n = pair[first], n[first]
            q_loc, r_loc = pair // nr, pair % nr
            cov = n / np.maximum(nk[q_loc], 1)
            if q_ids is not None:
                same = q_ids[s + q_loc] == r_ids[r_loc]
                cov = np.where(same, -1.0, cov)
            o = np.lexsort((-cov, q_loc))
            q_loc, r_loc, cov = q_loc[o], r_loc[o], cov[o]
            rank = np.arange(len(q_loc)) - np.searchsorted(q_loc, q_loc, "left")
            keep = (rank < top) & (cov >= max(min_cov, 0.0)) & (cov > 0)
            R[s + q_loc[keep], rank[keep]] = self.rows[r_loc[keep]]
            C[s + q_loc[keep], rank[keep]] = cov[keep]
        return R, C
