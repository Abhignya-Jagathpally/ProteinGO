"""Neighbour lists and copy-bias votes for every query set in one cross-fitting space.

Two-stage retrieval, entirely in the encoder's representation space:
  1. candidates: cosine top-C on the whitened adapted mean-pooled states h (this fold's adapter), optionally merged
     with the exact k-mer channel (retrieval.kmer, Phase A3: references sharing in-register 8-mers with the query);
  2. re-rank: identity along the embedding alignment on the shared residue codes; the re-ranked score is the
     evidence similarity.
Query sets and references
  train  training proteins the fold's adapter did NOT see -> references: training proteins (T0 labels)
  tv     TV v2 queries                                   -> references: training proteins (T0 labels)
  tvr    TV-remote: TV v2 queries under the saved mask (runs/phaseA/tv_remote.npz; tracego/eval/remote.py)
  val / test / fr24p (full scale only)                   -> references: GO-GPT train + val (full labels)
Same-accession references are always excluded.

  python -m tracego.train.build_evidence --config configs/full.yaml --fold 0 [--sets tvr --suffix _tvr]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.data.instructions import sim_edges
from tracego.models.retrieval import ResidueCodes, maxsim_rerank, topk_cosine, vote, whiten
from tracego.train.common import fold_of, known_mask, load_config, log, train_rows

TAG = "evid"


def merge_candidates(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise concatenation of two candidate row sets (B may hold -1 for no candidate); -1 entries become copies
    of the row's first candidate and are removed with the other duplicates by `duplicates`."""
    M = np.concatenate([A, B], 1)
    return np.where(M < 0, M[:, :1], M)


def duplicates(M: np.ndarray) -> np.ndarray:
    """[n, C] True on every repeat of an earlier entry of the same row."""
    o = np.argsort(M, 1, kind="stable")
    s = np.take_along_axis(M, o, 1)
    dd = np.zeros_like(M, bool)
    dd[:, 1:] = s[:, 1:] == s[:, :-1]
    out = np.zeros_like(M, bool)
    np.put_along_axis(out, o, dd, 1)
    return out


class Retriever:
    """Candidates (cosine, + k-mer channel) and the identity re-rank for one fold space and one reference set."""

    def __init__(self, b, h, pos, ref_rows, codes, R, dev, kmer=None):
        self.b, self.h, self.pos, self.ref, self.codes, self.R, self.dev, self.kmer = b, h, pos, ref_rows, codes, R, dev, kmer
        self.ri = np.array([pos[r] for r in ref_rows])

    def cosine(self, q_rows, depth, exclude=None):
        qi = np.array([self.pos[r] for r in q_rows])
        I, S = topk_cosine(self.h[qi], self.h[self.ri], depth, q_ids=self.b.ids[q_rows],
                           r_ids=self.b.ids[self.ref], device=self.dev, exclude=exclude)
        return self.ref[I]

    def kmer_cands(self, q_rows):
        R = self.R
        KR, KC = self.kmer.search(self.b.prot.seq.values[q_rows], R.get("kmer_top", 20), q_ids=self.b.ids[q_rows],
                                  r_ids=self.b.ids[self.kmer.rows])
        return KR, KC

    def rerank(self, q_rows, cand):
        s = maxsim_rerank(self.codes, q_rows, cand, self.R["mode"])
        s[duplicates(cand)] = -1
        return s

    def blend(self, cand, s, KR, KC):
        """Optionally take the max of the re-rank score and the k-mer query coverage (Phase A3: union/max|kcov)."""
        if KR is None or self.R.get("kmer_comb") != "max":
            return s
        kc = ((cand[:, :, None] == KR[:, None, :]) * KC[:, None, :]).max(2)
        return np.maximum(s, kc)

    def neighbours(self, q_rows):
        R = self.R
        cand = self.cosine(q_rows, R["cands"])
        KR = KC = None
        if self.kmer is not None:
            KR, KC = self.kmer_cands(q_rows)
            cand = merge_candidates(cand, KR)
        if self.codes is None:
            return cand[:, :R["k_keep"]], np.zeros(cand[:, :R["k_keep"]].shape, np.float32)
        s = self.blend(cand, self.rerank(q_rows, cand), KR, KC)
        s[duplicates(cand)] = -1
        o = np.argsort(-s, 1, kind="stable")[:, :R["k_keep"]]
        return np.take_along_axis(cand, o, 1), np.take_along_axis(s, o, 1)

    def masked(self, q_rows, mask, sentinel):
        """TV-remote / curriculum neighbours: hidden references are excluded / dropped (tracego/eval/remote.py)."""
        from tracego.eval.remote import masked_search
        R = self.R
        KR = KC = None
        if self.kmer is not None:
            KR, KC = self.kmer_cands(q_rows)
        first = {"done": False}

        def cosine(todo, excl):
            cand = self.cosine(q_rows[todo], R.get("cands_remote", 200), exclude=excl)
            if KR is not None and not first["done"]:
                first["done"] = True
                kr = KR[todo].copy()
                for a, i in enumerate(todo):
                    hid = excl[a].indices if excl.nnz else np.array([], np.int64)
                    ok = kr[a] >= 0
                    if len(hid):
                        kr[a, ok & np.isin(np.searchsorted(self.ref, kr[a]), hid)] = -1
                cand = merge_candidates(cand, kr)
            return np.searchsorted(self.ref, cand)

        def rerank(qs, cand):
            s = self.rerank(qs, cand)
            if KR is None:
                return s
            # align k-mer coverage to the current query subset
            idx = np.array([np.where(q_rows == q)[0][0] for q in qs])
            return self.blend(cand, s, KR[idx], KC[idx])

        return masked_search(mask, q_rows, self.ref, cosine, rerank, R["k_keep"], sentinel,
                             log=lambda m: log(TAG, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--sets", default="train,tv,val,test,fr24p")
    ap.add_argument("--suffix", default="")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    R, E = cfg["retrieval"], cfg["encoder"]
    run = paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{a.fold}"
    dev = "cuda"
    t0 = time.time()
    b = Bundle()
    rows = np.load(run / "rows.npy")
    pos = {int(r): i for i, r in enumerate(rows)}
    tr = train_rows(b, cfg["data"]["subset"], cfg.get("seed", 0))
    fold = fold_of(tr)
    h = whiten(np.load(run / "h.npy"), np.array([pos[r] for r in tr]))
    codes = None
    if R.get("rerank", True):
        codes = ResidueCodes(paths.RUNS / cfg["name"] / f"codes_l{R['code_layer']}_d{R['code_dim']}.npz", dev,
                             seqs=b.prot.seq.values if R["mode"].startswith("chainaa") else None)
        log(TAG, f"residue codes {tuple(codes.X.shape)} loaded")
    want = a.sets.split(",")
    sets = {"train": (tr[fold != a.fold], tr, "t0"), "tv": (b.rows("tv"), tr, "t0"), "tvr": (b.rows("tv"), tr, "t0"),
            "train_cur": (tr[fold != a.fold], tr, "t0")}
    if not cfg["data"]["subset"]:
        rf = b.rows("ref_full")
        for nm in ("val", "test", "fr24p"):
            sets[nm] = (b.rows(nm), rf, "full")
    sets = {k: v for k, v in sets.items() if k in want}
    mask = None
    if "tvr" in sets:
        from tracego.eval.remote import RemoteMask
        mask = RemoteMask()
        log(TAG, f"TV-remote mask: {mask.n:,} queries; bins {dict(zip(*np.unique(mask.bin, return_counts=True)))}")
    cur_mask = None
    if "train_cur" in sets:
        from tracego.eval.remote import CurriculumMask
        cur_mask = CurriculumMask(sets["train_cur"][0], seed=cfg.get("seed", 0) + 17 * a.fold,
                                  unmasked=float(R.get("curriculum_unmasked", 0.25)))
        log(TAG, f"homology curriculum mask: {cur_mask.n:,} queries; mix {cur_mask._log_bins}")
    # a row without labels in either reference set fills the slots of a masked query with too few visible neighbours
    sentinel = int(b.rows("test")[0])
    kidx = {}
    out = {}
    for nm, (q, ref, lab) in sets.items():
        t = time.time()
        km = None
        if R.get("kmer", False):
            key = len(ref)
            if key not in kidx:
                from tracego.models.kmer import KmerIndex
                kidx[key] = KmerIndex(b.prot.seq.values[ref], ref, R.get("kmer_k", 8), R.get("kmer_max_occ", 256),
                                      log=lambda m: log(TAG, m))
            km = kidx[key]
        ret = Retriever(b, h, pos, ref, codes, R, dev, km)
        if nm == "tvr":
            I, S = ret.masked(q, mask, sentinel)
            out["tvr_bin"] = mask.bin.astype(np.int8)
            out["tvr_cut"] = mask.cut.astype(np.float32)
        elif nm == "train_cur":
            I, S = ret.masked(q, cur_mask, sentinel)
        else:
            I, S = ret.neighbours(q)
        Y = b.labels(lab)
        # vote over the vocabulary (copy bias); columns = vocab order; aspect-normalised per namespace
        ns = np.array(["FPC".index(x) for x in b.aspect])
        has = sp.csr_matrix(known_mask(Y, b.aspect).astype(np.float32))
        V = vote(I, S, Y, R["T"], R["k_vote"], aspect_cols=ns, ns_has=has)
        # curriculum / TV-remote examples are stored under a short name the decoder already understands
        key = "train" if nm == "train_cur" else nm
        out.update({f"{key}_q": q, f"{key}_I": I.astype(np.int32), f"{key}_S": S.astype(np.float32),
                    f"{key}_vd": V.data, f"{key}_vi": V.indices, f"{key}_vp": V.indptr, f"{key}_vs": np.array(V.shape)})
        if nm == "train_cur":
            out["train_hom_bin"] = cur_mask.bin.astype(np.int8)
            out["train_hom_cut"] = cur_mask.cut.astype(np.float32)
        log(TAG, f"{nm}: {len(q):,} queries vs {len(ref):,} references ({lab} labels) in {time.time() - t:.0f}s; "
                 f"top-1 score p10/p50/p90 {np.percentile(S[:, 0], [10, 50, 90]).round(4).tolist()}; vote nnz/query "
                 f"{V.nnz / max(1, len(q)):.0f}; sample {b.ids[q[0]]} -> {b.ids[I[0, :3]].tolist()} "
                 f"{S[0, :3].round(4).tolist()}")
    if "train_S" in out:
        out["sim_edges"] = sim_edges(out["train_S"][:, :R["k"]])
        log(TAG, f"similarity-bin edges (24 quantile bins of training-query evidence): "
                 f"{out['sim_edges'][[0, 6, 12, 18, 22]].round(4).tolist()}")
    elif "sim_edges" not in out:
        # TV / TV-remote-only rebuilds reuse the main evidence file's edges so the decoder's sim tokens stay aligned
        main = run / f"evidence_{R['space']}.npz"
        if main.exists():
            out["sim_edges"] = np.load(main)["sim_edges"]
            log(TAG, f"reused sim_edges from {main.name}")
    f_out = run / f"evidence_{R['space']}{a.suffix}.npz"
    np.savez(f_out, **out)
    log(TAG, f"wrote {f_out} ({sum(v.nbytes for v in out.values() if hasattr(v, 'nbytes')) / 1e6:.0f} MB) in "
             f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
