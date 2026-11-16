"""Phase A3 diagnostic on TV v2: fragment recall of the candidate set and the 100%-identity bin.

For both cross-fitting spaces: cosine top-100 on the whitened LoRA mean-pool (current candidates), optionally merged
with the exact k-mer channel (tracego/models/kmer.py); re-rank by embedding-alignment identity, either maxed over
both directions (current) or normalised over the query length; then the sharpened k=50 vote, averaged over the two
spaces. Reports recall of DIAMOND's best hit (all, and the 100%-identity ones) in the candidates and in the
re-ranked top-1/top-5, and IA-Fmax (plain, test-like, identity bins). DIAMOND hits are read for recall only.

  python scripts/probe_fragments.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator
from tracego.models.kmer import KmerIndex
from tracego.models.retrieval import ResidueCodes, maxsim_rerank, topk_cosine, vote, whiten
from tracego.train.common import known_mask, log

TAG = "frag"
OUT = paths.RUNS / "phaseA" / "a3"


def merge(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise union of candidate rows (B may hold -1); duplicates and -1 become copies of the row's first entry,
    which the caller masks after the re-rank."""
    M = np.concatenate([A, B], 1)
    M = np.where(M < 0, M[:, :1], M)
    return M


def dup_mask(M: np.ndarray) -> np.ndarray:
    o = np.argsort(M, 1, kind="stable")
    s = np.take_along_axis(M, o, 1)
    d = np.zeros_like(M, bool)
    dd = np.zeros_like(M, bool)
    dd[:, 1:] = s[:, 1:] == s[:, :-1]
    np.put_along_axis(d, o, dd, 1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-occ", type=int, default=256)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = Bundle()
    ids = b.ids
    tv, ref = b.rows("tv"), b.rows("train")
    h = pd.read_parquet(paths.TV2 / "diamond_hits_tv.parquet")
    h = h[h.s.isin(set(ids[ref])) & (h.q != h.s)].sort_values(["q", "bits"], ascending=[True, False])
    best = h.drop_duplicates("q").set_index("q")
    rowof = {x: int(r) for x, r in zip(ids[ref], ref)}
    bq = np.array([rowof.get(best.s.get(q), -1) if q in best.index else -1 for q in ids[tv]])
    b100 = np.array([q in best.index and best.pident[q] >= 100 for q in ids[tv]])
    log(TAG, f"TV {len(tv):,}; DIAMOND best hit available {int((bq >= 0).sum()):,}; best hit at 100% "
             f"{int(b100.sum()):,}")

    seqs = b.prot.seq.values
    km = KmerIndex(seqs[ref], ref, a.k, a.max_occ, log=lambda m: log(TAG, m))
    t = time.time()
    KR, KC = km.search(seqs[tv], a.top, q_ids=ids[tv], r_ids=ids[ref])
    log(TAG, f"k-mer search {len(tv):,} queries in {time.time() - t:.0f}s; queries with >=1 candidate "
             f"{(KR[:, 0] >= 0).mean():.3f}; top-1 coverage p10/p50/p90 "
             f"{np.percentile(KC[:, 0], [10, 50, 90]).round(3).tolist()}; sample {ids[tv[0]]} -> "
             f"{[ids[r] if r >= 0 else '-' for r in KR[0, :3]]} {KC[0, :3].round(3).tolist()}")

    codes = ResidueCodes(paths.RUNS / "full" / "codes_l8_d64.npz", "cuda", seqs=seqs)
    ev = Evaluator("tv")
    o = ev.o
    Yo = b.onto_labels("t0")
    roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]

    def recall(C, mask=None):
        ok = bq >= 0
        hit = np.array([bq[i] in set(C[i][mask[i]] if mask is not None else C[i]) for i in range(len(tv))])
        return round(float(hit[ok].mean()), 4), round(float(hit[ok & b100].mean()), 4)

    res = {}
    cands = {}
    for f in (0, 1):
        d = paths.RUNS / "full" / f"enc_lora35m_f{f}"
        rows = np.load(d / "rows.npy")
        pos = {int(r): i for i, r in enumerate(rows)}
        hw = whiten(np.load(d / "h.npy"), np.array([pos[r] for r in ref]))
        I, S = topk_cosine(hw[[pos[r] for r in tv]], hw[[pos[r] for r in ref]], 100, q_ids=ids[tv], r_ids=ids[ref])
        cands[f] = ref[I]
    log(TAG, f"recall of DIAMOND best hit (all, 100%): cosine top-100 f0 {recall(cands[0])} f1 {recall(cands[1])}; "
             f"k-mer top-{a.top} {recall(KR)}; union f0 {recall(merge(cands[0], KR))}")
    res["recall"] = {"cos100_f0": recall(cands[0]), "cos100_f1": recall(cands[1]), "kmer": recall(KR),
                     "union_f0": recall(merge(cands[0], KR)), "union_f1": recall(merge(cands[1], KR))}

    variants = {"cos100/max": (False, "chainaa0.0", None), "cos100/qnorm": (False, "chainaaq0.0", None),
                "union/max": (True, "chainaa0.0", None), "union/qnorm": (True, "chainaaq0.0", None),
                "union/max|kcov": (True, "chainaa0.0", "max"), "union/qnorm|kcov": (True, "chainaaq0.0", "max")}
    scored = {}
    for nm, (use_k, mode, comb) in variants.items():
        t = time.time()
        P = None
        rec = []
        for f in (0, 1):
            C = merge(cands[f], KR) if use_k else cands[f]
            key = (use_k, mode, f)
            if key not in scored:
                scored[key] = maxsim_rerank(codes, tv, C, mode)
            s = scored[key].copy()
            if comb == "max":
                kc = ((C[:, :, None] == KR[:, None, :]) * KC[:, None, :]).max(2)
                s = np.maximum(s, kc)
            s[dup_mask(C)] = -1
            oo = np.argsort(-s, 1, kind="stable")[:, :50]
            I2, S2 = np.take_along_axis(C, oo, 1), np.take_along_axis(s, oo, 1)
            rec.append((recall(I2[:, :1]), recall(I2[:, :5])))
            Pf = vote(I2, S2, Yo, 0.05, 50, aspect_cols=o.ns, roots=roots)
            P = Pf if P is None else P + Pf
        cur = ev.curves((P / 2).tocsr(), ids[tv], onto=True)
        s = ev.summary(cur)
        pb = {k: round(v[1], 4) for k, v in ev.per_bin(cur).items()}
        res[nm] = {**{k: round(v, 4) for k, v in s.items()}, "bins": pb, "recall_top1_top5_f0": rec[0]}
        log(TAG, f"{nm:18s} MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} mean {s['mean']:.4f} test-like "
                 f"{s['mean_testlike']:.4f}; bins 95-99.9 {pb['95-99.9']} 100 {pb['100']}; recall top-1/top-5 "
                 f"(f0) {rec[0]}; {time.time() - t:.0f}s")
    json.dump(res, open(OUT / f"probe_fragments_k{a.k}.json", "w"), indent=1)
    log(TAG, f"wrote {OUT}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
