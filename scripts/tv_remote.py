"""Builds the TV-remote mask (Phase A0): one identity cutoff per TV v2 protein, drawn so the realised identity-bin
mix matches fresh24 FR24-P. See tracego/eval/remote.py for the protocol.

Identity pool per query: the union of the cosine top-200 in three spaces (frozen ESM2-35M mean-pool, LoRA fold 0
and fold 1 mean-pools, all whitened) and the DIAMOND-listed training references; every pool pair gets its
embedding-alignment identity (chain+aa on the layer-8 codes). Seed 0; written once to runs/phaseA/tv_remote.npz.

  python scripts/tv_remote.py
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval import remote as rm
from tracego.models.retrieval import ResidueCodes, maxsim_rerank, topk_cosine, whiten
from tracego.train.common import log

TAG = "tvr"


def pool_candidates(b, tv, ref, depth, dev):
    spaces = {"frozen35m": (np.load(paths.CACHE / "frozen35m_mean.npy"), np.arange(len(b.ids)))}
    for f in (0, 1):
        d = paths.RUNS / "full" / f"enc_lora35m_f{f}"
        spaces[f"lora35m_f{f}"] = (np.load(d / "h.npy"), np.load(d / "rows.npy"))
    out = [set() for _ in tv]
    for nm, (X, rows) in spaces.items():
        pos = {int(r): i for i, r in enumerate(rows)}
        h = whiten(X, np.array([pos[r] for r in ref]))
        qi, ri = np.array([pos[r] for r in tv]), np.array([pos[r] for r in ref])
        I, S = topk_cosine(h[qi], h[ri], depth, q_ids=b.ids[tv], r_ids=b.ids[ref], device=dev)
        for i in range(len(tv)):
            out[i].update(ref[I[i]].tolist())
        log(TAG, f"{nm}: cosine top-{depth}; top-1 cosine p50 {np.median(S[:, 0]):.3f}")
    return out


@torch.no_grad()
def pool_identity(codes, tv, pools, width=200):
    E = []
    for i, q in enumerate(tv):
        c = np.array(sorted(pools[i]))
        e = np.zeros(len(c), np.float32)
        for s in range(0, len(c), width):
            e[s:s + width] = maxsim_rerank(codes, np.array([q]), c[None, s:s + width], "chainaa0.0")[0]
        E.append((c, e))
        if i % 2000 == 0:
            log(TAG, f"pool identity {i:,}/{len(tv):,}: pool size {len(c)}; max emb identity {e.max():.3f}")
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    b = Bundle()
    tv, ref = b.rows("tv"), b.rows("train")
    rpos = {a_: int(r) for a_, r in zip(b.ids[ref], ref)}
    h = pd.read_parquet(paths.TV2 / "diamond_hits_tv.parquet")
    h = h[h.s.isin(rpos) & (h.q != h.s)]
    listed = {}
    for q_, s_, p_ in zip(h.q.values, h.s.values, h.pident.values):
        d = listed.setdefault(q_, {})
        d[rpos[s_]] = max(d.get(rpos[s_], 0.0), float(p_))
    log(TAG, f"TV queries {len(tv):,}; training references {len(ref):,}; DIAMOND-listed pairs {len(h):,} for "
             f"{len(listed):,} queries")

    pools = pool_candidates(b, tv, ref, a.depth, dev)
    for i, q in enumerate(tv):
        pools[i].update(listed.get(b.ids[q], {}).keys())
    log(TAG, f"pool size per query p10/p50/p90 {np.percentile([len(p) for p in pools], [10, 50, 90]).tolist()}")
    codes = ResidueCodes(paths.RUNS / "full" / "codes_l8_d64.npz", dev, seqs=b.prot.seq.values)
    log(TAG, f"residue codes {tuple(codes.X.shape)} loaded ({time.time() - t0:.0f}s)")
    E = pool_identity(codes, tv, pools)

    nb = len(rm.BINS)
    realised = np.zeros((len(tv), nb))
    for i, q in enumerate(tv):
        c, e = E[i]
        L = listed.get(b.ids[q], {})
        is_l = np.array([int(x) in L for x in c])
        pid = np.array([L.get(int(x), 0.0) for x in c])
        for k, bn in enumerate(rm.BINS):
            cut = rm.CUT[bn]
            keep_l = is_l & (pid / 100 <= cut)
            keep_u = ~is_l & (e <= rm.emb_cut(np.array(cut)))
            best_l = pid[keep_l].max() if keep_l.any() else 0.0
            eu = e[keep_u]
            best_u = 100 * eu.max() if len(eu) and eu.max() > rm.HIT_EMB else 0.0
            realised[i, k] = max(best_l, best_u)
    ach = rm.bin_index(realised) == np.arange(nb)[None]
    target = np.array([rm.FR24P_COUNTS[x] for x in rm.BINS], float)
    target /= target.sum()
    log(TAG, f"achievable share per bin {dict(zip(rm.BINS, ach.mean(0).round(3).tolist()))}")
    rng = np.random.default_rng(a.seed)
    bi, expect = rm.ipf_assign(ach, target, rng)
    cut = np.array([rm.CUT[rm.BINS[k]] for k in bi])
    ident = realised[np.arange(len(tv)), bi]
    got = np.bincount(rm.bin_index(ident), minlength=nb) / len(tv)
    log(TAG, "bin mix target / expected / realised: " + "; ".join(
        f"{n} {t:.3f}/{e:.3f}/{g:.3f}" for n, t, e, g in zip(rm.BINS, target, expect, got)))

    hid_rows, hid_ptr, al_rows, al_ptr = [], [0], [], [0]
    for i, q in enumerate(tv):
        L = listed.get(b.ids[q], {})
        hd = [r for r, p in L.items() if p / 100 > cut[i]]
        al = [r for r, p in L.items() if p / 100 <= cut[i]]
        hid_rows += hd
        al_rows += al
        hid_ptr.append(len(hid_rows))
        al_ptr.append(len(al_rows))
    old = pd.read_parquet(paths.TV2 / "cells.parquet")
    old = old[old.split == "tv"].drop_duplicates("id").set_index("id").max_pident.reindex(b.ids[tv]).fillna(0).values
    log(TAG, f"plain TV mix {dict(zip(rm.BINS, (np.bincount(rm.bin_index(old), minlength=nb) / len(tv)).round(3)))}; "
             f"hidden DIAMOND-listed refs per query p50 {np.median(np.diff(hid_ptr)):.0f}; sample "
             f"{b.ids[tv[0]]}: plain {old[0]:.1f}% -> bin {rm.BINS[bi[0]]} (cut {cut[0]}), realised {ident[0]:.1f}%")
    rm.MASK.parent.mkdir(parents=True, exist_ok=True)
    np.savez(rm.MASK, q=tv, cut=cut.astype(np.float32), bin=np.array(rm.BINS)[bi], identity=ident.astype(np.float32),
             plain_identity=old.astype(np.float32), hid_rows=np.array(hid_rows, np.int64),
             hid_ptr=np.array(hid_ptr, np.int64), allow_rows=np.array(al_rows, np.int64),
             allow_ptr=np.array(al_ptr, np.int64), seed=a.seed, target=target, achievable=ach)
    log(TAG, f"wrote {rm.MASK} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
