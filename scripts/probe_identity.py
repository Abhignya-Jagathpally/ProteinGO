"""Diagnostic on TV v2: embedding-space identity proxies for the neighbour vote.

DIAMOND's kNN sharpens on percent identity (T=5). MaxSim averages cosines, which barely separates a 100% from a 90%
identical neighbour. Here each query/candidate pair gets residue-level proxies computed from the layer-8 codes only:
  a, b      MaxSim q->c and c->q (mean best cosine)
  fq, fc    fraction of query (candidate) residues whose best match exceeds tau
  cq, cc    fraction of query (candidate) residues whose best match exceeds tau AND continues the previous residue's
            match on the diagonal (j*(i+1) = j*(i)+1), an ungapped-alignment identity estimate
and the vote is scored with each proxy as the sharpening score. Saves the feature tensor for later use.

  python scripts/probe_identity.py
"""
import argparse
import time

import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator
from tracego.models.retrieval import ResidueCodes, topk_cosine, vote, whiten
from tracego.train.common import log

TAG = "ident"
TAUS = (0.9, 0.95, 0.98)


@torch.no_grad()
def pair_features(codes, q_rows, cand_rows):
    nf = 2 + 4 * len(TAUS)
    out = np.zeros(cand_rows.shape + (nf,), np.float32)
    for i, q in enumerate(q_rows):
        x = codes.get(q)
        C, m = codes.padded(cand_rows[i])
        sim = torch.einsum("id,cjd->cij", x, C).float().masked_fill(~m[:, None, :], -1)
        vq, jq = sim.max(2)                     # [c, Lq]
        vc, jc = sim.max(1)                     # [c, Lc]
        mf = m.float()
        nc = mf.sum(1)
        f = [vq.mean(1), (vc * mf).sum(1) / nc]
        chq = torch.zeros_like(vq, dtype=torch.bool)
        chq[:, 1:] = jq[:, 1:] == jq[:, :-1] + 1
        chq[:, :-1] |= jq[:, 1:] == jq[:, :-1] + 1
        chc = torch.zeros_like(vc, dtype=torch.bool)
        chc[:, 1:] = jc[:, 1:] == jc[:, :-1] + 1
        chc[:, :-1] |= jc[:, 1:] == jc[:, :-1] + 1
        for t in TAUS:
            f += [(vq > t).float().mean(1), ((vc > t) & m).float().sum(1) / nc,
                  ((vq > t) & chq).float().mean(1), ((vc > t) & chc & m).float().sum(1) / nc]
        out[i] = torch.stack(f, 1).cpu().numpy()
    return out


def names():
    n = ["a", "b"]
    for t in TAUS:
        n += [f"fq{t}", f"fc{t}", f"cq{t}", f"cc{t}"]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=str(paths.RUNS / "full" / "codes_l8_d64.npz"))
    ap.add_argument("--cands", type=int, default=100)
    a = ap.parse_args()
    b = Bundle()
    ids = b.ids
    ref, tv = b.rows("train"), b.rows("tv")
    E = np.load(paths.CACHE / "frozen35m_mean.npy").astype(np.float32)
    Zw = whiten(E, ref)
    I, S = topk_cosine(Zw[tv], Zw[ref], a.cands, q_ids=ids[tv], r_ids=ids[ref])
    t = time.time()
    codes = ResidueCodes(a.codes)
    log(TAG, f"codes loaded in {time.time() - t:.0f}s")
    t = time.time()
    F = pair_features(codes, tv, ref[I])
    log(TAG, f"pair features {F.shape} in {time.time() - t:.0f}s")
    out = paths.CACHE / "probe_identity.npz"
    np.savez(out, I=I, S=S, F=F.astype(np.float16), names=np.array(names()))
    log(TAG, f"saved {out}")
    ev = Evaluator("tv")
    Yo = b.onto_labels("t0")[ref]
    from tracego.train.common import known_mask
    has = known_mask(b.labels("t0")[ref], b.aspect)
    cols = ev.o.ns
    nm = names()

    def score(s, T, k=50, tag=""):
        o = np.argsort(-s, 1, kind="stable")
        I2, S2 = np.take_along_axis(I, o, 1), np.take_along_axis(s, o, 1)
        P = vote(I2, S2, Yo, T, k, aspect_cols=cols, ns_has=has)
        cur = ev.curves(P, ids[tv], onto=True)
        r = ev.summary(cur)
        log(TAG, f"{tag:28s} T={T:<6} k={k}: mean {r['mean']:.4f} test-like {r['mean_testlike']:.4f} "
                 f"(MF {r['MF']:.4f} BP {r['BP']:.4f} CC {r['CC']:.4f})")
        return r["mean_testlike"], cur

    mx = np.maximum(F[..., 0], F[..., 1])
    score(mx, 0.01, tag="max(a,b) [current]")
    best = (0, None, None)
    for j, n in enumerate(nm[2:], 2):
        for T in (0.02, 0.05, 0.1):
            v, cur = score(F[..., j], T, tag=n)
            if v > best[0]:
                best = (v, (n, T), cur)
    for t_i, t in enumerate(TAUS):
        j = 2 + 4 * t_i
        for nmx, s in ((f"max(fq,fc){t}", np.maximum(F[..., j], F[..., j + 1])),
                       (f"max(cq,cc){t}", np.maximum(F[..., j + 2], F[..., j + 3])),
                       (f"max(cq,cc){t}+a", np.maximum(F[..., j + 2], F[..., j + 3]) + 0.1 * mx)):
            for T in (0.02, 0.05, 0.1):
                v, cur = score(s, T, tag=nmx)
                if v > best[0]:
                    best = (v, (nmx, T), cur)
    log(TAG, f"best {best[1]} test-like {best[0]:.4f}; bins {ev.per_bin(best[2])}")


if __name__ == "__main__":
    main()
