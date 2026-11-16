"""Diagnostic on TV v2: can residue-level late interaction in ESM2 space recover DIAMOND-quality neighbour ranking?

Candidates come from the pooled (whitened) embedding; they are re-ranked by MaxSim over PCA-compressed residue
states of one ESM2 layer (ColBERT-style), and the re-ranked neighbours vote. DIAMOND hits are read only to report
candidate recall and rank correlation (diagnostic), never as a model input.

  python scripts/probe_rerank.py --layer 8 --dim 64
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator
from tracego.models.encoder import ESMEncoder, esm_tokenizer, length_batches, tokenize
from tracego.models.retrieval import topk_cosine, vote
from tracego.train.common import log

TAG = "rerank"


@torch.no_grad()
def residue_codes(rows, b, layer, dim, dev, esm="facebook/esm2_t12_35M_UR50D", pca=None):
    tok = esm_tokenizer(esm)
    enc = ESMEncoder(esm).to(dev).eval()
    seqs, lens = b.prot.seq.values[rows], b.prot.length.values[rows]
    out = [None] * len(rows)
    for bi in length_batches(lens, 100000):
        ids, am = tokenize(tok, [seqs[j] for j in bi], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = enc.esm(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[layer].float()
        m = enc.residue_mask(ids, am)
        if pca is None:
            X = hs[m][torch.randperm(int(m.sum()), device=dev)[:200000]]
            mu = X.mean(0)
            U, S, V = torch.pca_lowrank(X - mu, q=dim, center=False)
            pca = (mu, V[:, :dim])
        z = torch.nn.functional.normalize((hs - pca[0]) @ pca[1], dim=-1).half()
        for k, j in enumerate(bi):
            out[j] = z[k][m[k]].clone()
    return out, pca


def maxsim(q, cands):
    """q [Lq, d]; cands list of [Lc, d] -> (q->c mean max, c->q mean max) per candidate."""
    Lm = max(c.shape[0] for c in cands)
    C = torch.zeros(len(cands), Lm, q.shape[1], dtype=q.dtype, device=q.device)
    msk = torch.zeros(len(cands), Lm, dtype=torch.bool, device=q.device)
    for i, c in enumerate(cands):
        C[i, :len(c)] = c
        msk[i, :len(c)] = True
    sim = torch.einsum("id,cjd->cij", q, C).float()
    sim = sim.masked_fill(~msk[:, None, :], -1)
    a = sim.max(2).values.mean(1)
    bb = (sim.max(1).values * msk).sum(1) / msk.sum(1)
    return a, bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--cands", type=int, default=100)
    a = ap.parse_args()
    dev = torch.device("cuda")
    b = Bundle()
    ids = b.prot.id.to_numpy(dtype=str)
    ref, tv = b.rows("train"), b.rows("tv")
    E = np.load(paths.CACHE / "frozen35m_mean.npy").astype(np.float32)
    X = E - E[ref].mean(0)
    w, V = np.linalg.eigh(np.cov(X[ref].T))
    Zw = X @ (V / np.sqrt(w + 1e-6 * w.max()))
    I, S = topk_cosine(Zw[tv], Zw[ref], a.cands, q_ids=ids[tv], r_ids=ids[ref])
    h = pd.read_parquet(paths.TV2 / "diamond_hits_tv.parquet")
    h = h[h.s.isin(set(ids[ref]))].sort_values(["q", "bits"], ascending=[True, False]).drop_duplicates("q")
    best = dict(zip(h.q, h.s))
    cand_ids = ids[ref][I]
    for k in (1, 5, 10, 50, a.cands):
        hit = np.mean([best[q] in set(cand_ids[i, :k]) for i, q in enumerate(ids[tv]) if q in best])
        log(TAG, f"recall of DIAMOND best hit in pooled top-{k}: {hit:.3f}")
    t = time.time()
    need = np.unique(np.concatenate([tv, ref[np.unique(I)]]))
    codes, _ = residue_codes(need, b, a.layer, a.dim, dev)
    pos = {r: i for i, r in enumerate(need)}
    log(TAG, f"residue codes layer {a.layer} dim {a.dim} for {len(need):,} proteins in {time.time() - t:.0f}s")
    t = time.time()
    R1 = np.zeros_like(S)
    R2 = np.zeros_like(S)
    for i, q in enumerate(tv):
        x, y = maxsim(codes[pos[q]], [codes[pos[ref[j]]] for j in I[i]])
        R1[i], R2[i] = x.cpu().numpy(), y.cpu().numpy()
    log(TAG, f"MaxSim for {len(tv):,} x {a.cands} pairs in {time.time() - t:.0f}s")
    for k in (1, 5):
        for nm, R in (("q2c", R1), ("sym", (R1 + R2) / 2), ("min", np.minimum(R1, R2))):
            o = np.argsort(-R, 1)
            ci = np.take_along_axis(cand_ids, o, 1)
            hit = np.mean([best[q] in set(ci[i, :k]) for i, q in enumerate(ids[tv]) if q in best])
            log(TAG, f"recall of DIAMOND best hit in re-ranked ({nm}) top-{k}: {hit:.3f}")
    ev = Evaluator("tv")
    o = ev.o
    roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    Yo = b.onto_labels("t0")[ref]
    np.savez(paths.CACHE / f"probe_rerank_l{a.layer}_d{a.dim}.npz", I=I, S=S, R1=R1, R2=R2)
    for nm, R in (("q2c", R1), ("sym", (R1 + R2) / 2), ("max", np.maximum(R1, R2))):
        oo = np.argsort(-R, 1)
        I2, S2 = np.take_along_axis(I, oo, 1), np.take_along_axis(R, oo, 1)
        for T in (0.005, 0.01, 0.02):
            for k in (50,):
                cur = ev.curves(vote(I2, S2, Yo, T, k, aspect_cols=o.ns, roots=roots), ids[tv], onto=True)
                s = ev.summary(cur)
                log(TAG, f"re-ranked ({nm}) kNN T={T} k={k}: mean {s['mean']:.4f} test-like {s['mean_testlike']:.4f} "
                         f"(MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f})")
        log(TAG, f"bins ({nm}, T=0.01 k=50): {ev.per_bin(ev.curves(vote(I2, S2, Yo, 0.01, 50, aspect_cols=o.ns, roots=roots), ids[tv], onto=True))}")


if __name__ == "__main__":
    main()
