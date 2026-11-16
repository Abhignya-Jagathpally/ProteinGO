"""Diagnostic: a learned evidence calibrator. A small MLP maps the residue-level pair features of
probe_identity.py (plus rank and length ratio) to the IA-weighted label overlap of query and neighbour; trained on
train-fold queries against train references (self excluded), scored on TV v2 as the vote's sharpening score.

  python scripts/probe_calib.py --n 20000
"""
import argparse
import time

import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator
from tracego.models.retrieval import ResidueCodes, topk_cosine, vote, whiten
from tracego.train.common import known_mask, log
from scripts.probe_identity import pair_features

TAG = "calib"


def extra(F, I, lens_q, lens_r):
    k = F.shape[1]
    rank = np.broadcast_to(np.log1p(np.arange(k))[None, :, None], F.shape[:2] + (1,))
    lr = np.log(lens_r[I] / lens_q[:, None])[..., None]
    return np.concatenate([F, rank, lr, np.abs(lr)], -1).astype(np.float32)


def overlap(Yq, Yr, I, w):
    """IA-weighted Jaccard between each query and its candidates (namespace-pooled)."""
    import scipy.sparse as sp
    out = np.zeros(I.shape, np.float32)
    Yq, Yr = (Yq > 0).astype(np.float32).tocsr(), (Yr > 0).astype(np.float32).tocsr()
    Qw = Yq @ sp.diags(w.astype(np.float32))
    nq_w, nr_w = np.asarray(Qw.sum(1)).ravel(), np.asarray((Yr @ sp.diags(w.astype(np.float32))).sum(1)).ravel()
    k = I.shape[1]
    for s in range(0, len(I), 1000):
        ii = np.arange(s, min(s + 1000, len(I)))
        inter = np.asarray(Qw[np.repeat(ii, k)].multiply(Yr[I[ii].ravel()]).sum(1)).ravel().reshape(len(ii), k)
        out[ii] = inter / (nq_w[ii, None] + nr_w[I[ii]] - inter + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--cands", type=int, default=100)
    a = ap.parse_args()
    dev = torch.device("cuda")
    b = Bundle()
    ids = b.ids
    ref, tv = b.rows("train"), b.rows("tv")
    rng = np.random.default_rng(0)
    trq = np.sort(rng.choice(ref, a.n, replace=False))
    E = np.load(paths.CACHE / "frozen35m_mean.npy").astype(np.float32)
    Zw = whiten(E, ref)
    It, _ = topk_cosine(Zw[trq], Zw[ref], a.cands, q_ids=ids[trq], r_ids=ids[ref])
    z = np.load(paths.CACHE / "probe_identity.npz")
    Iv, Fv = z["I"], z["F"].astype(np.float32)
    codes = ResidueCodes(paths.RUNS / "full" / "codes_l8_d64.npz")
    t = time.time()
    Ft = pair_features(codes, trq, ref[It])
    log(TAG, f"train pair features {Ft.shape} in {time.time() - t:.0f}s")
    del codes
    torch.cuda.empty_cache()
    lens = b.prot.length.values.astype(np.float32)
    Xt, Xv = extra(Ft, It, lens[trq], lens[ref]), extra(Fv, Iv, lens[tv], lens[ref])
    Yo = b.onto_labels("t0")
    ev = Evaluator("tv")
    w = ev.o.ia.astype(np.float64) + 1.0 if hasattr(ev.o, "ia") else np.ones(Yo.shape[1])
    t = time.time()
    yt = overlap(Yo[trq], Yo[ref], It, w)
    log(TAG, f"targets in {time.time() - t:.0f}s: mean overlap {yt.mean():.3f}, top-1 {yt[:, 0].mean():.3f}")
    mu, sd = Xt.reshape(-1, Xt.shape[-1]).mean(0), Xt.reshape(-1, Xt.shape[-1]).std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[-1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64),
                              torch.nn.GELU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    X = torch.tensor((Xt - mu) / sd, device=dev).reshape(-1, Xt.shape[-1])
    Y = torch.tensor(yt, device=dev).reshape(-1)
    for ep in range(8):
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for s in range(0, len(X), 8192):
            j = perm[s:s + 8192]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[j]).squeeze(-1), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        log(TAG, f"epoch {ep}: bce {tot / len(X):.4f}")
    with torch.no_grad():
        sv = torch.sigmoid(net(torch.tensor((Xv - mu) / sd, device=dev)).squeeze(-1)).cpu().numpy()
    Yr = Yo[ref]
    has = known_mask(b.labels("t0")[ref], b.aspect)
    best = (0, None)
    for T in (0.02, 0.05, 0.1, 0.2):
        for k in (20, 50):
            o = np.argsort(-sv, 1, kind="stable")
            I2, S2 = np.take_along_axis(Iv, o, 1), np.take_along_axis(sv, o, 1)
            cur = ev.curves(vote(I2, S2, Yr, T, k, aspect_cols=ev.o.ns, ns_has=has), ids[tv], onto=True)
            r = ev.summary(cur)
            log(TAG, f"calibrated T={T} k={k}: mean {r['mean']:.4f} test-like {r['mean_testlike']:.4f} "
                     f"(MF {r['MF']:.4f} BP {r['BP']:.4f} CC {r['CC']:.4f})")
            if r["mean_testlike"] > best[0]:
                best = (r["mean_testlike"], (T, k), cur)
    log(TAG, f"best {best[1]} {best[0]:.4f}; bins {ev.per_bin(best[2])}")
    torch.save({"net": net.state_dict(), "mu": mu, "sd": sd}, paths.CACHE / "probe_calib.pt")


if __name__ == "__main__":
    main()
