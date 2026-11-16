"""Raw ESM2 baselines on TV v2 (selection data): frozen mean-pool + linear probe, and frozen-embedding kNN vote,
next to the repo's DIAMOND kNN with the same (train-only, T0) references.

  python scripts/baseline_frozen.py --emb runs/cache/frozen35m_mean.npy --out runs/baselines/frozen35m
"""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator, load_knn_contract
from tracego.models.retrieval import topk_cosine, vote
from tracego.train.common import known_mask, log, save_json, term_weights, topk_csr, train_rows

TAG = "base"


def linear_probe(X, Y, K, aspect_idx, w, Xq, dev, epochs=30, lr=3e-3):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt = (Xt - mu) / sd
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(dev)
    torch.nn.init.zeros_(lin.weight)
    with torch.no_grad():
        p = np.asarray(Y.mean(0)).ravel().clip(1e-4, 1 - 1e-4)
        lin.bias.copy_(torch.as_tensor(np.log(p / (1 - p)), device=dev))
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1e-4)
    wt = torch.as_tensor(w, device=dev)
    n = len(X)
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(n)
        tot = 0
        for s in range(0, n, 2048):
            i = perm[s:s + 2048]
            yb = torch.as_tensor(Y[i].toarray(), device=dev)
            kb = torch.as_tensor(K[i][:, aspect_idx], dtype=torch.float32, device=dev)
            l = torch.nn.functional.binary_cross_entropy_with_logits(lin(Xt[i]), yb, reduction="none")
            loss = (l * wt * kb).sum() / kb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(i)
        if ep % 10 == 0 or ep == epochs - 1:
            log(TAG, f"  probe epoch {ep} loss {tot / n:.4f}")
    with torch.no_grad():
        Q = (torch.as_tensor(Xq, dtype=torch.float32, device=dev) - mu) / sd
        return torch.sigmoid(lin(Q)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(paths.CACHE / "frozen35m_mean.npy"))
    ap.add_argument("--out", default=str(paths.RUNS / "baselines" / "frozen35m"))
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    b = Bundle()
    E = np.load(a.emb)
    ref = train_rows(b, a.subset)
    tv = b.rows("tv")
    ids = b.prot.id.values
    log(TAG, f"embeddings {E.shape}; references {len(ref):,} (v2 train, T0 labels); TV queries {len(tv):,}")
    ev = Evaluator("tv")
    res = {}
    if not a.subset:
        P, pid = load_knn_contract(paths.TV2 / "knn" / "tv_knn.npz", full=True)
        cur = ev.curves(P, pid, onto=True)
        res["diamond_knn_T5_k50_na (train refs)"] = ev.summary(cur)
        res["diamond_knn_bins"] = ev.per_bin(cur)
        log(TAG, f"DIAMOND kNN (repo, train refs): {res['diamond_knn_T5_k50_na (train refs)']}")
    t = time.time()
    I, S = topk_cosine(E[tv], E[ref], 50, q_ids=ids[tv], r_ids=ids[ref])
    log(TAG, f"cosine top-50 for TV in {time.time() - t:.1f}s; sim@1 p50 {np.median(S[:, 0]):.4f}; "
             f"sample: query {ids[tv[0]]} -> {ids[ref[I[0, :3]]].tolist()} sims {S[0, :3].round(4).tolist()}")
    Yo = b.onto_labels("t0")[ref]
    o = ev.o
    roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    best = None
    for T in (0.002, 0.005, 0.01, 0.02, np.inf):
        for k in (5, 10, 50):
            P = vote(I, S, Yo, T, k, aspect_cols=o.ns, roots=roots)
            s = ev.summary(ev.curves(P, ids[tv], onto=True))
            res[f"emb_knn_T{T}_k{k}"] = s
            log(TAG, f"frozen-embedding kNN T={T} k={k}: mean {s['mean']:.4f} test-like {s['mean_testlike']:.4f} "
                     f"(MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f})")
            if best is None or s["mean_testlike"] > best[0]:
                best = (s["mean_testlike"], T, k)
    P = vote(I, S, Yo, best[1], best[2], aspect_cols=o.ns, roots=roots)
    res["emb_knn_best"] = {"T": best[1], "k": best[2], "bins": ev.per_bin(ev.curves(P, ids[tv], onto=True))}
    log(TAG, f"best frozen-embedding kNN on TV v2: T={best[1]} k={best[2]} test-like {best[0]:.4f}; bins "
             f"{res['emb_knn_best']['bins']}")
    K = known_mask(b.Y_t0, b.aspect)
    aspect_idx = np.array(["FPC".index(x) for x in b.aspect])
    w = term_weights(b.ia, True)
    t = time.time()
    p = linear_probe(E[ref].astype(np.float32), b.Y_t0[ref], K[ref], aspect_idx, w, E[tv].astype(np.float32), dev)
    s = ev.summary(ev.curves(topk_csr(p), ids[tv]))
    res["frozen_meanpool_linear_probe"] = s
    res["probe_bins"] = ev.per_bin(ev.curves(topk_csr(p), ids[tv]))
    log(TAG, f"frozen mean-pool linear probe ({time.time() - t:.0f}s): {s}")
    save_json(res, out / "tv_results.json")
    log(TAG, f"wrote {out / 'tv_results.json'}")


if __name__ == "__main__":
    main()
