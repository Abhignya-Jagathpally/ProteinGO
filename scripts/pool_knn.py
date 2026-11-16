"""Phase A2: frozen mean-pool kNN votes from the large-encoder caches vs the current LoRA-35M vote.

Whitened cosine top-k (no residue re-rank: pools are mean vectors only). Scores TV v2 and TV-remote (DIAMOND-listed
hidden references excluded via the saved mask). Selection metrics only; no test/fresh24.

  python scripts/pool_knn.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import scipy.sparse as sp

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.remote import RemoteMask
from tracego.eval.score import Evaluator
from tracego.models.retrieval import topk_cosine, vote, whiten
from tracego.train.common import known_mask, log

TAG = "poolknn"
OUT = paths.RUNS / "phaseA" / "a2"
POOLS = {
    "esm2_650m": OUT.parent / "pools" / "esm2_650m_mean.npy",
    "esmc_600m": OUT.parent / "pools" / "esmc_600m_mean.npy",
    "prot_t5": OUT.parent / "pools" / "prot_t5_mean.npy",
    "lora35m_f0": paths.RUNS / "full" / "enc_lora35m_f0" / "h.npy",
}


def score_vote(I, S, Yo, R, o, has, ids, ev, roots):
    P = vote(I, S, Yo, R["T"], R["k_vote"], aspect_cols=o.ns, roots=roots)
    cur = ev.curves(P, ids, onto=True)
    s = ev.summary(cur)
    bins = {k: round(v[1], 4) for k, v in ev.per_bin(cur).items()}
    return {**{k: round(float(v), 4) for k, v in s.items()}, "bins": bins}, cur, P


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    b = Bundle()
    rows = np.load(paths.RUNS / "phaseA" / "pools" / "rows.npy")
    pos = {int(r): i for i, r in enumerate(rows)}
    tr, tv = b.rows("train"), b.rows("tv")
    ri = np.array([pos[r] for r in tr])
    qi = np.array([pos[r] for r in tv])
    mask = RemoteMask()
    excl = mask.exclusion(tv, tr)
    log(TAG, f"TV {len(tv):,}; train refs {len(tr):,}; TV-remote hidden pairs nnz {excl.nnz:,}")

    R = {"T": 0.05, "k_vote": 50, "cands": 100, "k_keep": 50}
    ev_tv = Evaluator("tv")
    o = ev_tv.o
    roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    Yo = b.onto_labels("t0")
    has = sp.csr_matrix(known_mask(b.labels("t0"), b.aspect).astype(np.float32))
    # TV-remote evaluator: same GT as TV, but report with remote identity bins from the mask
    res = {}
    for nm, path in POOLS.items():
        t = time.time()
        if nm.startswith("lora"):
            # LoRA fold-0 h is indexed by enc rows (same as pools/rows.npy)
            X = np.load(path)
            assert len(X) == len(rows)
        else:
            X = np.load(path)
        h = whiten(X.astype(np.float32), ri)
        I, S = topk_cosine(h[qi], h[ri], R["cands"], q_ids=b.ids[tv], r_ids=b.ids[tr], device="cuda")
        # keep top k_keep by cosine (no residue re-rank available for large pools)
        I, S = I[:, :R["k_keep"]], S[:, :R["k_keep"]]
        I_rows = tr[I]
        s_tv, cur_tv, _ = score_vote(I_rows, S, Yo, R, o, has, b.ids[tv], ev_tv, roots)
        # TV-remote: drop hidden references from the candidate list, then refill from a deeper cosine list
        I2, S2 = topk_cosine(h[qi], h[ri], 200, q_ids=b.ids[tv], r_ids=b.ids[tr], device="cuda", exclude=excl)
        I2r, S2r = tr[I2], S2.copy()
        # also drop by embedding-cosine proxy against the remote emb cut (unlisted near-identical leakage)
        from tracego.eval.remote import emb_cut
        over = S2r > emb_cut(mask.cut)[:, None]
        for i in np.where(over.any(1))[0]:
            al = mask.allow[i]
            if al:
                over[i] &= ~np.isin(I2r[i], list(al))
        S2r = np.where(over, -1.0, S2r)
        oo = np.argsort(-S2r, 1, kind="stable")[:, :R["k_keep"]]
        Ir = np.take_along_axis(I2r, oo, 1)
        Sr = np.take_along_axis(S2r, oo, 1)
        Sr = np.maximum(Sr, 0)
        s_tvr, cur_tvr, _ = score_vote(Ir, Sr, Yo, R, o, has, b.ids[tv], ev_tv, roots)
        # remote-bin Fmax using the mask's realised identity (not DIAMOND max_pident)
        rem = {}
        for bi, name in enumerate(["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]):
            ids = b.ids[tv[mask.bin == bi]]
            rem[name] = (int((mask.bin == bi).sum()),
                         round(float(ev_tv.sc.fmax(cur_tvr, subset=ids)["mean"]) if len(ids) else float("nan"), 4))
        res[nm] = {"tv": s_tv, "tvr": {**s_tvr, "remote_bins": rem}}
        log(TAG, f"{nm:12s} TV mean {s_tv['mean']:.4f} test-like {s_tv['mean_testlike']:.4f}; "
                 f"TV-remote mean {s_tvr['mean']:.4f} test-like {s_tvr['mean_testlike']:.4f}; "
                 f"remote no_hit {rem['no_hit'][1]} 100 {rem['100'][1]}; {time.time() - t:.0f}s")
    json.dump(res, open(OUT / "pool_knn.json", "w"), indent=1)
    log(TAG, f"wrote {OUT / 'pool_knn.json'}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
