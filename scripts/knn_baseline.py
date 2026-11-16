"""Retrieval-only baseline: the embedding-space neighbour vote (pooled candidates on the adapted encoder, chain-
identity re-rank) over the full ontology, averaged over the two cross-fitting spaces. This is TraceGO's copy channel
on its own, i.e. the "no decoder" reference point.

  python scripts/knn_baseline.py --config configs/full.yaml --benches tv,test,fr24p
"""
import argparse

import numpy as np

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator, save_pred
from tracego.models.retrieval import vote
from tracego.train.common import load_config, log, save_json

TAG = "knn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benches", default="tv")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    R, E = cfg["retrieval"], cfg["encoder"]
    b = Bundle()
    out = paths.RUNS / cfg["name"] / (a.out or f"knn_{R['space']}")
    out.mkdir(parents=True, exist_ok=True)
    evs = [np.load(paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{f}" / f"evidence_{R['space']}.npz") for f in (0, 1)]
    res = {}
    for bench in a.benches.split(","):
        ev = Evaluator(bench)
        o = ev.o
        roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
        Yo = b.onto_labels("t0" if bench == "tv" else "full")
        P = None
        for z in evs:
            Pf = vote(z[f"{bench}_I"], z[f"{bench}_S"], Yo, R["T"], R["k_vote"], aspect_cols=o.ns, roots=roots)
            P = Pf if P is None else P + Pf
        P = (P / 2).tocsr()
        ids = b.ids[evs[0][f"{bench}_q"]]
        cur = ev.curves(P, ids, onto=True)
        s = ev.summary(cur)
        res[bench] = s
        save_pred(out / f"pred_{bench}.npz", P, ids, o.terms)
        log(TAG, f"{bench}: MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} mean {s['mean']:.4f}"
                 + (f" test-like {s['mean_testlike']:.4f}" if "mean_testlike" in s else ""))
    save_json(res, out / "summary.json")


if __name__ == "__main__":
    main()
