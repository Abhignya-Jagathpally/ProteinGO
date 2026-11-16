"""Score a saved evidence file's neighbour vote on TV and/or TV-remote (Phase A selection).

  python scripts/score_evidence.py --suffix _k8 --benches tv,tvr
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.remote import BINS, RemoteMask
from tracego.eval.score import Evaluator
from tracego.models.retrieval import vote
from tracego.train.common import load_config, log

TAG = "score"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--benches", default="tv,tvr")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    R, E = cfg["retrieval"], cfg["encoder"]
    b = Bundle()
    evs = [np.load(paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{f}" / f"evidence_{R['space']}{a.suffix}.npz")
           for f in (0, 1)]
    out = {}
    for bench in a.benches.split(","):
        key = bench
        ev = Evaluator("tv")  # TV-remote uses the same ground truth as TV
        o = ev.o
        roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
        Yo = b.onto_labels("t0")
        P = None
        for z in evs:
            Pf = vote(z[f"{key}_I"], z[f"{key}_S"], Yo, R["T"], R["k_vote"], aspect_cols=o.ns, roots=roots)
            P = Pf if P is None else P + Pf
        P = (P / 2).tocsr()
        ids = b.ids[evs[0][f"{key}_q"]]
        cur = ev.curves(P, ids, onto=True)
        s = {k: round(float(v), 4) for k, v in ev.summary(cur).items()}
        s["bins"] = {k: round(v[1], 4) for k, v in ev.per_bin(cur).items()}
        if bench == "tvr":
            mask = RemoteMask()
            rem = {}
            for bi, name in enumerate(BINS):
                sub = ids[mask.bin == bi]
                rem[name] = (int((mask.bin == bi).sum()),
                             round(float(ev.sc.fmax(cur, subset=sub)["mean"]) if len(sub) else float("nan"), 4))
            s["remote_bins"] = rem
        out[bench] = s
        log(TAG, f"{bench}{a.suffix}: MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} mean {s['mean']:.4f}"
                 + (f" test-like {s['mean_testlike']:.4f}" if "mean_testlike" in s else ""))
    path = paths.RUNS / "phaseA" / (a.out or f"vote{a.suffix}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    log(TAG, f"wrote {path}")


if __name__ == "__main__":
    main()
