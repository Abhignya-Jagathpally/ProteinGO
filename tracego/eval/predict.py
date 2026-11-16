"""Predictions of a trained decoder on TV v2, the GO-GPT test and fresh24 FR24-P, written in the repo's npz contract.

Scores on vocabulary terms come from the decoder (readout chosen on TV v2); terms outside the 7,901-term
vocabulary come from the copy channel only (the neighbour vote over the references' full label sets). Both cross-
fitting spaces are scored and averaged.

  python -m tracego.eval.predict --config configs/full.yaml --tag main --readout ccms --benches tv,test,fr24p
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator, save_pred
from tracego.models.decoder import Hierarchy, TraceDecoder
from tracego.models.retrieval import vote
from tracego.train.common import load_config, log, save_json
from tracego.train import train_decoder as td

TAG = "pred"


def oov_vote(b, ev, I, S, which, R) -> sp.csr_matrix:
    """Neighbour vote over every ontology term outside the vocabulary (ontology columns)."""
    Yo = b.onto_labels(which)
    o = ev.o
    roots = [o.idx[r] for r in ("GO:0003674", "GO:0008150", "GO:0005575")]
    P = vote(I, S, Yo, R["T"], R["k_vote"], aspect_cols=o.ns, roots=roots).tocsc()
    keep = np.ones(P.shape[1], bool)
    keep[ev.vcols] = False
    D = sp.diags(keep.astype(np.float32))
    return (P @ D).tocsr()


def load_model(cfg, run, b, dev, nterms_dim):
    D = cfg["decoder"]
    model = TraceDecoder(D["t5"], b.terms, b.aspect, d_latent=nterms_dim, copy_bias=D["copy_bias"],
                         homology=bool(D.get("homology_token", False))).to(dev)
    st = torch.load(run / "model.pt", map_location=dev, weights_only=False)
    model.load_state_dict(st["state"])
    return model.eval()


def setup(cfg, tag, dev, names):
    """Everything predict() needs: bundle, spaces, model, instruction sets, collators, hierarchies."""
    D, E, R = cfg["decoder"], cfg["encoder"], cfg["retrieval"]
    run = paths.RUNS / cfg["name"] / f"dec_{tag}"
    b = Bundle()
    enc_dirs = [paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{f}" for f in (0, 1)]
    evid_suffix = R.get("evid_suffix", "")
    spaces = [td.Space(d, d / f"evidence_{R['space']}{evid_suffix}.npz") for d in enc_dirs]
    model = load_model(cfg, run, b, dev, spaces[0].Z.shape[-1])
    tok_ids = {"encode": lambda s: model.tok(s, add_special_tokens=False)["input_ids"], "nb": model.nb_id,
               "sim0": model.sim0, "tax0": model.tax0, "hom0": getattr(model, "hom0", model.sim0),
               "go0": model.go0, "eos": model.tok.eos_token_id}
    from tracego.data.instructions import Collator, leaves_by_ns
    ref_leaves = {}
    for which in ("t0", "full"):
        Yl = b.labels(which)
        rows = np.where(np.diff(Yl.indptr) > 0)[0]
        ref_leaves[which] = {int(r): l for r, l in zip(rows, leaves_by_ns(Yl[rows], b.anc, b.aspect, b.ia,
                                                                         D["leaves_per_nb"]))}
    sets = td.build_sets(cfg, b, spaces, tok_ids, ref_leaves, {}, names)
    Hs, parents = [], []
    for ns in range(3):
        H = Hierarchy(b.anc, model.ns_cols[ns].numpy(), dev)
        Hs.append(H)
        d = {}
        for ci, pi in zip(*H.parents_np):
            d.setdefault(int(ci), []).append(int(pi))
        parents.append(d)
    coll = [Collator(spaces[f].Z, spaces[f].zpos, model.ns_cols, parents, D["max_src"], D["max_tgt"],
                     D["constrained"]) for f in (0, 1)]
    return {"b": b, "spaces": spaces, "model": model, "sets": sets, "coll": coll, "Hs": Hs, "run": run}


def predict_bench(S_, cfg, bench, readouts, dev, oov=True):
    b, spaces, model, sets, coll, Hs = (S_[k] for k in ("b", "spaces", "model", "sets", "coll", "Hs"))
    R = cfg["retrieval"]
    ev = Evaluator(bench)
    mats = {r: None for r in readouts}
    O = None
    for f in (0, 1):
        iset = sets[(bench, f)]
        vt = td.vote_matrix(spaces[f], bench)
        dense = {r: {} for r in readouts}
        for ns in range(3):
            res = td.predict(model, iset, ns, vt, coll[f], Hs[ns], cfg, dev, readouts)
            for r in readouts:
                dense[r][ns] = res[r]
        for r in readouts:
            M = td.to_vocab_csr(dense[r], model.ns_cols, len(iset.q_rows), model.n_terms)
            mats[r] = M if mats[r] is None else mats[r] + M
        if oov:
            which = "t0" if bench in ("tv", "tvr") else "full"
            Of = oov_vote(b, ev, spaces[f].ev[f"{bench}_I"], spaces[f].ev[f"{bench}_S"], which, R)
            O = Of if O is None else O + Of
    ids = b.ids[sets[(bench, 0)].q_rows]
    out = {}
    for r in readouts:
        P = ev.to_onto(mats[r] / 2)
        if O is not None:
            P = (P + O / 2).tocsr()
        out[r] = P
    return out, ids, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--readouts", default="step0,ccms,mix")
    ap.add_argument("--benches", default="tv")
    ap.add_argument("--no-oov", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dev = torch.device("cuda")
    benches = a.benches.split(",")
    readouts = tuple(a.readouts.split(","))
    S_ = setup(cfg, a.tag, dev, ["tv"] + [x for x in benches if x != "tv"])
    res = {}
    for bench in benches:
        t = time.time()
        P, ids, ev = predict_bench(S_, cfg, bench, readouts, dev, oov=not a.no_oov)
        for r, M in P.items():
            cur = ev.curves(M, ids, onto=True)
            s = ev.summary(cur)
            res[f"{bench}/{r}"] = s
            save_pred(S_["run"] / f"pred_{bench}_{r}.npz", M, ids, ev.o.terms)
            log(TAG, f"{bench} readout {r}: MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} mean {s['mean']:.4f}"
                     + (f" test-like {s['mean_testlike']:.4f}" if "mean_testlike" in s else "") +
                     f" ({time.time() - t:.0f}s)")
    save_json(res, S_["run"] / f"pred_summary_{'_'.join(benches)}.json")


if __name__ == "__main__":
    main()
