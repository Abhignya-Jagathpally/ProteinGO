"""Phase D2: taxonomic distance vs GO accuracy, leave-one-clade-out retrieval, ortho vs para neighbours.

Uses existing evidence_ca.npz + TV predictions; leave-clade-out re-runs whitened-h cosine with same-clade
references masked, then chain-identity re-rank (ResidueCodes) and the knn vote. Decoder is not re-run (anchored).

  CUDA_VISIBLE_DEVICES=0 python scripts/analysis_d2_taxonomy.py           # full
  CUDA_VISIBLE_DEVICES=0 python scripts/analysis_d2_taxonomy.py --skip-loo # tables only
"""
from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tracego.analysis.common import (FULL, disk_free_gb, evidence, log, out_dir, per_protein_f, tv_identity,
                                     tv_predictions)
from tracego.analysis.taxonomy import (COARSE, COARSE_ORDER, CONDITIONS, GROUPS, clade_keys, ensemble_vote,
                                       group_scores, grid, id4_of, keep_top, lca_group, missing_ranks,
                                       per_protein_table, rerank_union, topk_cosine_clade)
from tracego.data.bundle import Bundle, RANKS
from tracego.eval.score import Evaluator, bin_of
from tracego.models.retrieval import ResidueCodes, whiten
from tracego.train.common import save_json, train_rows

TAG = "d2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loo", action="store_true")
    ap.add_argument("--cands", type=int, default=100)
    a = ap.parse_args()
    od = out_dir("d2")
    log(TAG, f"disk free {disk_free_gb():.1f} GB; out {od}")
    b = Bundle()
    tv = b.rows("tv")
    ids = b.ids[tv]
    ev = Evaluator("tv")
    ident = tv_identity(ids)
    id4 = id4_of(ident)
    id_bin = bin_of(ident)

    # ---- existing predictions
    Pv, idv = tv_predictions("vote")
    assert list(idv) == list(ids) or set(idv) >= set(ids)
    # align
    pos = {str(x): i for i, x in enumerate(idv)}
    ord_ = np.array([pos[str(x)] for x in ids])
    Pv = Pv[ord_]
    cur_v = ev.curves(Pv, ids, onto=True)
    s_v = ev.summary(cur_v)
    log(TAG, f"vote TV mean {s_v['mean']:.4f} test-like {s_v['mean_testlike']:.4f}")
    try:
        Pt, idt = tv_predictions("tracego")
        pos_t = {str(x): i for i, x in enumerate(idt)}
        Pt = Pt[np.array([pos_t[str(x)] for x in ids])]
        cur_t = ev.curves(Pt, ids, onto=True)
        s_t = ev.summary(cur_t)
        log(TAG, f"tracego TV mean {s_t['mean']:.4f} test-like {s_t['mean_testlike']:.4f}")
    except Exception as e:
        log(TAG, f"tracego preds unavailable: {e}; vote-only for TraceGO columns")
        cur_t, s_t = cur_v, s_v

    ppf_v = per_protein_table(ev, cur_v, per_protein_f)
    ppf_t = per_protein_table(ev, cur_t, per_protein_f)

    # ---- best neighbour LCA from fold-0 evidence
    z0 = evidence(0, "ca")
    q = z0["tv_q"]
    assert np.array_equal(q, tv)
    n0 = z0["tv_I"][:, 0]
    s0 = z0["tv_S"][:, 0]
    g = lca_group(b, tv, n0)
    log(TAG, f"LCA groups: {dict(zip(*np.unique(g, return_counts=True)))}")
    log(TAG, f"missing ranks: {missing_ranks(b, tv, n0)}")
    # cross-check fold 1
    z1 = evidence(1, "ca")
    g1 = lca_group(b, z1["tv_q"], z1["tv_I"][:, 0])
    # map fold1 to same tv order
    pos1 = {int(r): i for i, r in enumerate(z1["tv_q"])}
    g1a = np.array([g1[pos1[int(r)]] for r in tv])
    agree = float((g == g1a).mean())
    log(TAG, f"fold0 vs fold1 LCA agreement {agree:.3f}")

    order = [x for x in GROUPS if (g == x).any()]
    tab_v = group_scores(ev, cur_v, ids, g, order, ppf_v)
    tab_t = group_scores(ev, cur_t, ids, g, order, ppf_t)
    pd.DataFrame(tab_v).to_csv(od / "fmax_by_lca_vote.tsv", sep="\t", index=False)
    pd.DataFrame(tab_t).to_csv(od / "fmax_by_lca_tracego.tsv", sep="\t", index=False)
    log(TAG, f"LCA x vote:\n{pd.DataFrame(tab_v).to_string(index=False)}")

    coarse = np.array([COARSE.get(x, x) for x in g])
    m_v, n_v = grid(ppf_v.reindex(ids), coarse, id4, COARSE_ORDER, ("<50", "50-80", "80-95", "95-100"))
    m_v.to_csv(od / "ppF_lca_x_id_vote.tsv", sep="\t")
    n_v.to_csv(od / "n_lca_x_id.tsv", sep="\t")
    log(TAG, f"ppF LCA x id (vote):\n{m_v}\ncounts:\n{n_v}")

    # ---- ortho vs para (best neighbour same species?)
    same = g == "species"
    # DIAMOND best_same_org cross-check
    from tracego import paths as Tpaths
    cells = pd.read_parquet(Tpaths.TV2 / "cells.parquet")
    cells = cells[cells.split == "tv"].drop_duplicates("id").set_index("id")
    if "best_same_org" in cells.columns:
        d_same = cells.best_same_org.reindex(ids).fillna(False).astype(bool).values
        log(TAG, f"same-species best nb {same.mean():.3f}; DIAMOND best_same_org {d_same.mean():.3f}; "
                 f"agreement {(same == d_same).mean():.3f}")
    ortho = {}
    for name, cur, ppf in (("vote", cur_v, ppf_v), ("tracego", cur_t, ppf_t)):
        rows = []
        for lab, mask in (("paralogue_like", same), ("orthologue_like", ~same & (g != "unmapped"))):
            sel = ids[mask]
            f = ev.fmax(cur, subset=list(sel))
            rows.append({"kind": lab, "n": int(mask.sum()), **{a: round(f[a], 4) for a in ("MF", "BP", "CC", "mean")},
                         "ppF": round(float(ppf.reindex(sel).mean()), 4)})
        ortho[name] = rows
        log(TAG, f"{name} ortho/para:\n{pd.DataFrame(rows).to_string(index=False)}")
    pd.DataFrame(ortho["vote"]).to_csv(od / "ortho_para_vote.tsv", sep="\t", index=False)

    # figure: Fmax vs LCA
    fig, ax = plt.subplots(figsize=(8, 4))
    xv = [r["group"] for r in tab_v]
    ax.plot(xv, [r["mean"] for r in tab_v], "o-", label="vote")
    ax.plot([r["group"] for r in tab_t], [r["mean"] for r in tab_t], "s-", label="tracego")
    for r in tab_v:
        ax.annotate(str(r["n"]), (r["group"], r["mean"]), textcoords="offset points", xytext=(0, 6), fontsize=7)
    ax.set_ylabel("IA-Fmax mean")
    ax.set_xlabel("LCA rank of best neighbour")
    ax.legend()
    ax.set_title("D2: TV Fmax vs taxonomic LCA of top-1 neighbour")
    fig.tight_layout()
    fig.savefig(od / "fmax_vs_lca.png", dpi=120)
    log(TAG, f"wrote {od / 'fmax_vs_lca.png'}")

    results = {"vote": s_v, "tracego": s_t, "lca_vote": tab_v, "lca_tracego": tab_t, "ortho": ortho,
               "fold_lca_agree": agree, "top1_score_p50": float(np.median(s0))}

    if a.skip_loo:
        save_json(results, od / "results.json")
        log(TAG, "skip-loo; done")
        return

    # ---- leave-one-clade-out
    log(TAG, "leave-one-clade-out: loading whitened h + residue codes")
    t0 = time.time()
    tr = train_rows(b, 0, 0)
    Yo = b.onto_labels("t0")
    o = ev.o
    from tracego import paths as Tpaths
    from tracego.analysis.common import cached
    from tracego.models.retrieval import maxsim_rerank
    codes = ResidueCodes(Tpaths.RUNS / "full" / "codes_l8_d64.npz", seqs=b.prot.seq.values)
    log(TAG, f"codes loaded in {time.time() - t0:.0f}s")

    loo = {}
    for cond, nlev in CONDITIONS.items():
        t1 = time.time()
        lists = []
        lost_best = 0
        for fold in (0, 1):
            rows_c, h = cached(fold, "h")
            pos = {int(r): i for i, r in enumerate(rows_c)}
            tr_i = np.array([pos[r] for r in tr if r in pos])
            # whiten returns the full whitened matrix fit on training rows present in this cache
            hw = whiten(np.asarray(h, np.float32), tr_i)
            hq = hw[np.array([pos[r] for r in tv])]
            hr = hw[np.array([pos[r] for r in tr])]
            qk = clade_keys(b, tv)[:nlev] if nlev else np.zeros((0, len(tv)), np.int64)
            rk = clade_keys(b, tr)[:nlev] if nlev else np.zeros((0, len(tr)), np.int64)
            I, S, n_drop = topk_cosine_clade(hq, hr, a.cands, b.ids[tv], b.ids[tr], qk, rk, device="cuda")
            cand = tr[I]
            Sc = maxsim_rerank(codes, tv, cand, "chainaa0.0")
            Ik, Sk = keep_top(cand, Sc, 50)
            lists.append((Ik, Sk))
            if fold == 0 and cond != "none":
                lost_best = int((Ik[:, 0] != n0).sum())
            log(TAG, f"  {cond} fold {fold}: mean dropped refs {float(n_drop.mean()):.0f}; "
                     f"top1 identity p50 {np.median(Sk[:, 0]):.3f}")
        P = ensemble_vote(lists, Yo, o, 0.05, 50)
        cur = ev.curves(P, ids, onto=True)
        s = ev.summary(cur)
        bins = {}
        for bn, mask in (("<50", ident < 50), ("50-80", (ident >= 50) & (ident < 80)),
                         ("80-95", (ident >= 80) & (ident < 95)), ("95-100", ident >= 95)):
            sel = ids[mask]
            if not len(sel):
                continue
            bins[bn] = {k: round(v, 4) for k, v in ev.fmax(cur, subset=list(sel)).items()
                        if k in ("MF", "BP", "CC", "mean")}
            bins[bn]["n"] = int(len(sel))
        loo[cond] = {"summary": s, "bins": bins, "lost_best": lost_best, "seconds": time.time() - t1}
        log(TAG, f"LOO {cond}: mean {s['mean']:.4f} test-like {s['mean_testlike']:.4f} "
                 f"lost_best {lost_best} in {loo[cond]['seconds']:.0f}s")
        torch.cuda.empty_cache()

    # drop vs none
    base = loo["none"]["summary"]["mean"]
    drops = {c: round(loo[c]["summary"]["mean"] - base, 4) for c in loo}
    log(TAG, f"LOO mean-Fmax drops vs none: {drops}")
    results["loo"] = loo
    results["loo_drops"] = drops
    save_json(results, od / "results.json")

    # bar chart of drops per id bin
    fig, ax = plt.subplots(figsize=(8, 4))
    bins_order = ("<50", "50-80", "80-95", "95-100")
    x = np.arange(len(bins_order))
    w = 0.25
    for i, cond in enumerate(("species", "genus", "family")):
        ys = [loo[cond]["bins"].get(bn, {}).get("mean", np.nan) - loo["none"]["bins"].get(bn, {}).get("mean", 0)
              for bn in bins_order]
        ax.bar(x + (i - 1) * w, ys, w, label=f"drop {cond}")
    ax.set_xticks(x)
    ax.set_xticklabels(bins_order)
    ax.set_ylabel("Δ IA-Fmax mean")
    ax.legend()
    ax.set_title("D2: leave-clade-out Fmax drop by identity bin")
    fig.tight_layout()
    fig.savefig(od / "loo_drop_by_id.png", dpi=120)
    log(TAG, f"wrote {od / 'loo_drop_by_id.png'}; disk free {disk_free_gb():.1f} GB")
    del codes
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
