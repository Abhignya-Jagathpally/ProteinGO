"""Final descriptive evaluation of TraceGO, its ablations and the repo baselines on TV v2, the GO-GPT temporal test and
fresh24 FR24-P. Nothing here selects anything: the TraceGO readout is the one with the best TV v2 test-like score, read
from the prediction summary. Nothing is appended to cafa6/runs/test_looks.md.

Outputs runs/final/: summary.tsv (IA-Fmax per namespace), per_bin.tsv, extended.json (Fmax, S-min, AUPR, AUC, MCC,
sparse-term AUPR), bootstrap.json (paired protein bootstrap, 1,000 reps, TraceGO minus baseline).

  python scripts/final_eval.py
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import BINS, Evaluator, load_knn_contract, load_pred_npz
from tracego.train.common import log

TAG = "final"
FULL = paths.RUNS / "full"
C6 = paths.CAFA6 / "runs"
OUT = paths.RUNS / "final"


def load_long(path, onto):
    df = pd.read_csv(path, sep="\t", header=None, names=["id", "term", "score"])
    df = df[df.term.isin(onto.idx)]
    ids = np.array(sorted(df.id.unique()))
    pos = {p: i for i, p in enumerate(ids)}
    P = sp.csr_matrix((df.score.values.astype(np.float32), (df.id.map(pos).values, df.term.map(onto.idx).values)),
                      shape=(len(ids), len(onto.terms)))
    return P, ids


def readout(tag):
    """Best TV v2 test-like readout of a decoder run (from its prediction summary)."""
    tv = {}
    for path in sorted(FULL.glob(f"dec_{tag}/pred_summary_*.json")):
        d = json.load(open(path))
        for k, v in d.items():
            if k.startswith("tv/") and "mean_testlike" in v:
                tv[k.split("/")[1]] = v["mean_testlike"]
    return max(tv, key=tv.get) if tv else None


def tracego_methods(bench):
    m = {}
    # Phase A frozen TraceGO = homology-curriculum decoder (mix readout: selected on TV-remote go criterion)
    r_cur = "mix"
    if not (FULL / f"dec_curriculum/pred_{bench}_{r_cur}.npz").exists():
        r_cur = readout("curriculum")
    if r_cur and (FULL / f"dec_curriculum/pred_{bench}_{r_cur}.npz").exists():
        m["TraceGO"] = FULL / f"dec_curriculum/pred_{bench}_{r_cur}.npz"
    r = readout("anchor")
    if r and (FULL / f"dec_anchor/pred_{bench}_{r}.npz").exists():
        key = "TraceGO (pre-A anchor)" if "TraceGO" in m else "TraceGO"
        m[key] = FULL / f"dec_anchor/pred_{bench}_{r}.npz"
    # Phase A hybrid retrieval (test = k-mer union; fr24p without) preferred over pre-A knn_ca
    hybrid = FULL / "knn_hybrid" / f"pred_{bench}.npz"
    pre = FULL / "knn_ca" / f"pred_{bench}.npz"
    if hybrid.exists():
        m["TraceGO retrieval only (embedding kNN vote)"] = hybrid
        if pre.exists():
            m["retrieval only (pre-A, no k-mer)"] = pre
    else:
        m["TraceGO retrieval only (embedding kNN vote)"] = pre
    m["retrieval only, frozen-encoder candidates"] = FULL / f"knn_frozen/pred_{bench}.npz"
    for tag, name in (("abl_frozen", "ablation: frozen ESM2 (no LoRA)"),
                      ("abl_noevid", "ablation: no retrieval evidence / copy"),
                      ("abl_free", "ablation: no hierarchy constraint"),
                      ("abl_nodef", "ablation: no definition-text init"),
                      ("abl_noia", "ablation: no IA weighting")):
        r = readout(tag)
        if r:
            m[name] = FULL / f"dec_{tag}/pred_{bench}_{r}.npz"
    m["ablation: classifier head only (no decoder)"] = FULL / f"dec_clf/pred_{bench}_clf.npz"
    m["ablation: unanchored decoder (dev. run)"] = FULL / f"dec_main/pred_{bench}_step0.npz"
    return {k: v for k, v in m.items() if v.exists()}


def baselines(bench, onto):
    out = {}
    if bench == "test":
        out["DIAMOND kNN train+val (repo)"] = lambda: load_knn_contract(paths.TV2 / "knn_trainval" / "test_knn.npz")
        out["HiGO-R2 (prelim)"] = lambda: load_pred_npz(C6 / "stacker_tv/l3_c0/pred_test_twin.npz")
        out["GO-GPT ten-sample"] = lambda: load_long(C6 / "gogpt_sample10/gogpt_sample10_freq.tsv.gz", onto)
        for k, f in (("BLAST-KNN", "blast_knn"), ("BLAST-KNN train+val", "blast_knn_trainval"),
                     ("SPROF-GO", "sprof_go")):
            out[k] = (lambda f=f: load_pred_npz(C6 / f"compare_paper/preds/{f}.npz"))
    elif bench == "fr24p":
        for k, f in (("SPROF-GO", "sprof_go"), ("HiGO-R2 / Ontologue (prelim)", "ontologue"),
                     ("Ontologue kNN", "ontologue_knn"), ("GO-GPT ten-sample", "gogpt_sample10"),
                     ("BLAST-KNN train+val", "blast_knn_trainval")):
            out[k] = (lambda f=f: load_pred_npz(C6 / f"fresh24/preds/{f}.npz"))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = Bundle()
    rows_s, rows_b, ext, boot = [], [], {}, {}
    for bench in ("tv", "test", "fr24p"):
        ev = Evaluator(bench)
        o = ev.o
        # sparse-term groups by training prevalence (references of this benchmark)
        Yo = b.onto_labels("t0" if bench == "tv" else "full")
        ref = b.rows("train" if bench == "tv" else "ref_full")
        prev = np.asarray((Yo[ref] > 0).sum(0)).ravel()
        groups = {"prev<20": np.where((prev >= 1) & (prev < 20))[0], "prev20-100": np.where((prev >= 20) & (prev <= 100))[0],
                  "prev0": np.where(prev == 0)[0]}
        meths = {k: (lambda p=p: load_pred_npz(p)) for k, p in tracego_methods(bench).items()}
        meths.update(baselines(bench, o))
        curs = {}
        for name, loader in meths.items():
            try:
                P, ids = loader()
            except FileNotFoundError as e:
                log(TAG, f"{bench} {name}: missing ({e})")
                continue
            cur = ev.curves(P, ids, onto=True)
            curs[name] = (cur, P, ids)
            s = ev.summary(cur)
            pb = ev.per_bin(cur)
            rows_s.append({"bench": bench, "method": name, **{k: round(v, 4) for k, v in s.items()}})
            rows_b.append({"bench": bench, "method": name, **{k: (round(v[1], 4) if v[0] else np.nan)
                                                                  for k, v in pb.items()}})
            log(TAG, f"{bench:5s} {name:48s} MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} mean {s['mean']:.4f}"
                     + (f" test-like {s['mean_testlike']:.4f}" if "mean_testlike" in s else ""))
        if bench == "tv":
            n = {k: v[0] for k, v in ev.per_bin(curs["TraceGO retrieval only (embedding kNN vote)"][0]).items()}
            log(TAG, f"tv identity-bin sizes {n}")
            if "TraceGO" in curs:
                for w in (False, True):
                    r = ev.bootstrap(curs["TraceGO"][0], curs["TraceGO retrieval only (embedding kNN vote)"][0],
                                     reps=1000, weighted=w)
                    boot[f"tv{' test-like' if w else ''}: TraceGO - retrieval only"] = r
                    log(TAG, f"tv{' test-like' if w else ''} bootstrap TraceGO - retrieval only: {r['diff']:+.4f} "
                             f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
            continue
        n = {k: v[0] for k, v in ev.per_bin(next(iter(curs.values()))[0]).items()}
        log(TAG, f"{bench} identity-bin sizes {n}")
        best = {"test": ["DIAMOND kNN train+val (repo)", "HiGO-R2 (prelim)"],
                "fr24p": ["SPROF-GO", "HiGO-R2 / Ontologue (prelim)"]}[bench]
        for a_name in ("TraceGO", "TraceGO retrieval only (embedding kNN vote)"):
            if a_name not in curs:
                continue
            for b_name in best:
                if b_name in curs:
                    r = ev.bootstrap(curs[a_name][0], curs[b_name][0], reps=1000)
                    boot[f"{bench}: {a_name} - {b_name}"] = r
                    log(TAG, f"{bench} bootstrap {a_name} - {b_name}: {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        for name in ["TraceGO", "TraceGO retrieval only (embedding kNN vote)"] + best:
            if name in curs:
                _, P, ids = curs[name]
                e = ev.extended(P, ids, onto=True, sparse_terms=groups)
                ext[f"{bench}: {name}"] = e
                log(TAG, f"{bench} extended {name}: Fmax(unweighted) {e['Fmax_mean']:.4f} S-min(IA) {e['Smin_w_mean']:.3f} "
                         f"AUPR {e['AUPR_all_mean']:.4f} AUC {e['AUC_all_mean']:.4f} MCC {e['MCC_mean']:.4f} "
                         f"AUPR prev<20 {e.get('AUPR_prev<20_mean', np.nan):.4f} prev20-100 "
                         f"{e.get('AUPR_prev20-100_mean', np.nan):.4f}")
        pd.DataFrame(rows_s).to_csv(OUT / "summary.tsv", sep="\t", index=False)
        pd.DataFrame(rows_b).to_csv(OUT / "per_bin.tsv", sep="\t", index=False)
        json.dump(ext, open(OUT / "extended.json", "w"), indent=1)
        json.dump(boot, open(OUT / "bootstrap.json", "w"), indent=1)
    pd.DataFrame(rows_s).to_csv(OUT / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(rows_b).to_csv(OUT / "per_bin.tsv", sep="\t", index=False)
    log(TAG, f"wrote {OUT}")


if __name__ == "__main__":
    main()
