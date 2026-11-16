"""Scoring through the repo's official machinery (`cafa6.knn_sharp.Scorer`, IA-weighted Fmax, prop=max, norm=cafa).

Benchmarks
  tv     TV v2 (7,566; gt_tv.tsv). Selection data. Plain mean and test-like weighted mean (cell weights).
  val    GO-GPT random validation split (convergence monitoring only).
  test   GO-GPT temporal test (8,630). Descriptive only; nothing is logged to runs/test_looks.md.
  fr24p  fresh24 FR24-P (11,231; gt_nk.tsv, hpa_only False). Descriptive only.

Predictions are CSR [n, 7901] over the bundle vocabulary, or CSR over the full ontology (`onto=True`).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tracego import paths

paths.cafa6_import()
from cafa6 import knn_sharp as ks  # noqa: E402
from cafa6 import score_fresh24 as sf  # noqa: E402
from cafa6.compare_paper import load_npz_pred  # noqa: E402

ASP = ("MF", "BP", "CC")
_ONTO = None


def ontology() -> "ks.Ontology":
    global _ONTO
    if _ONTO is None:
        _ONTO = ks.Ontology()
    return _ONTO


def ground_truth(bench: str) -> pd.DataFrame:
    if bench in ("tv", "tvr"):  # TV-remote shares TV v2 ground truth; only the references are masked
        return pd.read_csv(paths.TV2 / "gt_tv.tsv", sep="\t", header=None, names=["id", "term"])
    if bench in ("val", "test"):
        return pd.read_csv(paths.PROC / f"gt_{bench}.tsv", sep="\t", header=None, names=["id", "term"])
    if bench == "fr24p":
        p = pd.read_csv(paths.FRESH / "proteins.tsv", sep="\t")
        prim = set(p.accession[~p.hpa_only.astype(bool)])
        nk = pd.read_csv(paths.FRESH / "gt_nk.tsv", sep="\t")
        return nk[nk.protein.isin(prim)].rename(columns={"protein": "id"})[["id", "term"]].drop_duplicates()
    raise KeyError(bench)


def identity(bench: str) -> pd.Series:
    """Best DIAMOND identity to the reference set, for the descriptive per-bin breakdown only."""
    if bench in ("tv", "tvr"):
        c = pd.read_parquet(paths.TV2 / "cells.parquet")
        c = c[c.split == "tv"].drop_duplicates("id")
        return c.set_index("id").max_pident
    if bench == "test":
        z = np.load(paths.TV2 / "knn_trainval" / "test_knn.npz", allow_pickle=False)
        return pd.Series(z["max_pident"], index=z["ids"].astype(str))
    if bench == "fr24p":
        p = pd.read_csv(paths.FRESH / "proteins.tsv", sep="\t")
        return p.set_index("accession").best_identity
    raise KeyError(bench)


def bin_of(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.select([x <= 0, x < 30, x < 50, x < 80, x < 95, x < 100], ["no_hit", "<30", "30-50", "50-80", "80-95",
                                                                        "95-99.9"], "100")


BINS = ["no_hit", "<30", "30-50", "50-80", "80-95", "95-99.9", "100"]


class Evaluator:
    def __init__(self, bench: str):
        self.bench = bench
        self.o = ontology()
        self.gt = ground_truth(bench)
        self.sc = ks.Scorer(self.o, self.gt)
        self.w = None
        if bench in ("tv", "tvr"):
            cells = pd.read_parquet(paths.TV2 / "cells.parquet")
            self.w, _ = ks.cell_weights(cells, self.sc.ids)
        vocab = pd.read_csv(paths.BUNDLE / "vocab.tsv", sep="\t", usecols=["term"])
        self.vcols = np.array([self.o.idx[t] for t in vocab.term])

    def to_onto(self, P: sp.csr_matrix) -> sp.csr_matrix:
        P = P.tocoo()
        return sp.csr_matrix((P.data, (P.row, self.vcols[P.col])), shape=(P.shape[0], len(self.o.terms)))

    def curves(self, P: sp.csr_matrix, ids, onto: bool = False) -> dict:
        if not onto:
            P = self.to_onto(P)
        return self.sc.curves(P, np.asarray(ids).astype(str))

    def fmax(self, cur: dict, weighted: bool = False, subset=None) -> dict:
        return self.sc.fmax(cur, subset=subset, weights=self.w if weighted else None)

    def summary(self, cur: dict) -> dict:
        r = {f"{a}": v for a, v in self.fmax(cur).items()}
        if self.w is not None:
            r["mean_testlike"] = self.fmax(cur, weighted=True)["mean"]
        return r

    def bootstrap(self, curA, curB, reps=1000, seed=0, weighted=False):
        return self.sc.bootstrap(curA, curB, reps=reps, seed=seed, weights=self.w if weighted else None)

    def per_bin(self, cur: dict) -> dict:
        ident = identity(self.bench).reindex(self.sc.ids).fillna(0).values
        b = bin_of(ident)
        out = {}
        for name in BINS:
            ids = self.sc.ids[b == name]
            out[name] = (len(ids), self.sc.fmax(cur, subset=ids)["mean"] if len(ids) else np.nan)
        return out

    # ---- extended metrics (S-min, unweighted Fmax, AUPR, AUC, MCC) ----
    def extended(self, P: sp.csr_matrix, ids, onto: bool = False, sparse_terms: dict | None = None) -> dict:
        if not onto:
            P = self.to_onto(P)
        S = sf.processed(self.sc, np.asarray(ids).astype(str), P)
        st = sf.build_stats(self.sc, S)
        W = np.ones((1, self.sc.n))
        ev = sf.evaluate(st, W)
        res = {}
        for a in ASP + ("mean",):
            res[f"Fmax_w_{a}"] = float(ev[a]["Fmax_w"][0])
            res[f"Fmax_{a}"] = float(ev[a]["Fmax"][0])
            res[f"Smin_w_{a}"] = float(ev[a]["Smin_w"][0])
        res.update(self.term_metrics(S, sparse_terms))
        return res

    def term_metrics(self, S: sp.csr_matrix, sparse_terms: dict | None = None) -> dict:
        """Term-centric AUPR and ROC-AUC (macro over terms with >= 1 positive and >= 1 negative among proteins with
        ground truth in the namespace), and protein-centric MCC at the best-F threshold (micro over pairs)."""
        from sklearn.metrics import average_precision_score, roc_auc_score
        G = self.sc.G.tocsc()
        Sc = S.tocsc()
        out = {}
        groups = {"all": None}
        if sparse_terms:
            groups.update(sparse_terms)
        for k, a in enumerate(ASP):
            rows = np.where(self.sc.in_ns[k])[0]
            cols = np.where(self.o.ns == k)[0]
            pos = np.asarray(G[rows][:, cols].sum(0)).ravel()
            cols = cols[(pos >= 1) & (pos < len(rows))]
            Gd = G[rows][:, cols].toarray().astype(bool)
            Sd = Sc[rows][:, cols].toarray()
            for gname, gset in groups.items():
                m = np.ones(len(cols), bool) if gset is None else np.isin(cols, gset)
                ap, auc = [], []
                for j in np.where(m)[0]:
                    y, s = Gd[:, j], Sd[:, j]
                    ap.append(average_precision_score(y, s))
                    auc.append(roc_auc_score(y, s))
                out[f"AUPR_{gname}_{a}"] = float(np.mean(ap)) if ap else np.nan
                out[f"AUC_{gname}_{a}"] = float(np.mean(auc)) if auc else np.nan
                out[f"nterms_{gname}_{a}"] = int(m.sum())
            best = -1
            for tau in np.arange(0.05, 1.0, 0.05):
                pr = Sd >= tau
                tp = (pr & Gd).sum(); fp = (pr & ~Gd).sum(); fn = (~pr & Gd).sum(); tn = Gd.size - tp - fp - fn
                den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
                best = max(best, mcc)
            out[f"MCC_{a}"] = float(best)
        for key in [k for k in list(out) if k.endswith("_MF")]:
            base = key[:-3]
            vals = [out.get(f"{base}_{a}") for a in ASP]
            if all(isinstance(v, float) for v in vals):
                out[f"{base}_mean"] = float(np.nanmean(vals))
        return out


def load_pred_npz(path) -> tuple[sp.csr_matrix, np.ndarray]:
    """Repo prediction contract (ids, data, indices, indptr, shape, terms) -> (P over ontology columns, ids)."""
    return load_npz_pred(path, ontology())


def load_knn_contract(path, full: bool = True) -> tuple[sp.csr_matrix, np.ndarray]:
    """cafa6 kNN prior file (p_* over vocab, pf_* over terms_full) -> (P over ontology columns, ids)."""
    z = np.load(path, allow_pickle=False)
    o = ontology()
    if full:
        Q = sp.csr_matrix((z["pf_data"], z["pf_indices"], z["pf_indptr"]), shape=tuple(z["pf_shape"]))
        terms = z["terms_full"].astype(str)
    else:
        Q = sp.csr_matrix((z["p_data"], z["p_indices"], z["p_indptr"]), shape=tuple(z["p_shape"]))
        terms = pd.read_csv(paths.BUNDLE / "vocab.tsv", sep="\t", usecols=["term"]).term.values
    cols = np.array([o.canon(t)[0] if o.canon(t) else -1 for t in terms])
    keep = cols >= 0
    Q = Q[:, np.where(keep)[0]].tocoo()
    return sp.csr_matrix((Q.data, (Q.row, cols[keep][Q.col])), shape=(Q.shape[0], len(o.terms))), z["ids"].astype(str)


def save_pred(path, P: sp.csr_matrix, ids, terms):
    """Writes the repo prediction contract."""
    P = P.tocsr()
    np.savez_compressed(path, ids=np.asarray(ids).astype(str), data=P.data.astype(np.float32), indices=P.indices,
                        indptr=P.indptr, shape=np.array(P.shape), terms=np.asarray(terms).astype(str))
