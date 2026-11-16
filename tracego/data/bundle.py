"""One on-disk bundle with every protein, label matrix, vocabulary DAG, GO definition and taxon lineage.

Rows: the 141,684 rows of `proc_gogpt_tv2/train.parquet` (train / val / test / tv, same order) followed by the
24,141 fresh24 proteins. Test accessions that also occur in train/val keep one row per split, as upstream.

Label matrices (CSR over the 7,901-term vocabulary, vocab.tsv order):
  Y_t0    T0-stripped propagated labels of the v2 training rows (split == "train"); every other row is empty.
          This is the training target and the reference label set for TV v2 queries.
  Y_full  full propagated GO-GPT labels of train + val rows (`proc_gogpt/Y.npz`); every other row is empty.
          Reference label set for the GO-GPT test and fresh24 queries (the repo's train+val kNN references).
Test rows of both upstream Y files contain test labels; they are dropped here and never read again.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tracego import paths

RANKS = ("species", "genus", "family", "order", "class", "phylum", "kingdom", "superkingdom")
ASPECTS = ("F", "P", "C")
ASPECT_NAME = {"F": "molecular function", "P": "biological process", "C": "cellular component"}
ROOTS = {"F": "GO:0003674", "P": "GO:0008150", "C": "GO:0005575"}


def log(msg: str):
    print(f"[bundle {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_csr(path: Path) -> sp.csr_matrix:
    z = np.load(path, allow_pickle=False)
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def parse_definitions(obo: Path) -> dict:
    """{GO id: (name, definition)} from the OBO, definition stripped of its reference list."""
    out, cur = {}, {}
    for line in open(obo, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]" or line.startswith("["):
            if cur.get("id"):
                out[cur["id"]] = (cur.get("name", ""), cur.get("def", ""))
            cur = {}
        elif line.startswith("id: "):
            cur["id"] = line[4:]
        elif line.startswith("name: "):
            cur["name"] = line[6:]
        elif line.startswith("def: "):
            m = re.match(r'def: "(.*)"', line)
            cur["def"] = m.group(1) if m else line[5:]
    if cur.get("id"):
        out[cur["id"]] = (cur.get("name", ""), cur.get("def", ""))
    return out


def ncbi_taxids(prot: pd.DataFrame) -> np.ndarray:
    """NCBI taxon per row. The v2 `taxon` column is an internal organism index, so the GO-GPT rows are mapped from
    the organism name (scientific name, then any other NCBI name class, then 'Genus species', then the genus);
    fresh24 rows already carry NCBI taxa. 0 when nothing matches."""
    names = pd.read_csv(paths.TAXDUMP / "names.dmp", sep="|", header=None, usecols=[0, 1, 3],
                        names=["tax", "name", "cls"], dtype=str)
    names["name"] = names.name.str.strip()
    names["cls"] = names.cls.str.strip()
    sci = names[names.cls == "scientific name"]
    m_sci = dict(zip(sci.name, sci.tax.astype(int)))
    other = names[names.cls != "scientific name"].drop_duplicates("name")
    m_oth = dict(zip(other.name, other.tax.astype(int)))
    org = prot.organism.fillna("").str.split(" (", regex=False).str[0]
    out = np.zeros(len(prot), np.int64)
    cache = {}
    for i, o in enumerate(org.values):
        if o not in cache:
            w = o.split()
            cache[o] = (m_sci.get(o) or m_oth.get(o) or m_sci.get(" ".join(w[:2])) or
                        (m_sci.get(w[0]) if w else None) or 0)
        out[i] = cache[o]
    fr = (prot.split == "fresh24").values
    out[fr] = prot.taxon.values[fr]
    return out


def taxon_lineages(taxa: np.ndarray) -> pd.DataFrame:
    """[taxon, species, genus, ..., superkingdom] (NCBI taxdump; 0 when a rank is absent)."""
    nodes = pd.read_csv(paths.TAXDUMP / "nodes.dmp", sep="|", header=None, usecols=[0, 1, 2],
                        names=["tax", "parent", "rank"], dtype={0: np.int64, 1: np.int64, 2: str})
    nodes["rank"] = nodes["rank"].str.strip()
    merged = pd.read_csv(paths.TAXDUMP / "merged.dmp", sep="|", header=None, usecols=[0, 1], names=["old", "new"])
    remap = dict(zip(merged.old.astype(int), merged.new.astype(int)))
    parent = dict(zip(nodes.tax.values, nodes.parent.values))
    rank = dict(zip(nodes.tax.values, nodes["rank"].values))
    rows = []
    for t in np.unique(taxa):
        r = {"taxon": int(t)}
        x = remap.get(int(t), int(t))
        seen = 0
        while x in parent and seen < 100:
            rk = rank[x]
            if rk in RANKS and rk not in r:
                r[rk] = x
            if rk in ("domain",) and "superkingdom" not in r:
                r["superkingdom"] = x
            if parent[x] == x:
                break
            x = parent[x]
            seen += 1
        rows.append(r)
    df = pd.DataFrame(rows).fillna(0)
    for k in RANKS:
        if k not in df:
            df[k] = 0
    return df[["taxon", *RANKS]].astype(np.int64)


def build_full_terms(prot: pd.DataFrame, out: Path):
    """Label matrices over every ontology term (cafa6 Ontology column order), not only the vocabulary: the
    out-of-vocabulary terms carry about 0.01 of TV v2 IA-Fmax and are scored through the copy channel.
    Yo_t0 = T0 labels of v2 training rows; Yo_full = full labels of train + val rows."""
    paths.cafa6_import()
    from cafa6 import knn_sharp as ks
    from cafa6.temporal_val import GO, full_labels, t0_labels
    go, onto = GO(), ks.Ontology()
    full_tr, full_va = full_labels(go, "train"), full_labels(go, "val")
    t0 = t0_labels(go, full_tr, out=paths.TV2)
    ids, split = prot.id.values, prot.split.values
    sets_t0 = [t0.get(p, set()) if s == "train" else set() for p, s in zip(ids, split)]
    sets_full = [full_tr.get(p, set()) if s in ("train", "tv") else full_va.get(p, set()) if s == "val" else set()
                 for p, s in zip(ids, split)]
    Yo_t0, Yo_full = onto.label_matrix(sets_t0), onto.label_matrix(sets_full)
    # v2 "tv" rows are training proteins whose post-T0 labels are the TV targets; their full labels are only
    # references for test/fresh24 (GO-GPT train + val), never for TV queries.
    sp.save_npz(out / "Yo_t0.npz", Yo_t0)
    sp.save_npz(out / "Yo_full.npz", Yo_full)
    np.save(out / "onto_terms.npy", np.array(onto.terms))
    log(f"full-ontology labels: {len(onto.terms):,} terms; Yo_t0 nnz {Yo_t0.nnz:,} "
        f"(terms used {int((np.diff(Yo_t0.tocsc().indptr) > 0).sum()):,}); Yo_full nnz {Yo_full.nnz:,}")


def build(out: Path = paths.BUNDLE):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(paths.TV2 / "train.parquet")
    df["row"] = np.arange(len(df))
    log(f"v2 rows {len(df):,}: {df.split.value_counts().to_dict()}")
    fp = pd.read_csv(paths.FRESH / "proteins.tsv", sep="\t")
    fs = pd.read_parquet(paths.FRESH / "work" / "seqs.parquet").set_index("accession").seq
    fr = pd.DataFrame({"id": fp.accession.values, "taxon": fp.taxon.values.astype(np.int64),
                       "seq": fs.reindex(fp.accession).values, "length": fp.length.values,
                       "split": "fresh24", "cluster": "", "organism": fp.organism.values, "tv": False,
                       "tv_stratum": ""})
    assert fr.seq.notna().all()
    prot = pd.concat([df.drop(columns=["row"]), fr], ignore_index=True)
    prot["fr24p"] = False
    nk = set(pd.read_csv(paths.FRESH / "gt_nk.tsv", sep="\t").protein)
    prot.loc[prot.split == "fresh24", "fr24p"] = (~fp.hpa_only.astype(bool) & fp.accession.isin(nk)).values
    prot["ncbi_taxon"] = ncbi_taxids(prot)
    log(f"NCBI taxon mapped for {(prot.ncbi_taxon > 0).mean():.3f} of rows")
    tv_ids = open(paths.TV2 / "tv_ids.txt").read().split()
    # TV query row: the first v2 row of the accession with tv == True
    prot["tv_query"] = False
    tvrow = prot[(prot.tv) & prot.split.isin(["train", "tv"])].drop_duplicates("id")
    prot.loc[tvrow.index[tvrow.id.isin(set(tv_ids))], "tv_query"] = True
    log(f"proteins {len(prot):,}; tv queries {prot.tv_query.sum():,}/{len(tv_ids):,}; fr24p {prot.fr24p.sum():,}; "
        f"seq len p50 {prot.length.median():.0f} p99 {prot.length.quantile(.99):.0f}")
    prot.to_parquet(out / "proteins.parquet")

    vocab = pd.read_csv(paths.TV2 / "vocab.tsv", sep="\t")
    n = len(prot)
    Yv2 = _load_csr(paths.TV2 / "Y.npz")
    keep = np.zeros(n, bool)
    keep[: len(df)] = (df.split == "train").values
    Y_t0 = sp.diags(keep[: len(df)].astype(np.float32)) @ Yv2
    Y_t0 = sp.vstack([Y_t0, sp.csr_matrix((len(fr), Yv2.shape[1]), dtype=np.float32)]).tocsr()
    Y_t0.eliminate_zeros()
    Yf = _load_csr(paths.PROC / "Y.npz")
    pg = pd.read_parquet(paths.PROC / "train.parquet", columns=["id", "split"])
    assert (pg.id.values == df.id.values).all()
    kf = pg.split.isin(["train", "val"]).values
    Y_full = sp.diags(kf.astype(np.float32)) @ Yf
    Y_full = sp.vstack([Y_full, sp.csr_matrix((len(fr), Yf.shape[1]), dtype=np.float32)]).tocsr()
    Y_full.eliminate_zeros()
    assert Y_t0[np.where(prot.split.values == "test")[0]].nnz == 0
    assert Y_full[np.where(prot.split.values == "test")[0]].nnz == 0
    sp.save_npz(out / "Y_t0.npz", Y_t0)
    sp.save_npz(out / "Y_full.npz", Y_full)
    log(f"Y_t0 rows with labels {int((np.diff(Y_t0.indptr) > 0).sum()):,} nnz {Y_t0.nnz:,}; "
        f"Y_full rows {int((np.diff(Y_full.indptr) > 0).sum()):,} nnz {Y_full.nnz:,}")

    build_full_terms(prot, out)
    anc = np.load(paths.TV2 / "go_ancestors.npy")
    np.save(out / "go_ancestors.npy", anc)
    defs = parse_definitions(paths.OBO)
    vocab["name"] = [defs.get(t, ("", ""))[0] for t in vocab.term]
    vocab["definition"] = [defs.get(t, ("", ""))[1] for t in vocab.term]
    vocab["n_anc"] = np.bincount(anc[:, 0], minlength=len(vocab))
    log(f"vocab {len(vocab):,} terms {vocab.aspect.value_counts().to_dict()}; ancestor pairs {len(anc):,}; "
        f"missing definitions {(vocab.definition == '').sum()}")
    vocab.to_csv(out / "vocab.tsv", sep="\t", index=False)
    lin = taxon_lineages(prot.ncbi_taxon.values)
    lin.to_parquet(out / "taxa.parquet")
    log(f"taxa {len(lin):,}; with genus {int((lin.genus > 0).sum()):,}; with family {int((lin.family > 0).sum()):,}")
    for i in range(2):
        r = prot.iloc[i]
        log(f"sample row {i}: {r.id} {r.split} taxon {r.taxon} ({r.organism[:40]}) len {r.length} "
            f"seq {r.seq[:40]}... T0 terms {Y_t0[i].nnz}")
    json.dump({"n": int(n), "splits": prot.split.value_counts().to_dict(), "seconds": round(time.time() - t0, 1)},
              open(out / "report.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


class Bundle:
    """Read-only view of the bundle."""

    def __init__(self, d: Path = paths.BUNDLE):
        self.d = d
        self.prot = pd.read_parquet(d / "proteins.parquet")
        self.vocab = pd.read_csv(d / "vocab.tsv", sep="\t", keep_default_na=False)
        self.Y_t0 = sp.load_npz(d / "Y_t0.npz").tocsr()
        self.Y_full = sp.load_npz(d / "Y_full.npz").tocsr()
        self.anc = np.load(d / "go_ancestors.npy")
        self.taxa = pd.read_parquet(d / "taxa.parquet").set_index("taxon")
        self.ids = self.prot.id.to_numpy(dtype=str)
        self.terms = self.vocab.term.to_numpy(dtype=str)
        self.aspect = self.vocab.aspect.values
        self.ia = self.vocab.ia.values.astype(np.float32)

    def onto_labels(self, which: str) -> sp.csr_matrix:
        """Full-ontology label matrix: which in {"t0", "full"}."""
        return sp.load_npz(self.d / f"Yo_{which}.npz").tocsr()

    def labels(self, which: str) -> sp.csr_matrix:
        return {"t0": self.Y_t0, "full": self.Y_full}[which]

    def rows(self, name: str) -> np.ndarray:
        p = self.prot
        sel = {"train": p.split == "train", "val": p.split == "val", "test": p.split == "test",
               "tv": p.tv_query, "fr24p": p.fr24p, "fresh24": p.split == "fresh24",
               # the repo's train + val kNN reference set (133,054 accessions: GO-GPT train incl. the v2
               # "tv" rows, plus val), used with Y_full for test and fresh24 queries
               "ref_full": p.split.isin(["train", "tv", "val"])}[name]
        return np.where(sel.values)[0]

    def lineage(self, rows: np.ndarray) -> np.ndarray:
        """[len(rows), 8] taxon ids per rank (RANKS order)."""
        tx = self.prot.ncbi_taxon.values[rows]
        return self.taxa.reindex(tx).fillna(0).values.astype(np.int64)


if __name__ == "__main__":
    build()
