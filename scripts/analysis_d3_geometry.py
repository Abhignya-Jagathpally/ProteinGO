"""Phase D3: UMAP/t-SNE, NMI/ARI/ASW, covariance spectra, GO-slim labels, GO DAG Laplacian eigenvectors.

  python scripts/analysis_d3_geometry.py
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

from tracego.analysis.common import FULL, disk_free_gb, log, out_dir
from tracego.analysis.geometry import cluster_metrics, effective_rank, goslim_labels, mf_class_labels
from tracego.data.bundle import Bundle
from tracego import paths
from tracego.train.common import fold_of, save_json, train_rows

TAG = "d3"


def parse_goslim(path) -> list[str]:
    terms, cur = [], None
    for line in open(path):
        if line.startswith("[Term]"):
            cur = {}
        elif line.startswith("id: ") and cur is not None:
            cur["id"] = line.split()[1].strip()
        elif line.startswith("subset:") and cur is not None and "goslim_generic" in line:
            if cur.get("id", "").startswith("GO:"):
                terms.append(cur["id"])
        elif line.strip() == "" and cur:
            cur = None
    return sorted(set(terms))


def go_laplacian_eigs(n_eigs: int = 64):
    """Normalised Laplacian eigenvectors of the undirected is_a/part_of graph, per GO namespace."""
    paths.cafa6_import()
    from cafa6 import knn_sharp as ks
    from collections import deque
    o = ks.Ontology()
    # re-parse immediate parents (Ontology keeps only the transitive closure in A)
    parents = {t: [] for t in o.terms}
    cur = None
    for line in open(ks.OBO):
        kv = line.strip().split(": ")
        if len(kv) < 2:
            continue
        k, v = kv[0], ": ".join(kv[1:])
        if k == "id":
            cur = v
        elif cur in parents and k == "is_a":
            p = v.split("!")[0].strip()
            if p in parents:
                parents[cur].append(p)
        elif cur in parents and k == "relationship" and v.startswith("part_of"):
            p = v.split()[1].strip()
            if p in parents:
                parents[cur].append(p)
    rows, cols = [], []
    for t, ps in parents.items():
        i = o.idx[t]
        for p in ps:
            j = o.idx[p]
            rows += [i, j]; cols += [j, i]
    n = len(o.terms)
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A.setdiag(0); A.eliminate_zeros()
    ns = {"F": "GO:0003674", "P": "GO:0008150", "C": "GO:0005575"}
    evecs = np.zeros((n, n_eigs), np.float32)
    filled, per_ns, info = 0, n_eigs // 3, {}
    for a, root in ns.items():
        seen, q = set(), deque([o.idx[root]])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            for v in A[u].indices:
                if v not in seen:
                    q.append(int(v))
        idx = np.array(sorted(seen))
        sub = A[idx][:, idx]
        d = np.asarray(sub.sum(1)).ravel()
        d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        L = sparse.eye(len(idx)) - sparse.diags(d_inv) @ sub @ sparse.diags(d_inv)
        k = min(per_ns + 1, max(2, len(idx) - 2))
        vals, vecs = eigsh(L.astype(np.float64), k=k, which="SM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        take = [i for i, v in enumerate(vals) if v > 1e-6][:per_ns]
        V = vecs[:, take]
        evecs[idx, filled:filled + V.shape[1]] = V.astype(np.float32)
        info[a] = {"n_nodes": int(len(idx)), "eigs": [float(vals[i]) for i in take], "n_eigs": int(V.shape[1])}
        filled += V.shape[1]
        log(TAG, f"Laplacian {a}: nodes {len(idx):,} eigs {V.shape[1]} first λ "
                 f"{vals[take[0]] if take else None}")
    return evecs[:, :filled], o, info


def main():
    od = out_dir("d3")
    slim_path = od / "goslim_generic.obo"
    log(TAG, f"disk free {disk_free_gb():.1f} GB; slim {slim_path.exists()}")
    b = Bundle()
    tr = train_rows(b, 0, 0)
    hold = tr[fold_of(tr) != 0]
    has, lab, cand = mf_class_labels(b, hold)
    rng = np.random.default_rng(0)
    pick = rng.choice(np.where(has)[0], min(6000, int(has.sum())), replace=False)
    rows, y_mf = hold[pick], lab[pick]
    log(TAG, f"holdout sample {len(rows):,}; MF classes {len(cand)}")

    # representations
    E_fr = np.load(paths.CACHE / "frozen35m_mean.npy", mmap_mode="r")
    run = FULL / "enc_lora35m_f0"
    pos = {int(r): i for i, r in enumerate(np.load(run / "rows.npy"))}
    zi = np.array([pos[r] for r in rows])
    reps = {
        "frozen": np.asarray(E_fr[rows], np.float32),
        "adapted": np.asarray(np.load(run / "h.npy", mmap_mode="r")[zi], np.float32),
        "latents": np.asarray(np.load(run / "Z.npy", mmap_mode="r")[zi], np.float32).mean(1),
    }
    # optional competitive latents
    cl = out_dir("c1") / "latents_holdout.npz"
    if cl.exists():
        z = np.load(cl)
        reps["competitive"] = np.asarray(z["Z"], np.float32).mean(1)

    slim_terms = parse_goslim(slim_path) if slim_path.exists() else []
    log(TAG, f"GO-slim terms parsed: {len(slim_terms)}")
    has_s, y_slim, kept = goslim_labels(b, rows, slim_terms, "F") if slim_terms else (np.zeros(len(rows), bool), None, [])
    log(TAG, f"GO-slim labelled proteins {int(has_s.sum())}/{len(rows)}; classes used {len(kept)}")

    metrics = {}
    spectra = {}
    fig, axes = plt.subplots(2, len(reps), figsize=(4.2 * len(reps), 8))
    if len(reps) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    import umap
    from sklearn.manifold import TSNE
    for j, (nm, X) in enumerate(reps.items()):
        m = cluster_metrics(X, y_mf)
        metrics[f"{nm}/MF"] = m
        log(TAG, f"{nm}/MF: NMI {m['NMI_mean']:.3f}±{m['NMI_sd']:.3f} ARI {m['ARI_mean']:.3f} "
                 f"ASW {m['ASW']:.3f} knn10 {m['knn10_acc']:.3f}")
        if has_s.any() and len(np.unique(y_slim[has_s])) > 1:
            ms = cluster_metrics(X[has_s], y_slim[has_s])
            metrics[f"{nm}/slim"] = ms
            log(TAG, f"{nm}/slim: NMI {ms['NMI_mean']:.3f} ASW {ms['ASW']:.3f} knn10 {ms['knn10_acc']:.3f}")
        er = effective_rank(X)
        spectra[nm] = {k: er[k] for k in ("eff_rank", "participation", "pcs90", "n_features")}
        log(TAG, f"{nm} spectrum: eff_rank {er['eff_rank']:.1f} PR {er['participation']:.1f} PCs@90% {er['pcs90']}")
        Xp = PCA(50, random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
        U = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=0).fit_transform(Xp)
        T = TSNE(2, perplexity=30, metric="cosine", random_state=0, init="pca").fit_transform(Xp)
        axes[0, j].scatter(U[:, 0], U[:, 1], c=y_mf, cmap="tab20", s=2)
        axes[0, j].set_title(f"UMAP {nm}")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].scatter(T[:, 0], T[:, 1], c=y_mf, cmap="tab20", s=2)
        axes[1, j].set_title(f"t-SNE {nm}")
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        # save eigs for spectrum plot
        spectra[nm]["eigs"] = er["eigs"][:200].tolist()

    fig.tight_layout()
    fig.savefig(od / "umap_tsne_grid.png", dpi=110)
    # spectrum figure
    fig2, ax = plt.subplots(figsize=(6, 4))
    for nm in reps:
        e = np.asarray(spectra[nm]["eigs"])
        ax.loglog(np.arange(1, len(e) + 1), e / e.sum(), label=nm)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("fraction of variance")
    ax.legend()
    ax.set_title("Embedding covariance spectrum")
    fig2.tight_layout()
    fig2.savefig(od / "spectrum.png", dpi=120)

    # Laplacian term embeddings
    log(TAG, "GO DAG Laplacian eigenvectors ...")
    evecs, o, lap_info = go_laplacian_eigs(64)
    # vocab rows
    vi = np.array([o.idx[t] for t in b.terms])
    L_emb = evecs[vi]
    # namespace recovery
    ns_lab = np.array([{"F": 0, "P": 1, "C": 2}[a] for a in b.aspect])
    # drop all-zero rows (terms outside components)
    ok = np.linalg.norm(L_emb, axis=1) > 1e-8
    m_ns = cluster_metrics(L_emb[ok], ns_lab[ok])
    # depth regression
    depth = o.depth[vi]
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), L_emb[ok], depth[ok], cv=5)
    r2 = float(r2_score(depth[ok], pred))
    rho = float(spearmanr(depth[ok], pred).correlation)
    # compare to text + learned
    text = np.load(paths.CACHE / "go_def_t5.npy")
    st = torch_load_term_emb()
    from sklearn.metrics.pairwise import cosine_similarity
    # CKA-ish: linear CKA between spaces on ok terms
    def cka(A, B):
        A, B = A - A.mean(0), B - B.mean(0)
        h = np.linalg.norm(A.T @ B, "fro") ** 2
        return float(h / (np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro") + 1e-30))
    lap_res = {"cluster_ns": m_ns, "depth_r2": r2, "depth_spearman": rho, "info": lap_info,
               "cka_lap_text": cka(L_emb[ok], text[ok]),
               "cka_lap_learned": cka(L_emb[ok], st[ok]) if st is not None else None,
               "cka_text_learned": cka(text[ok], st[ok]) if st is not None else None}
    log(TAG, f"Laplacian ns NMI {m_ns['NMI_mean']:.3f} knn {m_ns['knn10_acc']:.3f}; "
             f"depth R2 {r2:.3f} Spearman {rho:.3f}; CKA lap-text {lap_res['cka_lap_text']:.3f}")

    # scatter lap PC coloured by ns / depth
    P2 = PCA(2, random_state=0).fit_transform(L_emb[ok])
    fig3, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(P2[:, 0], P2[:, 1], c=ns_lab[ok], cmap="tab10", s=3)
    ax[0].set_title("Laplacian evecs by namespace")
    sc = ax[1].scatter(P2[:, 0], P2[:, 1], c=depth[ok], cmap="viridis", s=3)
    ax[1].set_title("Laplacian evecs by depth")
    fig3.colorbar(sc, ax=ax[1])
    fig3.tight_layout()
    fig3.savefig(od / "laplacian_scatter.png", dpi=120)

    save_json({"metrics": metrics, "spectra": {k: {kk: vv for kk, vv in v.items() if kk != "eigs"}
                                               for k, v in spectra.items()},
               "laplacian": lap_res}, od / "results.json")
    log(TAG, f"done; disk free {disk_free_gb():.1f} GB")


def torch_load_term_emb():
    import torch
    st = torch.load(FULL / "enc_lora35m_f0" / "model.pt", map_location="cpu", weights_only=False)
    te = st["state"].get("term_emb")
    return te.numpy() if te is not None else None


if __name__ == "__main__":
    main()
