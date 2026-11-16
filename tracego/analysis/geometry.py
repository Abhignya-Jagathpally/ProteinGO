"""Phase D3 helpers: clustering metrics, effective rank, GO-slim labels, Laplacian term embeddings."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def effective_rank(X: np.ndarray) -> dict:
    """Covariance eigenvalue spectrum summaries (Roy–Vetterli entropy effective rank, participation ratio, PCs@90%)."""
    X = X - X.mean(0)
    # economy SVD on centred data
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = (s ** 2) / max(len(X) - 1, 1)
    p = ev / ev.sum()
    # entropy effective rank
    ent = -np.sum(p * np.log(p + 1e-30))
    erank = float(np.exp(ent))
    pr = float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-30))
    c = np.cumsum(p)
    pcs90 = int(np.searchsorted(c, 0.9) + 1)
    return {"eff_rank": erank, "participation": pr, "pcs90": pcs90, "n_features": int(X.shape[1]),
            "eigs": ev.astype(np.float64)}


def cluster_metrics(X: np.ndarray, lab: np.ndarray, n_seeds: int = 3) -> dict:
    """k-means NMI/ARI (mean±sd over seeds), ASW (cosine on PCA-50, scIB-style (s+1)/2), 10-NN label accuracy."""
    Xp = PCA(min(50, X.shape[0] - 1, X.shape[1]), random_state=0).fit_transform(
        (X - X.mean(0)) / (X.std(0) + 1e-6))
    k = len(np.unique(lab))
    nmis, aris = [], []
    for s in range(n_seeds):
        km = KMeans(k, n_init=10, random_state=s).fit_predict(Xp)
        nmis.append(normalized_mutual_info_score(lab, km))
        aris.append(adjusted_rand_score(lab, km))
    sil = silhouette_score(Xp, lab, metric="cosine")
    asw = (sil + 1) / 2
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(Xp)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    pred = np.array([np.bincount(lab[i], minlength=lab.max() + 1).argmax() for i in idx])
    knn_acc = float((pred == lab).mean())
    return {"NMI_mean": float(np.mean(nmis)), "NMI_sd": float(np.std(nmis)),
            "ARI_mean": float(np.mean(aris)), "ARI_sd": float(np.std(aris)),
            "ASW": float(asw), "silhouette_raw": float(sil), "knn10_acc": knn_acc, "n": int(len(lab)), "k": k}


def mf_class_labels(b, rows, n_classes: int = 15, seed: int = 0):
    """Most-specific of the n_classes most frequent mid-depth MF terms (extras.py recipe)."""
    Y = b.Y_t0
    mf = np.where(b.aspect == "F")[0]
    n_anc = np.asarray(b.vocab.n_anc.values)
    prev = np.asarray((Y[rows][:, mf] > 0).sum(0)).ravel()
    cand = mf[(n_anc[mf] >= 3) & (n_anc[mf] <= 5)]
    cand = cand[np.argsort(-prev[np.searchsorted(mf, cand)])][:n_classes]
    Yc = Y[rows][:, cand].toarray() > 0
    has = Yc.any(1)
    spec = np.where(Yc, b.ia[cand][None, :], -1).argmax(1)
    return has, spec, cand


def goslim_labels(b, rows, slim_terms: list[str], aspect: str = "F"):
    """Map each protein to its most specific GO-slim ancestor among its annotations in `aspect`.
    Returns (has_label mask, label indices into slim_terms, slim_terms kept)."""
    from tracego import paths
    paths.cafa6_import()
    from cafa6 import knn_sharp as ks
    o = ks.Ontology()
    # ancestors: for each vocab term, which slim terms are ancestors (or self)
    slim_idx = {t: i for i, t in enumerate(slim_terms) if t in o.idx}
    kept = [t for t in slim_terms if t in slim_idx]
    slim_i = np.array([o.idx[t] for t in kept])
    # depth = n ancestors
    depth = np.asarray(o.A.sum(1)).ravel() if hasattr(o, "A") else np.zeros(len(o.terms))
    # for each vocab term -> best slim ancestor
    vocab_to_slim = np.full(len(b.terms), -1, np.int32)
    for j, t in enumerate(b.terms):
        if t not in o.idx:
            continue
        ti = o.idx[t]
        # ancestors including self via A or parents closure
        # Ontology.A is ancestor matrix (term x ancestor)? check — usually A[i,j]=1 if j ancestor of i
        anc = np.where(o.A[ti].toarray().ravel() > 0)[0] if hasattr(o.A[ti], "toarray") else np.where(o.A[ti] > 0)[0]
        anc = np.unique(np.concatenate([anc, [ti]]))
        hit = [k for k, si in enumerate(slim_i) if si in set(anc)]
        if not hit:
            continue
        # most specific = max depth
        vocab_to_slim[j] = max(hit, key=lambda k: depth[slim_i[k]])
    asp = np.where(b.aspect == aspect)[0]
    Y = b.Y_t0[rows][:, asp].tocsr()
    lab = np.full(len(rows), -1, np.int32)
    for i in range(len(rows)):
        terms = asp[Y[i].indices]
        cands = [vocab_to_slim[t] for t in terms if vocab_to_slim[t] >= 0]
        if cands:
            # most specific slim among carried terms
            lab[i] = max(cands, key=lambda k: depth[slim_i[k]])
    has = lab >= 0
    return has, lab, kept
