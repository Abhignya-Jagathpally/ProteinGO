"""Embedding geometry before vs after the adapter: UMAP, k-means NMI / ARI and silhouette against molecular-function
classes, and linear CKA between frozen and adapted representations.

Proteins: training proteins the fold-0 adapter did NOT see (cross-fitting holdout), with one MF class each = the most
specific of the 15 most frequent mid-depth MF terms the protein carries. Representations:
  frozen    ESM2-35M layer-12 mean-pool (no adapter)
  adapted   LoRA (layers 9-12) layer-12 mean-pool h, fold-0 adapter
  latents   mean of the 16 GO-grounded resampler latents Z, fold-0 adapter

  python scripts/extras.py
"""
import numpy as np

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.train.common import fold_of, log, save_json, train_rows

TAG = "extras"


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def main():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = Bundle()
    rng = np.random.default_rng(0)
    tr = train_rows(b, 0, 0)
    hold = tr[fold_of(tr) != 0]
    Y = b.Y_t0
    mf = np.where(b.aspect == "F")[0]
    n_anc = np.asarray(b.vocab.n_anc.values)
    prev = np.asarray((Y[hold][:, mf] > 0).sum(0)).ravel()
    cand = mf[(n_anc[mf] >= 3) & (n_anc[mf] <= 5)]
    cand = cand[np.argsort(-prev[np.searchsorted(mf, cand)])][:15]
    Yc = Y[hold][:, cand].toarray() > 0
    has = Yc.any(1)
    spec = np.where(Yc, b.ia[cand][None, :], -1).argmax(1)
    pick = rng.choice(np.where(has)[0], min(6000, has.sum()), replace=False)
    rows, lab = hold[pick], spec[pick]
    names = [b.vocab.name.values[c] for c in cand]
    log(TAG, f"{len(rows):,} held-out training proteins, {len(cand)} MF classes: "
             f"{[(n[:30], int((lab == k).sum())) for k, n in enumerate(names)][:6]} ...")

    E = np.load(paths.CACHE / "frozen35m_mean.npy", mmap_mode="r")
    run = paths.RUNS / "full" / "enc_lora35m_f0"
    pos = {int(r): i for i, r in enumerate(np.load(run / "rows.npy"))}
    zi = np.array([pos[r] for r in rows])
    reps = {"frozen": np.asarray(E[rows], np.float32),
            "adapted": np.asarray(np.load(run / "h.npy", mmap_mode="r")[zi], np.float32),
            "latents": np.asarray(np.load(run / "Z.npy", mmap_mode="r")[zi], np.float32).mean(1)}
    res = {}
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for k, (nm, X) in enumerate(reps.items()):
        Xp = PCA(50, random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
        km = KMeans(len(cand), n_init=10, random_state=0).fit_predict(Xp)
        r = {"NMI": float(normalized_mutual_info_score(lab, km)), "ARI": float(adjusted_rand_score(lab, km)),
             "silhouette_cos": float(silhouette_score(Xp, lab, metric="cosine"))}
        res[nm] = r
        log(TAG, f"{nm:8s} k-means NMI {r['NMI']:.4f} ARI {r['ARI']:.4f}; MF-class silhouette (cosine) "
                 f"{r['silhouette_cos']:.4f}")
        U = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=0).fit_transform(Xp)
        ax[k].scatter(U[:, 0], U[:, 1], c=lab, cmap="tab20", s=2)
        ax[k].set_title(f"{nm}: NMI {r['NMI']:.3f} ARI {r['ARI']:.3f} sil {r['silhouette_cos']:.3f}")
        ax[k].set_xticks([]); ax[k].set_yticks([])
    out = paths.RUNS / "final"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "umap_before_after.png", dpi=110)
    res["CKA_frozen_vs_adapted"] = linear_cka(reps["frozen"], reps["adapted"])
    res["CKA_frozen_vs_latents"] = linear_cka(reps["frozen"], reps["latents"])
    res["classes"] = names
    log(TAG, f"linear CKA frozen vs adapted h {res['CKA_frozen_vs_adapted']:.4f}; frozen vs latents "
             f"{res['CKA_frozen_vs_latents']:.4f}")
    save_json(res, out / "extras.json")
    log(TAG, f"wrote {out / 'umap_before_after.png'} and extras.json")


if __name__ == "__main__":
    main()
