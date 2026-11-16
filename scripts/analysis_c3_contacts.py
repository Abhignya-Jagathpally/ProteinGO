"""Phase C3: ESM2-35M contact confidence for TV queries vs per-protein F and homolog count.

  CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c3_contacts.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tracego.analysis.common import disk_free_gb, evidence, log, out_dir, per_protein_f, tv_identity, tv_predictions
from tracego.analysis.contacts import contact_confidence, plot_contact_map
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator, bin_of
from tracego.models.encoder import esm_tokenizer, length_batches, tokenize
from tracego.train.common import save_json

TAG = "c3"
MAX_L = 1022


def main():
    od = out_dir("c3")
    log(TAG, f"disk free {disk_free_gb():.1f} GB")
    b = Bundle()
    tv = b.rows("tv")
    ids = b.ids[tv]
    seqs, lens = b.prot.seq.values, b.prot.length.values
    ev = Evaluator("tv")
    P, pid = tv_predictions("vote")
    pos = {str(x): i for i, x in enumerate(pid)}
    P = P[np.array([pos[str(x)] for x in ids])]
    cur = ev.curves(P, ids, onto=True)
    ppf = per_protein_f(ev, cur, "MF").set_index("id").f
    # also mean over aspects
    from tracego.analysis.taxonomy import per_protein_table
    ppFm = per_protein_table(ev, cur, per_protein_f)
    ident = tv_identity(ids)
    z0 = evidence(0, "ca")
    # homolog counts among top-50
    S = z0["tv_S"]
    n_h03 = (S >= 0.3).sum(1)
    n_h05 = (S >= 0.5).sum(1)

    dev = torch.device("cuda")
    from transformers import EsmForMaskedLM
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D", attn_implementation="eager").to(dev).eval()
    tok = esm_tokenizer("facebook/esm2_t12_35M_UR50D")
    log(TAG, f"ESM2-35M contact head ready; scoring {len(tv):,} TV proteins (cap L={MAX_L})")

    rows = []
    # store a few maps for plotting
    want_plots = []
    # prefer proteins from full/trace if present
    try:
        tr = json.load(open(__import__("tracego.paths", fromlist=["p"]).paths.RUNS / "full" / "trace" / "trace.json"))
        want_plots = [e["protein"] for e in tr][:4]
    except Exception:
        want_plots = []
    sites_path = out_dir("c1") / "tv_sites.json"
    sites = json.load(open(sites_path)) if sites_path.exists() else {}
    if len(want_plots) < 5:
        for pid_ in ids:
            if str(pid_) in sites and str(pid_) not in want_plots:
                want_plots.append(str(pid_))
            if len(want_plots) >= 5:
                break
    maps = {}

    with torch.no_grad():
        for bi in length_batches(np.minimum(lens[tv], MAX_L), 24000, 8, False):
            br = tv[bi]
            batch_seqs = [seqs[j][:MAX_L] for j in br]
            ids_t, am = tokenize(tok, batch_seqs, dev)
            contacts = model.predict_contacts(ids_t, am).float().cpu().numpy()
            for j, r in enumerate(br):
                L = len(batch_seqs[j])
                C = contacts[j][:L, :L]
                conf12 = contact_confidence(C, 12)
                conf6 = contact_confidence(C, 6)
                pid_ = str(b.ids[r])
                rows.append({
                    "id": pid_, "L": L, "ident": float(ident[bi[j]] if False else ident[list(tv).index(r)]),
                    "n_h03": int(n_h03[list(tv).index(r)]), "n_h05": int(n_h05[list(tv).index(r)]),
                    "topL_mean_12": conf12["topL_mean"], "topL_mass_12": conf12["topL_mass"],
                    "topL_mean_6": conf6["topL_mean"], "topL_mass_6": conf6["topL_mass"],
                    "ppF_MF": float(ppf.get(pid_, np.nan)), "ppF_mean": float(ppFm.get(pid_, np.nan)),
                })
                if pid_ in want_plots:
                    maps[pid_] = (batch_seqs[j], C)
            if len(rows) % 500 < len(bi):
                log(TAG, f"contacts {len(rows):,}/{len(tv):,}")

    df = pd.DataFrame(rows)
    # fix ident alignment properly
    id2i = {str(x): i for i, x in enumerate(ids)}
    df["ident"] = [float(ident[id2i[i]]) for i in df.id]
    df["n_h03"] = [int(n_h03[id2i[i]]) for i in df.id]
    df["n_h05"] = [int(n_h05[id2i[i]]) for i in df.id]
    df["err"] = 1.0 - df["ppF_mean"]
    df.to_csv(od / "contact_confidence.tsv", sep="\t", index=False)

    from scipy.stats import spearmanr
    summary = {}
    for col in ("topL_mean_12", "topL_mass_12", "n_h03", "n_h05", "ident"):
        rho, p = spearmanr(df[col], df["err"], nan_policy="omit")
        summary[f"spearman_{col}_vs_err"] = {"rho": float(rho), "p": float(p)}
        log(TAG, f"Spearman {col} vs err: rho={rho:.3f} p={p:.2e}")
    # confidence vs homolog count
    rho, p = spearmanr(df["topL_mean_12"], df["n_h03"], nan_policy="omit")
    summary["spearman_conf_vs_nh03"] = {"rho": float(rho), "p": float(p)}
    log(TAG, f"Spearman conf vs n_h03: rho={rho:.3f} p={p:.2e}")
    # partial: residualise conf and err on identity ranks
    def rank_resid(y, x):
        from scipy.stats import rankdata
        yr, xr = rankdata(y), rankdata(x)
        # simple linear residual
        x0 = xr - xr.mean()
        b_ = np.dot(x0, yr - yr.mean()) / (np.dot(x0, x0) + 1e-9)
        return (yr - yr.mean()) - b_ * x0
    r_conf = rank_resid(df["topL_mean_12"].values, df["ident"].values)
    r_err = rank_resid(df["err"].values, df["ident"].values)
    rho, p = spearmanr(r_conf, r_err)
    summary["partial_conf_err_given_ident"] = {"rho": float(rho), "p": float(p)}
    log(TAG, f"partial Spearman conf vs err | ident: rho={rho:.3f} p={p:.2e}")

    # bins by homolog count
    df["hbin"] = pd.cut(df["n_h03"], bins=[-0.1, 0, 2, 10, 50], labels=["0", "1-2", "3-10", "11-50"])
    hb = df.groupby("hbin", observed=False).agg(n=("id", "size"), conf=("topL_mean_12", "mean"),
                                                 err=("err", "mean")).reset_index()
    hb.to_csv(od / "conf_by_homolog_bin.tsv", sep="\t", index=False)
    log(TAG, f"conf by homolog bin:\n{hb.to_string(index=False)}")

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(df["topL_mean_12"], df["err"], s=3, alpha=0.3)
    ax[0].set_xlabel("top-L mean contact P (|i-j|>=12)")
    ax[0].set_ylabel("1 - per-protein F")
    ax[0].set_title("Contact conf vs GO error")
    df.boxplot(column="topL_mean_12", by="hbin", ax=ax[1])
    ax[1].set_title("Contact conf vs #homologs (S>=0.3)")
    ax[1].set_xlabel("# training homologs in top-50")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(od / "conf_vs_err_homologs.png", dpi=120)

    for pid_, (seq, C) in maps.items():
        site = None
        if pid_ in sites:
            from tracego.analysis.uniprot_sites import site_sets
            site = site_sets(sites[pid_], len(seq)).get("functional")
        plot_contact_map(seq, C, od / f"contacts_{pid_}.png", sites=site, title=f"{pid_} ESM2-35M contacts")
        log(TAG, f"wrote contacts_{pid_}.png")

    save_json(summary, od / "summary.json")
    log(TAG, f"done; disk free {disk_free_gb():.1f} GB")


if __name__ == "__main__":
    main()
