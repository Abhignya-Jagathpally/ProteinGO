"""Phase D1: ATLAS RMSF summaries → small ESM2 RMSF head → site/flexibility case studies.

Downloads only per-chain analysis zips (RMSF tables), deletes archives after extract. Cap ~1.5 GB.

  CUDA_VISIBLE_DEVICES=0 python scripts/analysis_d1_md.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F

from tracego.analysis.common import disk_free_gb, disk_guard, log, out_dir
from tracego.analysis.flexibility import RmsfHead
from tracego.models.encoder import esm_tokenizer, length_batches, tokenize
from tracego.train.common import n_params, save_json

TAG = "d1"
ATLAS_API = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{pdb_chain}"
MAX_CHAINS = 80  # was 250; capped for disk/time
MAX_DOWNLOAD_GB = 1.2


def fetch_rmsf(pdb_chain: str, raw: Path) -> pd.DataFrame | None:
    """Download one ATLAS analysis zip, extract RMSF table, delete zip. Returns DataFrame or None."""
    url = ATLAS_API.format(pdb_chain=pdb_chain)
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200 or len(r.content) < 100:
            return None
        zpath = raw / f"{pdb_chain}.zip"
        zpath.write_bytes(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            # prefer files with RMSF in the name
            cand = [n for n in names if "rmsf" in n.lower() or n.lower().endswith(".tsv") or n.lower().endswith(".csv")]
            if not cand:
                cand = names
            df = None
            for n in cand:
                try:
                    data = zf.read(n)
                    for sep in ("\t", ",", r"\s+"):
                        try:
                            d = pd.read_csv(io.BytesIO(data), sep=sep, engine="python")
                            cols = {c.lower(): c for c in d.columns}
                            if any("rmsf" in c for c in cols):
                                df = d
                                break
                        except Exception:
                            continue
                    if df is not None:
                        break
                except Exception:
                    continue
        zpath.unlink(missing_ok=True)
        return df
    except Exception as e:
        log(TAG, f"{pdb_chain}: {type(e).__name__}: {e}")
        return None


def normalize_rmsf_table(df: pd.DataFrame, seq: str) -> np.ndarray | None:
    cols = {c.lower(): c for c in df.columns}
    rcol = next((cols[c] for c in cols if "rmsf" in c), None)
    if rcol is None:
        return None
    y = df[rcol].to_numpy(dtype=np.float64)
    if len(y) == len(seq):
        return y
    # try residue index column
    icol = next((cols[c] for c in cols if c in ("resid", "res", "residue", "resi", "pdb_resid", "unp_num")), None)
    if icol is None:
        return None
    out = np.full(len(seq), np.nan)
    idx = df[icol].to_numpy()
    for i, v in zip(idx, y):
        try:
            i = int(i) - 1
            if 0 <= i < len(seq):
                out[i] = v
        except Exception:
            continue
    if np.isnan(out).mean() > 0.3:
        return None
    # interpolate nan lightly
    nans = np.isnan(out)
    if nans.any():
        xp = np.where(~nans)[0]
        out[nans] = np.interp(np.where(nans)[0], xp, out[~nans])
    return out


def main():
    od = out_dir("d1")
    raw = od / "raw"
    raw.mkdir(exist_ok=True)
    info = pd.read_csv(raw / "ATLAS_info.tsv", sep="\t")
    log(TAG, f"ATLAS_info {len(info):,} chains; columns {list(info.columns)[:8]}...")
    disk_guard(TAG, 0.5, floor_gb=5.0)

    # prefer chains with UniProt ids overlapping TV/train-ish; sample non-redundant
    info = info.dropna(subset=["UniProt", "sequence"]).copy()
    info = info[info["sequence"].str.len().between(50, 800)]
    if "non_redundant_protein" in info.columns:
        info = info[info["non_redundant_protein"].astype(str).str.lower().isin(["true", "1", "yes"])]
    info = info.drop_duplicates("UniProt")
    rng = np.random.default_rng(0)
    pick = info.sample(n=min(MAX_CHAINS, len(info)), random_state=0)
    log(TAG, f"fetching RMSF for {len(pick):,} chains (cap download {MAX_DOWNLOAD_GB} GB)")

    records = []
    downloaded = 0.0
    for i, row in enumerate(pick.itertuples()):
        pdb_chain = str(row.PDB)
        df = fetch_rmsf(pdb_chain, raw)
        if df is None:
            continue
        # rough size accounting: response already deleted; count ~0.5 MB avg
        downloaded += 0.0005
        y = normalize_rmsf_table(df, str(row.sequence))
        if y is None:
            continue
        records.append({"pdb": pdb_chain, "uniprot": str(row.UniProt), "seq": str(row.sequence), "rmsf": y})
        if (i + 1) % 25 == 0:
            log(TAG, f"fetched usable {len(records):,}/{i+1}; ~{downloaded:.2f} GB transferred")
        if downloaded > MAX_DOWNLOAD_GB:
            log(TAG, "download cap reached; stopping fetch")
            break

    log(TAG, f"usable RMSF chains {len(records):,}")
    if len(records) < 20:
        log(TAG, "too few chains; aborting head train")
        save_json({"n": len(records)}, od / "results.json")
        return

    # split by PDB id prefix
    pdb_ids = np.array([r["pdb"].split("_")[0] for r in records])
    uniq = np.unique(pdb_ids)
    rng.shuffle(uniq)
    n_te = max(5, int(0.2 * len(uniq)))
    te_set = set(uniq[:n_te])
    tr_idx = [i for i, p in enumerate(pdb_ids) if p not in te_set]
    te_idx = [i for i, p in enumerate(pdb_ids) if p in te_set]
    log(TAG, f"split train chains {len(tr_idx)} test {len(te_idx)} (by PDB id)")

    # baselines: residue type + relative position
    aa = "ACDEFGHIKLMNPQRSTVWY"
    aa_mean = {a: [] for a in aa}
    for i in tr_idx:
        for a, v in zip(records[i]["seq"], records[i]["rmsf"]):
            if a in aa_mean:
                aa_mean[a].append(v)
    aa_mu = {a: float(np.mean(v)) if v else 1.0 for a, v in aa_mean.items()}

    def baseline(seq):
        L = len(seq)
        pos = np.minimum(np.arange(L), np.arange(L)[::-1]) / max(L, 1)
        return np.array([aa_mu.get(a, 1.0) for a in seq]) * (0.7 + 0.6 * (1 - pos))

    # train head
    from transformers import AutoModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D", add_pooling_layer=False).to(dev).eval()
    for p in esm.parameters():
        p.requires_grad_(False)
    tok = esm_tokenizer("facebook/esm2_t12_35M_UR50D")
    head = RmsfHead(480, 128).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    log(TAG, f"RMSF head params {n_params(head):,}; device {dev}")

    def batches(idxs, shuffle=False):
        order = np.array(idxs)
        if shuffle:
            rng.shuffle(order)
        # group by length
        lens = np.array([len(records[i]["seq"]) for i in order])
        for bi in length_batches(lens, 24000, 16, shuffle, rng if shuffle else None):
            yield order[bi]

    for ep in range(8):
        head.train()
        tot, n = 0.0, 0
        for br in batches(tr_idx, True):
            seqs = [records[i]["seq"] for i in br]
            ys = [records[i]["rmsf"] for i in br]
            ids, am = tokenize(tok, seqs, dev)
            with torch.no_grad():
                H = esm(input_ids=ids, attention_mask=am).last_hidden_state
            m = am.bool().clone(); m[:, 0] = False
            lens = am.sum(1)
            m[torch.arange(len(br), device=dev), lens - 1] = False
            # pad targets
            Y = torch.zeros(len(br), ids.shape[1], device=dev)
            for j, y in enumerate(ys):
                Y[j, 1:1 + len(y)] = torch.as_tensor(y, device=dev)
            pred = head(H, m)
            loss = F.mse_loss(pred[m], Y[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * int(m.sum())
            n += int(m.sum())
        # eval
        head.eval()
        def score(idxs):
            ps, ys, bs = [], [], []
            with torch.no_grad():
                for br in batches(idxs, False):
                    seqs = [records[i]["seq"] for i in br]
                    ids, am = tokenize(tok, seqs, dev)
                    H = esm(input_ids=ids, attention_mask=am).last_hidden_state
                    m = am.bool().clone(); m[:, 0] = False
                    lens = am.sum(1)
                    m[torch.arange(len(br), device=dev), lens - 1] = False
                    pred = head(H, m)
                    for j, i in enumerate(br):
                        L = len(records[i]["seq"])
                        p = pred[j, 1:1 + L].float().cpu().numpy()
                        y = records[i]["rmsf"]
                        b = baseline(records[i]["seq"])
                        ps.append(p); ys.append(y); bs.append(b)
            def corr(a, b):
                from scipy.stats import pearsonr, spearmanr
                aa, bb = np.concatenate(a), np.concatenate(b)
                return float(pearsonr(aa, bb)[0]), float(spearmanr(aa, bb).correlation)
            return {"pearson": corr(ps, ys)[0], "spearman": corr(ps, ys)[1],
                    "base_pearson": corr(bs, ys)[0], "base_spearman": corr(bs, ys)[1]}
        tr_s, te_s = score(tr_idx), score(te_idx)
        log(TAG, f"epoch {ep}: loss {tot/max(n,1):.4f}; train P/S {tr_s['pearson']:.3f}/{tr_s['spearman']:.3f}; "
                 f"test P/S {te_s['pearson']:.3f}/{te_s['spearman']:.3f} "
                 f"(baseline {te_s['base_pearson']:.3f}/{te_s['base_spearman']:.3f})")

    torch.save({"state": head.state_dict(), "d_in": 480, "d": 128, "test": te_s}, od / "rmsf_head.pt")

    # case studies: overlap with UniProt sites if available
    import json
    sites_path = out_dir("c1") / "tv_sites.json"
    sites = json.load(open(sites_path)) if sites_path.exists() else {}
    cases = []
    for r in records:
        if r["uniprot"] not in sites:
            continue
        from tracego.analysis.uniprot_sites import site_sets
        # align ATLAS seq to UniProt seq if needed
        useq = sites[r["uniprot"]]["seq"]
        if r["seq"] != useq:
            # simple: skip if lengths differ a lot
            if abs(len(r["seq"]) - len(useq)) > 5:
                continue
        ss = site_sets(sites[r["uniprot"]], len(r["seq"]))
        pos = ss.get("functional", [])
        if not pos:
            continue
        y = r["rmsf"]
        z = (y - y.mean()) / (y.std() + 1e-9)
        site_z = float(np.mean(z[pos]))
        other_z = float(np.mean(np.delete(z, pos))) if len(pos) < len(z) else 0.0
        cases.append({"pdb": r["pdb"], "uniprot": r["uniprot"], "n_sites": len(pos),
                      "site_rmsf_z": site_z, "other_rmsf_z": other_z, "delta": site_z - other_z})
        if len(cases) >= 12:
            break
    if cases:
        pd.DataFrame(cases).to_csv(od / "site_flexibility.tsv", sep="\t", index=False)
        log(TAG, f"site flexibility cases {len(cases)}; mean site-other Δz "
                 f"{np.mean([c['delta'] for c in cases]):.3f}")
        # one plot
        r0 = next(r for r in records if r["pdb"] == cases[0]["pdb"])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(r0["rmsf"], lw=1)
        ss = site_sets(sites[r0["uniprot"]], len(r0["seq"])).get("functional", [])
        for s in ss:
            ax.axvline(s, color="C1", alpha=0.5, lw=0.8)
        ax.set_title(f"{r0['pdb']} / {r0['uniprot']} RMSF (sites in orange)")
        ax.set_xlabel("residue"); ax.set_ylabel("RMSF")
        fig.tight_layout(); fig.savefig(od / "rmsf_case.png", dpi=120)

    save_json({"n_chains": len(records), "test": te_s, "n_cases": len(cases),
               "mean_site_delta_z": float(np.mean([c["delta"] for c in cases])) if cases else None},
              od / "results.json")
    log(TAG, f"done; disk free {disk_free_gb():.1f} GB")


if __name__ == "__main__":
    main()
