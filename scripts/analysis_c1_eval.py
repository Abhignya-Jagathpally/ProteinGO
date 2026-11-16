"""Phase C1: score residue traces of competitive / control / collapsed heads against UniProt sites.

For every TV protein with matching UniProt active/binding sites and a mapped MF term, build the term's residue
profile (latent→residue attention weighted by term-head cosine) and report top-k precision/recall vs sites, vs a
random baseline and an amino-acid prior. Also re-report TV term-head Fmax from the saved checkpoints.

  CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c1_eval.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from tracego.analysis.common import disk_free_gb, load_fold_encoder, log, out_dir
from tracego.analysis.slot_resampler import CompetitiveResampler, latent_corr
from tracego.analysis.trace_eval import (aa_prior, build_aa_prior, residue_profile, summarise, term_for_sites,
                                         topk_hits)
from tracego.analysis.uniprot_sites import site_sets
from tracego.data.bundle import Bundle
from tracego.data.godefs import build as build_defs, latent_init
from tracego.models.encoder import esm_tokenizer, length_batches, tokenize
from tracego.models.resampler import Resampler
from tracego.train.common import save_json, topk_csr

TAG = "c1eval"


class TraceHead(torch.nn.Module):
    def __init__(self, resampler, term_init, n_terms, d=512):
        super().__init__()
        self.resampler = resampler
        self.cls_proj = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d))
        te = torch.as_tensor(term_init, dtype=torch.float32)
        te = te / te.norm(dim=1, keepdim=True)
        self.term_emb = torch.nn.Parameter(te.clone())
        self.term_bias = torch.nn.Parameter(torch.full((n_terms,), -4.0))
        self.scale = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, H, mask, need_weights=False):
        return self.resampler(H, mask, need_weights)

    def term_logits(self, Z):
        import torch.nn.functional as F
        q = F.normalize(self.cls_proj(Z), dim=-1)
        e = F.normalize(self.term_emb, dim=-1)
        return self.scale * (q @ e.T).amax(1) + self.term_bias


def make_head(kind, d_in, n_terms, defs, lat, ckpt, dev):
    if kind == "competitive":
        rs = CompetitiveResampler(d_in, 512, 16, 2, init=lat)
    else:
        rs = Resampler(d_in, 512, 16, 2, init=lat)
    h = TraceHead(rs, defs, n_terms).to(dev)
    h.load_state_dict(ckpt["state"])
    return h.eval()


@torch.no_grad()
def collect_profiles(name, head, enc, tok, b, want_ids, term_idx_by_id, dev):
    """For each protein id in want_ids: compute att [M,L], Z, and profiles for requested term indices."""
    seqs = dict(zip(b.ids, b.prot.seq.values))
    row_of = {str(i): int(r) for r, i in enumerate(b.ids)}
    out = {}
    ids = [i for i in want_ids if i in row_of]
    rows = np.array([row_of[i] for i in ids])
    lens = b.prot.length.values
    for bi in length_batches(lens[rows], 48000, 32, False):
        br = rows[bi]
        batch_ids = [str(b.ids[r]) for r in br]
        ids_t, am = tokenize(tok, [seqs[i] for i in batch_ids], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = enc(ids_t, am)
            Hs = o.last_hidden_state
            m = enc.residue_mask(ids_t, am)
            Z, ws = head(Hs, m, need_weights=True)
        att = ws[-1]
        for j, pid in enumerate(batch_ids):
            L = int(m[j].sum())
            zj, aj = Z[j], att[j][:, :L]
            profs = {}
            for t in term_idx_by_id.get(pid, []):
                profs[int(t)] = residue_profile(zj, aj, head.term_emb, head.cls_proj, int(t))
            out[pid] = {"L": L, "att_corr": latent_corr(att[j:j + 1], m[j:j + 1]),
                        "profiles": profs, "att_mean": aj.mean(0).float().cpu().numpy()}
        if len(out) % 200 < len(bi):
            log(TAG, f"{name}: profiles {len(out):,}/{len(ids):,}")
    return out


@torch.no_grad()
def collapsed_profiles(adapter, tok, b, want_ids, term_idx_by_id, dev):
    seqs = dict(zip(b.ids, b.prot.seq.values))
    row_of = {str(i): int(r) for r, i in enumerate(b.ids)}
    out = {}
    ids = [i for i in want_ids if i in row_of]
    rows = np.array([row_of[i] for i in ids])
    lens = b.prot.length.values
    for bi in length_batches(lens[rows], 48000, 32, False):
        br = rows[bi]
        batch_ids = [str(b.ids[r]) for r in br]
        ids_t, am = tokenize(tok, [seqs[i] for i in batch_ids], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = adapter(ids_t, am, need_weights=True)
        Z, att, m = o["Z"], o["att"][-1], o["mask"]
        for j, pid in enumerate(batch_ids):
            L = int(m[j].sum())
            zj, aj = Z[j], att[j][:, :L]
            profs = {}
            for t in term_idx_by_id.get(pid, []):
                profs[int(t)] = residue_profile(zj, aj, adapter.term_emb, adapter.cls_proj, int(t))
            out[pid] = {"L": L, "att_corr": latent_corr(att[j:j + 1], m[j:j + 1]),
                        "profiles": profs, "att_mean": aj.mean(0).float().cpu().numpy()}
    return out


def main():
    od = out_dir("c1")
    log(TAG, f"disk free {disk_free_gb():.1f} GB")
    dev = torch.device("cuda")
    b = Bundle()
    sites = json.load(open(od / "tv_sites.json"))
    terms_set = set(b.terms.tolist())
    term_pos = {t: i for i, t in enumerate(b.terms)}
    prior = build_aa_prior(sites)
    log(TAG, f"sites proteins {len(sites):,}; AA prior top {sorted(prior, key=prior.get, reverse=True)[:8]}")

    # build evaluation items: (pid, label, sites, term)
    items = []
    term_idx_by_id = defaultdict(set)
    for pid, e in sites.items():
        ss = site_sets(e, len(e["seq"]))
        for label, pos in ss.items():
            if label not in ("act", "functional") and not label.startswith("lig:"):
                continue
            term = term_for_sites(e, label, terms_set)
            if term is None or term not in term_pos:
                continue
            items.append({"id": pid, "label": label, "sites": pos, "term": term, "tidx": term_pos[term],
                          "seq": e["seq"]})
            term_idx_by_id[pid].add(term_pos[term])
    log(TAG, f"eval items {len(items):,} across {len(term_idx_by_id):,} proteins")

    adapter = load_fold_encoder(0, b, dev)
    tok = esm_tokenizer("facebook/esm2_t12_35M_UR50D")
    defs = build_defs(b.vocab)
    lat = latent_init(defs, b.aspect, 16)
    d_in = adapter.enc.hidden

    models = {"collapsed": None}
    for kind in ("control", "competitive"):
        ck = od / f"{kind}_best.pt"
        if not ck.exists():
            log(TAG, f"missing {ck}; skip {kind}")
            continue
        models[kind] = make_head(kind, d_in, len(b.terms), defs, lat, torch.load(ck, map_location=dev,
                                                                                 weights_only=False), dev)
        log(TAG, f"loaded {ck} epoch {torch.load(ck, map_location='cpu', weights_only=False).get('epoch')}")

    want = list(term_idx_by_id.keys())
    profiles = {}
    profiles["collapsed"] = collapsed_profiles(adapter, tok, b, want, term_idx_by_id, dev)
    log(TAG, f"collapsed mean att_corr "
             f"{np.mean([v['att_corr'] for v in profiles['collapsed'].values()]):.3f}")
    for kind, head in models.items():
        if head is None or kind == "collapsed":
            continue
        profiles[kind] = collect_profiles(kind, head, adapter.enc, tok, b, want, term_idx_by_id, dev)
        log(TAG, f"{kind} mean att_corr {np.mean([v['att_corr'] for v in profiles[kind].values()]):.3f}")

    # score
    rng = np.random.default_rng(0)
    rows_by = {k: [] for k in profiles}
    for it in items:
        pid, sites_pos, tidx, seq = it["id"], it["sites"], it["tidx"], it["seq"]
        L = len(seq)
        n_s = len(sites_pos)
        rand = {"prec": n_s / L, "rec": min(1.0, 10 / L) * (n_s / L) * L / max(n_s, 1)}  # E[prec]=n/L for any k
        # proper random: expected top-k prec = n_s/L, expected recall for k = min(k,L)* (n_s/L) / n_s * something
        # use Monte Carlo once per length bucket would be overkill; analytical E[prec]=n_s/L, E[rec]=k/L for k<=L
        aa = aa_prior(seq, prior)
        for kind, profs in profiles.items():
            if pid not in profs or tidx not in profs[pid]["profiles"]:
                continue
            p = profs[pid]["profiles"][tidx]
            # also uniform-over-latents (= mean attention) as a non-term-specific baseline
            uni = profs[pid]["att_mean"]
            uni = uni / uni.sum()
            r = {"id": pid, "label": it["label"], "term": it["term"], "n_sites": n_s, "L": L,
                 "att_corr": profs[pid]["att_corr"]}
            for tag, prof in (("term", p), ("uniform", uni), ("aa", aa)):
                for kname, k in (("k_sites", n_s), ("k10", 10)):
                    h = topk_hits(prof, sites_pos, k)
                    r[f"{tag}_{kname}_prec"] = h["prec"]
                    r[f"{tag}_{kname}_rec"] = h["rec"]
            r["rand_k_sites_prec"] = n_s / L
            r["rand_k_sites_rec"] = min(n_s, L) / L   # expected recall when k=#sites under uniform
            r["rand_k10_prec"] = n_s / L
            r["rand_k10_rec"] = min(10, L) / L
            rows_by[kind].append(r)

    summary = {}
    for kind, rows in rows_by.items():
        summary[kind] = summarise(rows)
        # also by label family
        for lab in ("act", "functional", "lig"):
            sub = [r for r in rows if r["label"] == lab or (lab == "lig" and r["label"].startswith("lig:"))]
            if sub:
                summary[f"{kind}/{lab}"] = summarise(sub)
        log(TAG, f"{kind}: n={summary[kind].get('n', 0)} term@k_sites prec "
                 f"{summary[kind].get('term_k_sites_prec', float('nan')):.4f} rec "
                 f"{summary[kind].get('term_k_sites_rec', float('nan')):.4f}; "
                 f"rand prec {summary[kind].get('rand_k_sites_prec', float('nan')):.4f}; "
                 f"aa prec {summary[kind].get('aa_k_sites_prec', float('nan')):.4f}; "
                 f"uniform prec {summary[kind].get('uniform_k_sites_prec', float('nan')):.4f}")

    save_json(summary, od / "trace_metrics.json")
    # keep a compact TSV of per-item competitive vs collapsed
    if "competitive" in rows_by and "collapsed" in rows_by:
        import pandas as pd
        A = { (r["id"], r["label"], r["term"]): r for r in rows_by["competitive"] }
        B = { (r["id"], r["label"], r["term"]): r for r in rows_by["collapsed"] }
        keys = sorted(set(A) & set(B))
        tab = []
        for k in keys:
            a, c = A[k], B[k]
            tab.append({"id": k[0], "label": k[1], "term": k[2], "n_sites": a["n_sites"], "L": a["L"],
                        "comp_prec": a["term_k_sites_prec"], "coll_prec": c["term_k_sites_prec"],
                        "comp_rec": a["term_k_sites_rec"], "coll_rec": c["term_k_sites_rec"],
                        "aa_prec": a["aa_k_sites_prec"], "rand_prec": a["rand_k_sites_prec"],
                        "comp_corr": a["att_corr"], "coll_corr": c["att_corr"]})
        pd.DataFrame(tab).to_csv(od / "trace_compare.tsv", sep="\t", index=False)
        log(TAG, f"wrote {od / 'trace_compare.tsv'} ({len(tab)} rows)")

    # figures: precision bars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kinds = [k for k in ("collapsed", "control", "competitive") if k in summary and summary[k]]
        metrics = ["term_k_sites_prec", "term_k_sites_rec", "aa_k_sites_prec", "rand_k_sites_prec"]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(kinds))
        w = 0.18
        for i, m in enumerate(metrics):
            ax.bar(x + (i - 1.5) * w, [summary[k].get(m, 0) for k in kinds], w, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(kinds)
        ax.set_ylabel("score")
        ax.set_title("C1 residue-trace top-k vs UniProt sites")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(od / "trace_metrics.png", dpi=120)
        log(TAG, f"wrote {od / 'trace_metrics.png'}")
    except Exception as e:
        log(TAG, f"figure skipped: {type(e).__name__}: {e}")

    # touch DONE marker for downstream C3/D1 overlap
    (od / "DONE").write_text("c1 eval complete\n")
    log(TAG, f"done; disk free {disk_free_gb():.1f} GB")


if __name__ == "__main__":
    main()
