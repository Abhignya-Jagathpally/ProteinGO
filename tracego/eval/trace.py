"""Reverse trace from emitted GO terms back to residues, neighbours and training sequences.

For each protein and each GO token the decoder emits:
  residues    decoder cross-attention on the 16 soft-prompt positions (mean over heads and layers), composed with the
              resampler's latent-to-residue attention (last block, mean over heads): attention rollout to residues;
  evidence    cross-attention mass on each retrieved neighbour's evidence span (which homologue carried the term);
  sequences   GO -> sequence retrieval: training proteins whose latents score highest for the term under the
              adapter's term head (max over latents of the cosine to the term embedding), with the residue window
              the best latent attends to.
Also an ESM2 contact map for the protein (contact head of the same pretrained encoder).

  python -m tracego.eval.trace --config configs/full.yaml --tag main --proteins P69905,Q9Y6K9 --out runs/full/trace
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from tracego import paths
from tracego.eval.predict import setup
from tracego.models.adapter import AdaptedEncoder
from tracego.models.encoder import esm_tokenizer, tokenize
from tracego.train.common import load_config, log

TAG = "trace"


def load_adapter(cfg, fold, b, dev):
    E = cfg["encoder"]
    st = torch.load(paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{fold}" / "model.pt", map_location=dev,
                    weights_only=False)
    m = AdaptedEncoder(E["esm"], len(b.terms), lora_r=E["lora_r"], n_latents=E["n_latents"],
                       depth=E.get("depth", 2), lora_from_layer=E.get("lora_from_layer", 0)).to(dev)
    missing = m.load_state_dict(st["state"], strict=False)
    assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
    return m.eval()


@torch.no_grad()
def go_to_sequences(adapter, Zcache, rows, term_idx, topn=5):
    """Training proteins whose cached latents best express the term (adapter term head)."""
    e = torch.nn.functional.normalize(adapter.term_emb[term_idx], dim=-1)
    best, arg = [], []
    for s in range(0, len(rows), 8192):
        Z = torch.as_tensor(np.asarray(Zcache[s:s + 8192]), dtype=torch.float32, device=e.device)
        q = torch.nn.functional.normalize(adapter.cls_proj(Z), dim=-1)
        sc = (q @ e).amax(1)
        best.append(sc.cpu().numpy())
        arg.append((q @ e).argmax(1).cpu().numpy())
    sc, am = np.concatenate(best), np.concatenate(arg)
    o = np.argsort(-sc)[:topn]
    return [(int(rows[i]), float(sc[i]), int(am[i])) for i in o]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--bench", default="tv")
    ap.add_argument("--proteins", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dev = torch.device("cuda")
    S_ = setup(cfg, a.tag, dev, ["tv"] + ([a.bench] if a.bench != "tv" else []))
    b, model, sets, coll, Hs, spaces = (S_[k] for k in ("b", "model", "sets", "coll", "Hs", "spaces"))
    # fused attention kernels return no weights, and switching after loading does not take effect: rebuild the T5
    # with eager attention and copy the trained weights in
    from transformers import T5ForConditionalGeneration
    t5e = T5ForConditionalGeneration.from_pretrained(cfg["decoder"]["t5"], attn_implementation="eager")
    t5e.resize_token_embeddings(len(model.tok), mean_resizing=False)
    t5e.load_state_dict(model.t5.state_dict())
    model.t5 = t5e.to(dev).eval()
    out_dir = paths.TRACEGO / a.out if not a.out.startswith("/") else paths.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = 0
    iset = sets[(a.bench, f)]
    want = a.proteins.split(",") if a.proteins else list(b.ids[iset.q_rows[:a.n]])
    pos = {p: i for i, p in enumerate(b.ids[iset.q_rows])}
    adapter = load_adapter(cfg, f, b, dev)
    tok = esm_tokenizer(cfg["encoder"]["esm"])
    from transformers import EsmForMaskedLM
    # contact head of the pretrained encoder (needs attention weights, hence eager)
    esm_c = EsmForMaskedLM.from_pretrained(cfg["encoder"]["esm"], attn_implementation="eager").to(dev).eval()
    Zc = spaces[f].Z
    tr_rows = spaces[f].rows
    is_train = np.isin(tr_rows, b.rows("train"))
    Ztr, rtr = Zc[np.where(is_train)[0]], tr_rows[is_train]
    report = []
    for pid in want:
        i = pos[pid]
        row = int(iset.q_rows[i])
        seq = b.prot.seq.values[row][:1022]
        ids, am = tokenize(tok, [seq], dev)
        enc = adapter(ids, am, need_weights=True)
        lat2res = enc["att"][-1][0][:, 1:len(seq) + 1].float()  # [M, L] (drop <cls>)
        contacts = esm_c.predict_contacts(ids, am)[0].float().cpu().numpy()
        entry = {"protein": pid, "length": len(seq), "organism": str(b.prot.organism.values[row]), "terms": []}
        for ns in range(3):
            bt = coll[f]([(iset, i)], ns, None, None, dev)
            from tracego.train.train_decoder import vote_matrix
            vt = vote_matrix(spaces[f], a.bench)[[i]]
            bt["vote"] = torch.as_tensor(vt[:, model.ns_cols[ns].numpy()].toarray(), device=dev)
            src = bt["src"][0].tolist()
            nb_pos = [j for j, t in enumerate(src) if t == model.nb_id]
            M = spaces[f].Z.shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc_h, enc_m = model.encode(bt["Z"], bt["src"], bt["src_mask"])
            # greedy constrained decode, keeping cross-attention of every step
            from transformers.cache_utils import DynamicCache, EncoderDecoderCache
            past = EncoderDecoderCache(DynamicCache(), DynamicCache())
            V = len(model.ns_cols[ns])
            Sm = torch.zeros(1, V, dtype=torch.bool, device=dev)
            cnt = torch.zeros(1, V, device=dev)
            cur = torch.zeros(1, 1, dtype=torch.long, device=dev)
            cols = model.ns_cols[ns].to(dev)
            for t in range(48):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model.decode(cur, enc_h, enc_m, past=past, output_attentions=True)
                past = o.past_key_values
                h = o.last_hidden_state[:, -1].float()
                mu = torch.sigmoid(model.member_logits(h, ns, bt["vote"]).float())
                lg, eos = model.next_logits(h, ns, bt["vote"])
                lg = lg.float().masked_fill(~Hs[ns].frontier(Sm, cnt), float("-inf"))
                full = torch.cat([lg, eos.float()[:, None]], -1)
                ch = int(full.argmax(-1))
                if ch == V:
                    break
                xa = torch.stack([x[0, :, -1].float() for x in o.cross_attentions]).mean((0, 1))  # [src]
                soft = xa[:M]
                res = (soft[:, None] * lat2res).sum(0)
                res = res / res.sum().clamp(min=1e-9)
                text_att = xa[M:]
                spans = []
                for k, p0 in enumerate(nb_pos):
                    p1 = nb_pos[k + 1] if k + 1 < len(nb_pos) else len(src)
                    spans.append(float(text_att[p0:p1].sum()))
                top_res = torch.topk(res, min(5, len(seq))).indices.sort().values.tolist()
                term = str(b.terms[int(cols[ch])])
                entry["terms"].append({
                    "ns": "FPC"[ns], "step": t, "term": term, "name": str(b.vocab.name.values[int(cols[ch])]),
                    "membership": float(mu[0, ch]), "soft_prompt_mass": float(soft.sum()),
                    "top_residues": [(r + 1, seq[r]) for r in top_res],
                    "neighbour_attention": [round(x, 4) for x in spans],
                    "neighbours": [str(b.ids[r]) for r in spaces[f].ev[f"{a.bench}_I"][i, :len(nb_pos)]],
                    "residue_profile": [round(float(x), 5) for x in res.cpu().numpy()]})
                Sm[0, ch] = True
                cnt[0] += Hs[ns].desc[ch].float()
                cur = (model.go0 + cols[ch]).view(1, 1)
        # GO -> sequence retrieval for the three most specific emitted terms
        deep = sorted(entry["terms"], key=lambda r: -b.ia[np.where(b.terms == r["term"])[0][0]])[:3]
        entry["go_to_sequence"] = {}
        for r in deep:
            ti = int(np.where(b.terms == r["term"])[0][0])
            hits = go_to_sequences(adapter, Ztr, rtr, ti)
            entry["go_to_sequence"][r["term"]] = [
                {"protein": str(b.ids[rr]), "score": round(s, 4), "latent": li,
                 "has_term_T0": bool(b.Y_t0[rr, ti] > 0)} for rr, s, li in hits]
        if contacts is not None:
            np.save(out_dir / f"{pid}_contacts.npy", contacts)
        np.save(out_dir / f"{pid}_lat2res.npy", lat2res.cpu().numpy())
        report.append(entry)
        log(TAG, f"{pid} ({entry['organism'][:30]}, L={len(seq)}): {len(entry['terms'])} emitted terms; first "
                 f"specific: {[(r['term'], r['name'][:30], r['top_residues'][:3]) for r in deep[:2]]}")
    json.dump(report, open(out_dir / "trace.json", "w"), indent=1)
    log(TAG, f"wrote {out_dir / 'trace.json'}")


if __name__ == "__main__":
    main()
