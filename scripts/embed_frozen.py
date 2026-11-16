"""Frozen ESM2 mean-pool embeddings of every bundle protein (float16), for the raw-embedding baselines.

  python scripts/embed_frozen.py --esm facebook/esm2_t12_35M_UR50D --out runs/cache/frozen35m_mean.npy
"""
import argparse
import time

import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.models.encoder import ESMEncoder, esm_tokenizer, length_batches, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--out", default=str(paths.CACHE / "frozen35m_mean.npy"))
    ap.add_argument("--max-tokens", type=int, default=120000)
    a = ap.parse_args()
    t0 = time.time()
    b = Bundle()
    seqs, lens = b.prot.seq.values, b.prot.length.values
    print(f"[embed] proteins {len(seqs):,}; encoder {a.esm}", flush=True)
    dev = torch.device("cuda")
    tok = esm_tokenizer(a.esm)
    enc = ESMEncoder(a.esm).to(dev).eval()
    print(f"[embed] hidden {enc.hidden}; params {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M (all frozen)")
    E = np.zeros((len(seqs), enc.hidden), np.float16)
    batches = length_batches(lens, a.max_tokens)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, bi in enumerate(batches):
            ids, am = tokenize(tok, [seqs[j] for j in bi], dev)
            H = enc(ids, am).last_hidden_state.float()
            m = enc.residue_mask(ids, am).float()
            E[bi] = ((H * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu().numpy().astype(np.float16)
            if i % 200 == 0:
                print(f"[embed] batch {i}/{len(batches)} size {len(bi)} {time.time() - t0:.0f}s", flush=True)
    np.save(a.out, E)
    print(f"[embed] wrote {E.shape} -> {a.out}; sample row 0 {E[0, :6]}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
