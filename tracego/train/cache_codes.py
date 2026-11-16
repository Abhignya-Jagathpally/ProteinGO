"""Residue codes for the late-interaction re-ranker: layer-L ESM2 states, PCA to d dims, L2-normalised, fp16.

The LoRA adapter is placed only on the layers above L, so these states are the same in every encoder variant
(frozen, LoRA fold 0, LoRA fold 1); they are computed once and shared.

  python -m tracego.train.cache_codes --config configs/full.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.models.encoder import ESMEncoder, esm_tokenizer, length_batches, tokenize
from tracego.train.common import load_config, log
from tracego.train.train_encoder import cache_rows

TAG = "codes"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    R, E = cfg["retrieval"], cfg["encoder"]
    out = paths.RUNS / cfg["name"] / f"codes_l{R['code_layer']}_d{R['code_dim']}.npz"
    dev = torch.device("cuda")
    b = Bundle()
    rows = cache_rows(b, cfg)
    t0 = time.time()
    tok = esm_tokenizer(E["esm"])
    enc = ESMEncoder(E["esm"]).to(dev).eval()
    seqs, lens = b.prot.seq.values[rows], b.prot.length.values[rows]
    log(TAG, f"{len(rows):,} proteins; layer {R['code_layer']} -> PCA {R['code_dim']}")
    pca = None
    parts = [None] * len(rows)
    for bi in length_batches(lens, 100000):
        ids, am = tokenize(tok, [seqs[j] for j in bi], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = enc.esm(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[R["code_layer"]]
        hs = hs.float()
        m = enc.residue_mask(ids, am)
        if pca is None:
            X = hs[m][torch.randperm(int(m.sum()), device=dev)[:200000]]
            mu = X.mean(0)
            _, _, V = torch.pca_lowrank(X - mu, q=R["code_dim"], center=False)
            pca = (mu, V[:, :R["code_dim"]])
            log(TAG, f"PCA fit on {len(X):,} residues of the first (shortest-length) batch")
        z = torch.nn.functional.normalize((hs - pca[0]) @ pca[1], dim=-1).half()
        for k, j in enumerate(bi):
            parts[j] = z[k][m[k]].cpu().numpy()
    off = np.zeros(len(rows) + 1, np.int64)
    off[1:] = np.cumsum([len(p) for p in parts])
    X = np.concatenate(parts)
    np.savez(out, X=X, off=off, rows=rows, mu=pca[0].cpu().numpy(), V=pca[1].cpu().numpy())
    log(TAG, f"codes {X.shape} {X.dtype} ({X.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s -> {out}; "
             f"sample protein {b.ids[rows[0]]} first residue {X[0, :4]}")


if __name__ == "__main__":
    main()
