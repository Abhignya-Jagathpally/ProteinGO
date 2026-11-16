"""Frozen mean-pooled embeddings (fp16) of every protein the decoder needs, from a large protein language model.

Phase A2 compares the retrieval / soft-prompt encoder: ESM2-650M, ESM-C 600M and ProtT5-XL-UniRef50 (SPROF-GO's
backbone). Only the residue-averaged last layer is kept (sequences truncated to 1,022 residues, as for ESM2-35M).
Rows follow runs/full/enc_lora35m_f0/rows.npy (training proteins, TV queries, val, test, fresh24 FR24-P).

  python scripts/embed_pools.py --model esm2_650m     # also esmc_600m, prot_t5
"""
import argparse
import time

import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.models.encoder import MAX_RES, length_batches

OUT = paths.RUNS / "phaseA" / "pools"
MODELS = {"esm2_650m": "facebook/esm2_t33_650M_UR50D", "esmc_600m": "biohub/ESMC-600M",
          "prot_t5": "Rostlab/prot_t5_xl_half_uniref50-enc"}


class Pooler:
    """Model-specific tokenisation and forward pass -> residue states [B, L, d] and a residue mask [B, L]."""

    def __init__(self, name: str, dev):
        self.name, self.dev = name, dev
        repo = MODELS[name]
        if name == "esm2_650m":
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(repo)
            self.m = AutoModel.from_pretrained(repo, add_pooling_layer=False, dtype=torch.bfloat16)
        elif name == "esmc_600m":
            from esm.models.esmc import EsmcModel, EsmcTokenizer
            self.tok = EsmcTokenizer.from_pretrained(repo)
            self.m = EsmcModel.from_pretrained(repo, device=dev, dtype=torch.bfloat16, attn_implementation="sdpa")
        else:
            from transformers import T5EncoderModel, T5Tokenizer
            self.tok = T5Tokenizer.from_pretrained(repo, do_lower_case=False)
            self.m = T5EncoderModel.from_pretrained(repo, dtype=torch.float16)
        self.m = self.m.to(dev).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.m.parameters())

    @torch.no_grad()
    def __call__(self, seqs):
        seqs = [s[:MAX_RES] for s in seqs]
        if self.name == "prot_t5":
            seqs = [" ".join(s.translate(str.maketrans("UZOB", "XXXX"))) for s in seqs]
        enc = self.tok(seqs, return_tensors="pt", padding=True)
        ids, am = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16 if self.name != "prot_t5" else torch.float16):
            H = self.m(input_ids=ids, attention_mask=am).last_hidden_state.float()
        m = am.bool().clone()
        lens = am.sum(1)
        m[torch.arange(len(lens)), lens - 1] = False          # <eos> / </s>
        if self.name != "prot_t5":
            m[:, 0] = False                                    # <cls>
        return H, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--max-tokens", type=int, default=60000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = Bundle()
    rows = np.load(paths.RUNS / "full" / "enc_lora35m_f0" / "rows.npy")
    seqs, lens = b.prot.seq.values[rows], b.prot.length.values[rows]
    dev = torch.device("cuda")
    P = Pooler(a.model, dev)
    print(f"[pool] {a.model} ({MODELS[a.model]}): {P.n_params() / 1e6:.0f}M params, frozen; proteins {len(rows):,}; "
          f"length p50 {np.median(lens):.0f} (truncated to {MAX_RES})", flush=True)
    E = None
    batches = length_batches(lens, a.max_tokens, max_batch=256)
    for i, bi in enumerate(batches):
        H, m = P([seqs[j] for j in bi])
        v = (H * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        if E is None:
            E = np.zeros((len(rows), v.shape[1]), np.float16)
            print(f"[pool] hidden size {v.shape[1]}; first batch {tuple(H.shape)}", flush=True)
        E[bi] = v.cpu().numpy().astype(np.float16)
        if i % 200 == 0:
            print(f"[pool] batch {i}/{len(batches)} size {len(bi)} max len {lens[bi].max()} "
                  f"{time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(E.astype(np.float32)).all(), "non-finite pooled embedding"
    np.save(OUT / f"{a.model}_mean.npy", E)
    np.save(OUT / "rows.npy", rows)
    print(f"[pool] wrote {E.shape} {E.dtype} ({E.nbytes / 1e6:.0f} MB) -> {OUT / (a.model + '_mean.npy')}; sample "
          f"{b.ids[rows[0]]} {E[0, :5]}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
