"""Phase D1: per-residue RMSF regression head on ESM2-35M states (ATLAS MD summaries).

  predict_rmsf(seqs, dev) -> list[np.ndarray]   # loads runs/analysis/d1/rmsf_head.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tracego.analysis.common import OUT

HEAD = OUT / "d1" / "rmsf_head.pt"


class RmsfHead(nn.Module):
    def __init__(self, d_in: int = 480, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, H, mask):
        y = self.net(H).squeeze(-1)
        return y * mask.float()


@torch.no_grad()
def predict_rmsf(seqs, dev="cuda", ckpt: Path = HEAD):
    from transformers import AutoModel
    from tracego.models.encoder import esm_tokenizer, tokenize
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    head = RmsfHead(st.get("d_in", 480), st.get("d", 128)).to(dev)
    head.load_state_dict(st["state"])
    head.eval()
    esm = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D", add_pooling_layer=False).to(dev).eval()
    tok = esm_tokenizer("facebook/esm2_t12_35M_UR50D")
    out = []
    for s in seqs:
        ids, am = tokenize(tok, [s], torch.device(dev))
        H = esm(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.bool(); m[:, 0] = False; m[0, am.sum() - 1] = False
        y = head(H, m)[0, 1:1 + len(s)].float().cpu().numpy()
        out.append(y)
    return out
