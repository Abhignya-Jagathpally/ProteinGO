"""Subcellular-localization head: multi-head attention pooling over frozen ESM2 residue states (DeepLoc 2.0 style:
per-residue scores smoothed by a width-5 Gaussian before the softmax, since sorting signals are contiguous), plus a
masked mean-pool, optionally concatenated with the 10-d signal-peptide features of `tracego.models.signal.features`,
then an MLP to 10 multi-label logits.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocHead(nn.Module):
    def __init__(self, d_in: int = 480, d: int = 256, heads: int = 4, n_out: int = 10, d_sig: int = 0,
                 hidden: int = 512, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, d), nn.GELU())
        self.score = nn.Linear(d, heads)
        g = torch.exp(-0.5 * torch.arange(-2, 3, dtype=torch.float32) ** 2)
        self.register_buffer("smooth", (g / g.sum()).view(1, 1, 5).repeat(heads, 1, 1))
        self.heads = heads
        self.d_sig = d_sig
        self.mlp = nn.Sequential(nn.LayerNorm(d * (heads + 1) + d_sig), nn.Dropout(dropout),
                                 nn.Linear(d * (heads + 1) + d_sig, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_out))

    def forward(self, H, mask, sig=None, return_att: bool = False):
        x = self.proj(H.float())
        s = self.score(x).transpose(1, 2)                                # [B, heads, L]
        s = F.conv1d(s, self.smooth, padding=2, groups=self.heads)
        s = s.masked_fill(~mask[:, None], -1e4)
        att = s.softmax(-1)
        pooled = torch.einsum("bhl,bld->bhd", att, x).flatten(1)
        mean = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = [pooled, mean]
        if self.d_sig:
            z.append(sig.float())
        logits = self.mlp(torch.cat(z, -1))
        return (logits, att) if return_att else logits


def mcc_threshold(y: np.ndarray, p: np.ndarray, grid=np.arange(0.02, 0.98, 0.01)) -> tuple[float, float]:
    """Threshold maximising MCC of one label."""
    best = (0.5, -1.0)
    y = y.astype(bool)
    for t in grid:
        pr = p >= t
        tp = float((pr & y).sum()); fp = float((pr & ~y).sum()); fn = float((~pr & y).sum())
        tn = float(len(y) - tp - fp - fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        m = (tp * tn - fp * fn) / den if den > 0 else 0.0
        if m > best[1]:
            best = (float(t), m)
    return best


def apply_thresholds(P: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Multi-label decision with at least one label per protein (argmax of p / threshold when nothing passes)."""
    D = P >= thr[None]
    none = ~D.any(1)
    D[none, np.argmax(P[none] / thr[None], 1)] = True
    return D


def multilabel_scores(Y: np.ndarray, P: np.ndarray, thr: np.ndarray) -> dict:
    """Exact-match accuracy, Jaccard, micro/macro F1, mean predicted label count, per-label MCC/F1."""
    from sklearn.metrics import f1_score, matthews_corrcoef
    D = apply_thresholds(P, thr)
    inter = (D & Y.astype(bool)).sum(1)
    union = (D | Y.astype(bool)).sum(1)
    per = {}
    for j in range(Y.shape[1]):
        per[j] = {"MCC": float(matthews_corrcoef(Y[:, j], D[:, j])),
                  "F1": float(f1_score(Y[:, j], D[:, j], zero_division=0))}
    return {"accuracy": float((D == Y.astype(bool)).all(1).mean()),
            "jaccard": float(np.mean(inter / np.maximum(union, 1))),
            "micro_f1": float(f1_score(Y, D, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(Y, D, average="macro", zero_division=0)),
            "pred_labels": float(D.sum(1).mean()),
            "per_label": per}


class LocPredictor:
    """Fold heads saved by scripts/run_loc.py. DeepLoc accessions are predicted out-of-fold; every other protein
    by the mean of all fold heads."""

    def __init__(self, run_dir, device, variant: str = "with_signal"):
        import torch
        ck = torch.load(run_dir / "models.pt", map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.variant = variant
        self.use_signal = bool(ck["variants"][variant]["use_signal"])
        self.thr = np.asarray(ck["variants"][variant]["thr"], np.float32)
        self.models = {}
        for fold, sd in ck["variants"][variant]["heads"].items():
            m = LocHead(**{**self.cfg["head"], "d_sig": 10 if self.use_signal else 0}).to(device).eval()
            m.load_state_dict(sd)
            self.models[int(fold)] = m
        self.part = {i: int(p) for i, p in zip(ck["ids"], ck["part"])}
        self.device = device

    def groups(self, ids) -> dict:
        p = np.array([self.part.get(i, -1) for i in ids])
        return {int(g): np.where(p == g)[0] for g in np.unique(p)}

    @torch.no_grad()
    def predict(self, H, mask, ids, sig=None) -> np.ndarray:
        """Return [n, 10] probabilities (sigmoid of ensembled logits)."""
        import torch
        out = np.zeros((len(ids), self.cfg["head"]["n_out"]), np.float32)
        for g, pos in self.groups(ids).items():
            heads = [m for f, m in self.models.items() if g < 0 or f == g]
            pt = torch.as_tensor(pos, device=self.device)
            s = None if sig is None or not self.use_signal else sig[pt]
            logits = torch.stack([m(H[pt], mask[pt], s) for m in heads]).mean(0)
            out[pos] = torch.sigmoid(logits).float().cpu().numpy()
        return out
