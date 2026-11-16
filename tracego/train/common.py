"""Config loading, fold assignment, label helpers and logging shared by the training stages."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml

from tracego import paths

T_START = time.time()


def log(tag: str, msg: str):
    print(f"[{tag} {time.strftime('%H:%M:%S')} +{time.time() - T_START:6.0f}s] {msg}", flush=True)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(open(path))
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cur = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            cur = cur.setdefault(kk, {})
        cur[ks[-1]] = yaml.safe_load(v)
    return cfg


def train_rows(bundle, subset: int = 0, seed: int = 0) -> np.ndarray:
    """v2 training rows, optionally a seeded subset (development scale)."""
    rows = bundle.rows("train")
    if subset and subset < len(rows):
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(rows, subset, replace=False))
    return rows


def fold_of(rows: np.ndarray, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Seeded fold id per row (depends only on the row index, so subsets and the full set agree)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(200000)
    return perm[rows] % n_folds


def term_weights(ia: np.ndarray, use_ia: bool) -> np.ndarray:
    w = (ia + 1.0) if use_ia else np.ones_like(ia)
    return (w / w.mean()).astype(np.float32)


def known_mask(Y: sp.csr_matrix, aspect: np.ndarray) -> np.ndarray:
    """[n, 3] True where the protein has any label in namespace F/P/C."""
    out = np.zeros((Y.shape[0], 3), bool)
    for k, a in enumerate("FPC"):
        out[:, k] = np.asarray(Y[:, np.where(aspect == a)[0]].sum(1)).ravel() > 0
    return out


def dense_rows(Y: sp.csr_matrix, rows, device) -> torch.Tensor:
    return torch.as_tensor(Y[rows].toarray(), dtype=torch.float32, device=device)


def topk_csr(S: np.ndarray, k: int = 300, min_score: float = 0.005) -> sp.csr_matrix:
    """Dense scores [n, V] -> CSR keeping the top-k per row above min_score."""
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(S, idx, 1)
    rows = np.repeat(np.arange(len(S)), k)
    m = val.ravel() >= min_score
    return sp.csr_matrix((val.ravel()[m], (rows[m], idx.ravel()[m])), shape=S.shape)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)


def n_params(model, trainable=True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)
