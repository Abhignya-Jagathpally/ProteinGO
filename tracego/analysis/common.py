"""Helpers shared by the Phase C/D analyses: output locations, disk guard, TV v2 identity bins, per-protein F,
loading the frozen fold encoders and the existing TV v2 predictions / evidence.

Scoring is TV v2 only (`Evaluator("tv")`); the GO-GPT test and fresh24 are never read here.
"""
from __future__ import annotations

import shutil
import time

import numpy as np
import pandas as pd
import torch

from tracego import paths
from tracego.train.common import load_config, log  # noqa: F401  (re-exported for the analysis scripts)

OUT = paths.RUNS / "analysis"
FULL = paths.RUNS / "full"
CFG = paths.TRACEGO / "configs" / "full.yaml"
ASP = ("MF", "BP", "CC")


def out_dir(name: str):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def disk_free_gb(path="/") -> float:
    return shutil.disk_usage(path).free / 1e9


def disk_guard(tag: str, need_gb: float, floor_gb: float = 8.0):
    """Refuses to write `need_gb` when the shared disk would fall below `floor_gb` free."""
    free = disk_free_gb()
    log(tag, f"disk free {free:.1f} GB; about to write ~{need_gb:.2f} GB")
    if free - need_gb < floor_gb:
        raise RuntimeError(f"disk guard: {free:.1f} GB free, {need_gb:.2f} GB requested, floor {floor_gb} GB")


def du_gb(path) -> float:
    return sum(f.stat().st_size for f in paths.Path(path).rglob("*") if f.is_file()) / 1e9


# ---------------------------------------------------------------- TV v2 queries, identity bins, per-protein F
def tv_identity(ids) -> np.ndarray:
    """Best DIAMOND identity of each TV v2 query to the training references (descriptive bins only)."""
    c = pd.read_parquet(paths.TV2 / "cells.parquet")
    c = c[c.split == "tv"].drop_duplicates("id").set_index("id")
    return c.max_pident.reindex(np.asarray(ids).astype(str)).fillna(0).values.astype(np.float32)


def per_protein_f(ev, cur: dict, ns: str) -> pd.DataFrame:
    """Per-protein IA-weighted precision / recall / F at the benchmark-level Fmax threshold of namespace `ns`
    (proteins without a prediction at that threshold get precision 0)."""
    c = cur[ns]
    ne = len(c["idx"])
    nh = c["has"].sum(0)
    P = np.divide(c["prec"].sum(0), nh, out=np.zeros(99), where=nh > 0)
    R = c["rec"].sum(0) / ne
    F = np.divide(2 * P * R, P + R, out=np.zeros(99), where=(P + R) > 0)
    t = int(F.argmax())
    p, r = c["prec"][:, t].astype(np.float64), c["rec"][:, t].astype(np.float64)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return pd.DataFrame({"id": ev.sc.ids[c["idx"]], "prec": p, "rec": r, "f": f, "tau_idx": t, "Fmax": F.max()})


def fmax_subset(ev, cur: dict, ids) -> dict:
    return ev.fmax(cur, subset=list(np.asarray(ids).astype(str)))


# ---------------------------------------------------------------- frozen TraceGO artefacts
def tv_predictions(which: str = "vote"):
    """(P over ontology columns, ids) of the existing TV v2 predictions: 'vote' = retrieval-only neighbour vote,
    'tracego' = the anchored decoder (best TV readout), 'clf' = classifier head only."""
    from tracego.eval.score import load_pred_npz
    import json
    if which == "vote":
        f = FULL / "knn_ca" / "pred_tv.npz"
    elif which == "tracego":
        s = json.load(open(sorted((FULL / "dec_anchor").glob("pred_summary_*.json"))[-1]))
        tv = {k.split("/")[1]: v["mean_testlike"] for k, v in s.items() if k.startswith("tv/")}
        f = FULL / "dec_anchor" / f"pred_tv_{max(tv, key=tv.get)}.npz"
    elif which == "clf":
        f = FULL / "dec_clf" / "pred_tv_clf.npz"
    else:
        raise KeyError(which)
    return load_pred_npz(f)


def evidence(fold: int, space: str = "ca"):
    """Neighbour lists of one cross-fitting space: npz with tv_q (query rows), tv_I (reference rows, top-50 after
    chain-identity re-rank), tv_S (re-ranked score), and the same for train queries."""
    return np.load(FULL / f"enc_lora35m_f{fold}" / f"evidence_{space}.npz")


def cached(fold: int, key: str, frozen: bool = False):
    """(rows, memmapped array) of the cached Z / r / h of a fold encoder."""
    run = FULL / f"enc_{'frozen' if frozen else 'lora'}35m_f{fold}"
    return np.load(run / "rows.npy"), np.load(run / f"{key}.npy", mmap_mode="r")


def load_fold_encoder(fold: int, b, dev, frozen: bool = False):
    """The trained AdaptedEncoder of one fold (ESM2-35M + LoRA layers 9-12 + resampler + term head), eval mode."""
    from tracego.models.adapter import AdaptedEncoder
    cfg = load_config(str(CFG))
    E = cfg["encoder"]
    tag = "frozen35m" if frozen else E["tag"]
    st = torch.load(FULL / f"enc_{tag}_f{fold}" / "model.pt", map_location=dev, weights_only=False)
    m = AdaptedEncoder(E["esm"], len(b.terms), lora_r=0 if frozen else E["lora_r"], n_latents=E["n_latents"],
                       depth=E.get("depth", 2), lora_from_layer=E.get("lora_from_layer", 0)).to(dev)
    res = m.load_state_dict(st["state"], strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    return m.eval()


class Timer:
    def __init__(self, tag: str, what: str):
        self.tag, self.what = tag, what

    def __enter__(self):
        self.t0 = time.time()
        log(self.tag, f"{self.what} ...")
        return self

    def __exit__(self, *a):
        log(self.tag, f"{self.what} done in {time.time() - self.t0:.0f}s")
