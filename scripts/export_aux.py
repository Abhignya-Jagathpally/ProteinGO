"""B3/B4: export signal + localization auxiliary evidence for TraceGO protein sets, and TV-only loc→CC AUPR.

Writes fp16 npz under runs/aux/:
  signal_{train,tv,test,fr24p}.npz  ids, sp_prob[6], cleavage, summary[9]
  loc_{train,tv,test,fr24p}.npz     ids, loc_prob[10]

B4 scores only the TV split (never test / fresh24).

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/export_aux.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from tracego import paths
from tracego.data import deeploc as dl
from tracego.data import signalp as sp
from tracego.data.bundle import Bundle
from tracego.models.encoder import length_batches
from tracego.models.localization import LocPredictor
from tracego.models.signal import FrozenESM, SignalPredictor, REGION_SUMMARY
from tracego.train.common import log, save_json

TAG = "aux"
SPLITS = ("train", "tv", "test", "fr24p")
# NCBI taxids
SK = {2759: "EUKARYA", 2: None, 2157: "ARCHAEA"}  # Bacteria: None (all SP types allowed)


def kingdoms_for(b: Bundle, rows: np.ndarray) -> list:
    sk = b.taxa.reindex(b.prot.ncbi_taxon.values[rows]).superkingdom.fillna(0).astype(int).values
    return [SK.get(int(x), None) for x in sk]


@torch.no_grad()
def predict_signal(pred: SignalPredictor, esm: FrozenESM, seqs, ids, kings, max_tokens=14000, max_batch=256):
    n = len(seqs)
    prob = np.zeros((n, len(sp.CLASSES)), np.float32)
    cs = np.zeros(n, np.int16)
    summ = np.zeros((n, 9), np.float32)
    lens = np.array([min(len(s), sp.MAX_LEN) for s in seqs])
    for local in length_batches(lens, max_tokens, max_batch):
        sl = [seqs[i][:sp.MAX_LEN] for i in local]
        H, M = esm(sl)
        if H.shape[1] < sp.MAX_LEN:
            H = F.pad(H, (0, 0, 0, sp.MAX_LEN - H.shape[1]))
            M = F.pad(M, (0, sp.MAX_LEN - M.shape[1]))
        else:
            H, M = H[:, :sp.MAX_LEN], M[:, :sp.MAX_LEN]
        r = pred.predict(H, M, [ids[i] for i in local], [kings[i] for i in local], sl)
        prob[local], cs[local], summ[local] = r["prob"], r["cs"], r["summary"]
    return prob, cs, summ


@torch.no_grad()
def predict_loc(pred: LocPredictor, esm: FrozenESM, seqs, ids, sig10, max_tokens=12000, max_batch=64):
    n = len(seqs)
    P = np.zeros((n, pred.cfg["head"]["n_out"]), np.float32)
    cropped = [dl.crop(s) for s in seqs]
    lens = np.array([len(s) for s in cropped])
    for local in length_batches(lens, max_tokens, max_batch):
        H, M = esm([cropped[i] for i in local])
        s = None if not pred.use_signal else torch.as_tensor(sig10[local], device=esm.device)
        # LocPredictor.predict expects full H for the ids subset — call heads directly for a contiguous batch
        heads = list(pred.models.values())
        # OOF when possible
        out = np.zeros((len(local), P.shape[1]), np.float32)
        groups = {}
        for j, i in enumerate(local):
            g = pred.part.get(ids[i], -1)
            groups.setdefault(g, []).append(j)
        for g, js in groups.items():
            ms = [m for f, m in pred.models.items() if g < 0 or f == g] or heads
            jt = torch.as_tensor(js, device=esm.device)
            sg = None if s is None else s[jt]
            logits = torch.stack([m(H[jt], M[jt], sg) for m in ms]).mean(0)
            out[js] = torch.sigmoid(logits).float().cpu().numpy()
        P[local] = out
    return P


def tv_cc_aupr(ids, loc_prob, out_json):
    """B4: per-location AUPR of loc probabilities vs TV ground-truth CC GO terms."""
    gt = pd.read_csv(paths.TV2 / "gt_tv.tsv", sep="\t", header=None, names=["id", "term"])
    id_to_i = {x: i for i, x in enumerate(ids)}
    rows = []
    for loc, go in dl.GO.items():
        j = dl.LOCS.index(loc)
        pos = set(gt.loc[gt.term == go, "id"])
        y = np.array([1 if ids[i] in pos else 0 for i in range(len(ids))], np.int8)
        # score only proteins that appear in gt_tv (have some annotation) OR all TV ids
        if y.sum() == 0:
            ap = float("nan")
        else:
            ap = float(average_precision_score(y, loc_prob[:, j]))
        prev = float(y.mean())
        rows.append({"location": loc, "go": go, "n_pos": int(y.sum()), "prevalence": prev, "AUPR": ap,
                     "AUPR_minus_prev": ap - prev if ap == ap else float("nan")})
        log(TAG, f"B4 {loc:24s} {go} n_pos={y.sum():4d} prev={prev:.4f} AUPR={ap:.4f}")
    # DeepLoc-train overlap among TV ids
    dl_ids = set(pd.read_csv(paths.RUNS / "aux" / "raw" / "deeploc2_swissprot.csv", usecols=["ACC"])["ACC"].astype(str))
    overlap = sum(i in dl_ids for i in ids)
    log(TAG, f"B4 TV proteins overlapping DeepLoc train: {overlap}/{len(ids)}")
    if overlap:
        mask = np.array([i not in dl_ids for i in ids])
        extra = []
        for loc, go in dl.GO.items():
            j = dl.LOCS.index(loc)
            pos = set(gt.loc[gt.term == go, "id"])
            y = np.array([1 if ids[i] in pos else 0 for i in np.where(mask)[0]], np.int8)
            p = loc_prob[mask, j]
            ap = float(average_precision_score(y, p)) if y.sum() and y.sum() < len(y) else float("nan")
            extra.append({"location": loc, "go": go, "n_pos": int(y.sum()), "AUPR": ap})
        save_json({"per_location": rows, "overlap_deeploc": overlap, "n_tv": len(ids),
                   "heldout_of_deeploc": extra}, out_json)
    else:
        save_json({"per_location": rows, "overlap_deeploc": overlap, "n_tv": len(ids)}, out_json)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-run", default="runs/signal/main")
    ap.add_argument("--loc-run", default="runs/loc/main")
    ap.add_argument("--loc-variant", default="with_signal")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    a = ap.parse_args()
    out = paths.RUNS / "aux"
    out.mkdir(parents=True, exist_ok=True)
    df_free = __import__("os").popen("df -h /").read()
    log(TAG, f"disk before export:\n{df_free}")

    sig_dir = paths.TRACEGO / a.signal_run
    loc_dir = paths.TRACEGO / a.loc_run
    assert (sig_dir / "models.pt").exists(), sig_dir
    assert (loc_dir / "models.pt").exists(), loc_dir

    dev = torch.device("cuda")
    b = Bundle()
    esm = FrozenESM("facebook/esm2_t12_35M_UR50D", dev)
    spre = SignalPredictor(sig_dir, dev)
    lpre = LocPredictor(loc_dir, dev, variant=a.loc_variant)
    log(TAG, f"signal heads {len(spre.models)}; loc heads {len(lpre.models)} variant={a.loc_variant}")

    report = {}
    for split in a.splits:
        rows = b.rows(split)
        ids = b.ids[rows]
        seqs = b.prot.seq.values[rows]
        kings = kingdoms_for(b, rows)
        log(TAG, f"=== {split}: {len(rows):,} proteins ===")
        t0 = time.time()
        prob, cs, summ = predict_signal(spre, esm, seqs, ids, kings)
        sig10 = np.concatenate([prob, cs[:, None] / sp.MAX_LEN, summ[:, :3] / sp.MAX_LEN], 1).astype(np.float32)
        sp_path = out / f"signal_{split}.npz"
        np.savez_compressed(sp_path, ids=ids, sp_prob=prob.astype(np.float16), cleavage=cs,
                            summary=summ.astype(np.float16),
                            fields_sp_prob=np.array(sp.CLASSES), fields_summary=np.array(REGION_SUMMARY))
        log(TAG, f"  wrote {sp_path.name} sp_prob {prob.shape} "
                 f"mean p(SP)={1 - prob[:, 0].mean():.3f} ({time.time() - t0:.0f}s)")

        t1 = time.time()
        loc_p = predict_loc(lpre, esm, seqs, ids, sig10)
        lp_path = out / f"loc_{split}.npz"
        np.savez_compressed(lp_path, ids=ids, loc_prob=loc_p.astype(np.float16),
                            fields_loc=np.array(dl.LOCS))
        log(TAG, f"  wrote {lp_path.name} loc_prob {loc_p.shape} "
                 f"mean labels@{0.5}={(loc_p >= 0.5).sum(1).mean():.2f} ({time.time() - t1:.0f}s)")
        report[split] = {"n": int(len(rows)), "signal": str(sp_path), "loc": str(lp_path),
                         "mean_sp": float(1 - prob[:, 0].mean()),
                         "mean_loc_ge_0.5": float((loc_p >= 0.5).sum(1).mean())}

        if split == "tv":
            tv_cc_aupr(ids, loc_p, out / "b4_tv_cc_aupr.json")

    save_json(report, out / "export_report.json")
    log(TAG, f"disk after export:\n{__import__('os').popen('df -h /').read()}")
    log(TAG, "done")


if __name__ == "__main__":
    main()
