"""B2: DeepLoc-style multi-label attention-pool head on frozen ESM2-35M, with ± signal-peptide feature ablation.

5-fold CV over the DeepLoc 2.0 homology partitions. Residue states are cached once on CPU (fp16, ragged) so the
head trains without re-running ESM each epoch. Signal features come from the B1 nested-CV heads.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/run_loc.py --config configs/loc.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from tracego import paths
from tracego.data import deeploc as dl
from tracego.data import signalp as sp
from tracego.models.encoder import length_batches
from tracego.models.localization import LocHead, mcc_threshold, multilabel_scores
from tracego.models.signal import FrozenESM, SignalPredictor, features as sig_features
from tracego.train.common import load_config, log, n_params, save_json

TAG = "loc"
# DeepLoc 2.0 Table 1 (ProtT5 column) on SwissProt CV
DL2 = {"Cytoplasm": 0.62, "Nucleus": 0.69, "Extracellular": 0.85, "Cell membrane": 0.66,
       "Mitochondrion": 0.76, "Plastid": 0.90, "Endoplasmic reticulum": 0.56, "Lysosome/Vacuole": 0.28,
       "Golgi apparatus": 0.34, "Peroxisome": 0.56,
       "micro_f1": 0.73, "macro_f1": 0.66, "accuracy": 0.55, "jaccard": 0.69}


@torch.no_grad()
def cache_residue_states(esm: FrozenESM, seqs: list[str], max_tokens: int = 24000, max_batch: int = 64):
    """CPU fp16 residue states (ragged list indexed by protein id) + lengths."""
    t0 = time.time()
    lens = np.array([len(s) for s in seqs])
    H_list = [None] * len(seqs)
    n_tok = 0
    for local in length_batches(lens, max_tokens, max_batch):
        h, m = esm([seqs[i] for i in local])
        for j, i in enumerate(local):
            L = int(m[j].sum())
            H_list[i] = h[j, :L].detach().cpu().half().contiguous()
            n_tok += L
        del h, m
        torch.cuda.empty_cache()
    assert all(x is not None for x in H_list)
    log(TAG, f"cached {len(seqs):,} residue states on CPU ({n_tok * 480 * 2 / 1e9:.2f} GB tokens) "
             f"in {time.time() - t0:.0f}s")
    return H_list, lens

def collate(H_list, idx, device):
    Ls = [H_list[i].shape[0] for i in idx]
    L = max(Ls)
    H = torch.zeros(len(idx), L, H_list[idx[0]].shape[-1], dtype=torch.float16, device=device)
    M = torch.zeros(len(idx), L, dtype=torch.bool, device=device)
    for j, i in enumerate(idx):
        H[j, :Ls[j]] = H_list[i].to(device, non_blocking=True)
        M[j, :Ls[j]] = True
    return H, M


@torch.no_grad()
def signal_matrix(pred: SignalPredictor, esm: FrozenESM, seqs: list[str], ids, batch: int = 256) -> np.ndarray:
    """10-d signal features for every sequence (first 70 residues)."""
    out = np.zeros((len(seqs), 10), np.float32)
    kings = np.array(["EUKARYA"] * len(seqs))
    for i in range(0, len(seqs), batch):
        sl = [s[:sp.MAX_LEN] for s in seqs[i:i + batch]]
        H, M = esm(sl)
        if H.shape[1] < sp.MAX_LEN:
            pad = sp.MAX_LEN - H.shape[1]
            H = F.pad(H, (0, 0, 0, pad))
            M = F.pad(M, (0, pad))
        else:
            H, M = H[:, :sp.MAX_LEN], M[:, :sp.MAX_LEN]
        r = pred.predict(H, M, ids[i:i + batch], kings[i:i + batch], sl)
        out[i:i + len(sl)] = sig_features(r)
    return out


def train_one(cfg, H_list, lens, Y, sig, tr, va, use_signal, device, seed):
    T = cfg["train"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LocHead(**{**cfg["head"], "d_sig": 10 if use_signal else 0}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["wd"])
    best, best_state, bad = np.inf, None, 0
    n_batches = max(1, len(length_batches(lens[tr], T["max_tokens"], T["max_batch"])))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, T["epochs"] * n_batches))
    step_i = 0

    def run_batches(idx, train: bool):
        nonlocal step_i
        model.train(train)
        tot, n = 0.0, 0
        batches = length_batches(lens[idx], T["max_tokens"], T["max_batch"], shuffle=train,
                                 rng=rng if train else None)
        for local in batches:
            b = idx[local]
            H, M = collate(H_list, b, device)
            s = None if not use_signal else torch.as_tensor(sig[b], device=device)
            y = torch.as_tensor(Y[b], device=device)
            with torch.set_grad_enabled(train):
                logits = model(H, M, s)
                # plain BCE (pos_weight collapsed the previous run to ~0.5 probs)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    if step_i < sched.T_max:
                        sched.step()
                        step_i += 1
            tot += loss.item() * len(b)
            n += len(b)
        return tot / max(n, 1)
    for ep in range(T["epochs"]):
        t0 = time.time()
        tl = run_batches(tr, True)
        with torch.no_grad():
            vl = run_batches(va, False)
        flag = ""
        if vl < best - 1e-4:
            best, bad, flag = vl, 0, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 2 == 0 or flag or bad >= T["patience"]:
            log(TAG, f"    epoch {ep:2d} train {tl:.4f} val {vl:.4f} ({time.time() - t0:.1f}s){flag}")
        if bad >= T["patience"]:
            break
    model.load_state_dict(best_state)
    return model.eval().to(device), best


@torch.no_grad()
def predict_idx(model, H_list, lens, sig, idx, use_signal, cfg, device) -> np.ndarray:
    T = cfg["train"]
    P = np.zeros((len(idx), cfg["head"]["n_out"]), np.float32)
    for local in length_batches(lens[idx], T["max_tokens"], T["max_batch"]):
        b = idx[local]
        H, M = collate(H_list, b, device)
        s = None if not use_signal else torch.as_tensor(sig[b], device=device)
        P[local] = torch.sigmoid(model(H, M, s)).float().cpu().numpy()
    return P


def fit_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array([mcc_threshold(Y[:, j], P[:, j])[0] for j in range(Y.shape[1])], np.float32)


def print_scores(name: str, scores: dict):
    log(TAG, f"{name}: acc {scores['accuracy']:.3f} [{DL2['accuracy']}]  "
             f"jaccard {scores['jaccard']:.3f} [{DL2['jaccard']}]  "
             f"microF1 {scores['micro_f1']:.3f} [{DL2['micro_f1']}]  "
             f"macroF1 {scores['macro_f1']:.3f} [{DL2['macro_f1']}]  "
             f"pred labels {scores['pred_labels']:.2f} (true 1.27)")
    print(f"  {'location':24s} {'MCC':>8s} {'DL2':>8s} {'F1':>8s}", flush=True)
    for j, loc in enumerate(dl.LOCS):
        r = scores["per_label"][j]
        print(f"  {loc:24s} {r['MCC']:8.3f} {DL2[loc]:8.3f} {r['F1']:8.3f}", flush=True)


def run_variant(cfg, H_list, lens, d: dl.LocData, sig, use_signal: bool, device):
    name = "with_signal" if use_signal else "no_signal"
    log(TAG, f"=== ablation {name} (d_sig={10 if use_signal else 0}) ===")
    probe = LocHead(**{**cfg["head"], "d_sig": 10 if use_signal else 0})
    log(TAG, f"head params {n_params(probe) / 1e6:.3f}M")
    del probe
    P = np.zeros_like(d.Y)
    heads = {}
    for test in range(5):
        te = np.where(d.part == test)[0]
        va_p = (test + 1) % 5
        va = np.where(d.part == va_p)[0]
        tr = np.where(~np.isin(d.part, [test, va_p]))[0]
        log(TAG, f"  fold test={test} ({len(te):,}) val={va_p} ({len(va):,}) train={len(tr):,}")
        m, vl = train_one(cfg, H_list, lens, d.Y, sig, tr, va, use_signal, device, cfg["train"]["seed"] + test)
        heads[test] = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        P[te] = predict_idx(m, H_list, lens, sig, te, use_signal, cfg, device)
        log(TAG, f"  fold {test} done (best val {vl:.4f})")
        del m
        torch.cuda.empty_cache()
    thr = fit_thresholds(d.Y, P)
    scores = multilabel_scores(d.Y, P, thr)
    print_scores(name, scores)
    return {"use_signal": use_signal, "heads": heads, "thr": thr.tolist(), "scores": scores,
            "prob": P.astype(np.float16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/loc.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    run = paths.RUNS / "loc" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    log(TAG, f"config {cfg}")
    d = dl.load()
    esm = FrozenESM(cfg["esm"], dev, cfg["layer"])

    sig_path = paths.TRACEGO / cfg["signal_run"]
    if not (sig_path / "models.pt").exists():
        raise FileNotFoundError(f"B1 models missing at {sig_path}/models.pt; finish SignalP training first")
    log(TAG, f"computing signal features from {sig_path}")
    spre = SignalPredictor(sig_path, dev)
    sig = signal_matrix(spre, esm, d.seqs, d.ids)
    log(TAG, f"signal features {sig.shape}; mean SP-present p={sig[:, 1:].sum(1).mean():.3f}; "
             f"sample {d.ids[0]} {np.round(sig[0], 3).tolist()}")
    del spre
    torch.cuda.empty_cache()

    H_list, lens = cache_residue_states(esm, d.seqs, cfg["train"]["max_tokens"] * 2, cfg["train"]["max_batch"])
    del esm
    torch.cuda.empty_cache()

    variants = {}
    for use_signal in (False, True):
        variants["with_signal" if use_signal else "no_signal"] = run_variant(
            cfg, H_list, lens, d, sig, use_signal, dev)

    np.savez_compressed(run / "cv_pred.npz",
                        ids=d.ids,
                        no_signal=variants["no_signal"].pop("prob"),
                        with_signal=variants["with_signal"].pop("prob"),
                        Y=d.Y.astype(np.float16))
    torch.save({"cfg": cfg, "ids": d.ids.tolist(), "part": d.part.tolist(), "variants": variants},
               run / "models.pt")
    save_json({"cfg": cfg, "deeploc2_reference": DL2, "locations": list(dl.LOCS),
               "no_signal": variants["no_signal"]["scores"],
               "with_signal": variants["with_signal"]["scores"],
               "delta_macro_f1": (variants["with_signal"]["scores"]["macro_f1"] -
                                 variants["no_signal"]["scores"]["macro_f1"])},
              run / "metrics.json")
    log(TAG, f"saved {run}/models.pt, cv_pred.npz, metrics.json")


if __name__ == "__main__":
    main()
