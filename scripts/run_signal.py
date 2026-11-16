"""B1: signal-peptide head on frozen ESM2-35M residue states, SignalP 6.0 data and protocol.

Nested cross-validation as in SignalP 6.0: for each outer test partition, two heads (train on one of the remaining
partitions, early-stop on the other); the test partition is predicted by the product-of-experts ensemble of the two.
Metrics per organism group and SP type: MCC1 (negatives = no-SP), MCC2 (negatives also include the other SP types),
cleavage-site precision / recall at exact position (and +-1..3), a CS counting only in the correct predicted class.

  python scripts/run_signal.py --config configs/signal.yaml [regions=latent --tag latent]
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

from tracego import paths
from tracego.data import signalp as sp
from tracego.models.signal import (FrozenESM, SignalHead, allowed_tensor, decode, ensemble_potentials,
                                   start_ok_tensor, REGION_SUMMARY)
from tracego.train.common import load_config, log, n_params, save_json

TAG = "signal"
# SignalP 6.0 (Teufel et al. 2022), Supplementary Tables 1-6 (benchmark subset of the CV test predictions):
# (MCC1, MCC2, CS recall +-0, CS precision +-0). Sec/SPIII and Tat/SPII are only shown in Fig. 2a.
SP6 = {("ARCHAEA", 1): (0.737, 0.728, 0.500, 0.643), ("EUKARYA", 1): (0.868, 0.868, 0.747, 0.661),
       ("NEGATIVE", 1): (0.811, 0.649, 0.639, 0.534), ("POSITIVE", 1): (0.878, 0.734, 0.800, 0.632),
       ("ARCHAEA", 2): (0.871, 0.719, 0.778, 0.583), ("NEGATIVE", 2): (0.838, 0.841, 0.852, 0.913),
       ("POSITIVE", 2): (0.894, 0.893, 0.875, 0.929),
       ("ARCHAEA", 3): (0.802, 0.807, 0.333, 0.375), ("NEGATIVE", 3): (0.946, 0.934, 0.706, 0.679),
       ("POSITIVE", 3): (0.788, 0.806, 0.556, 0.714)}


def embed(esm: FrozenESM, seqs, batch=512):
    t0 = time.time()
    H = torch.zeros(len(seqs), sp.MAX_LEN, esm.dim, dtype=torch.float16, device=esm.device)
    M = torch.zeros(len(seqs), sp.MAX_LEN, dtype=torch.bool, device=esm.device)
    for i in range(0, len(seqs), batch):
        h, m = esm(seqs[i:i + batch])
        H[i:i + len(h), :h.shape[1]] = h
        M[i:i + len(h), :m.shape[1]] = m
    log(TAG, f"ESM residue states {tuple(H.shape)} fp16 ({H.numel() * 2 / 1e9:.2f} GB on GPU) in {time.time() - t0:.0f}s")
    return H, M


def train_head(cfg, H, M, A, K, tr, va, seed):
    T = cfg["train"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SignalHead(**cfg["head"]).to(H.device)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, T["lr"], total_steps=T["epochs"] * (len(tr) // T["batch"] + 1),
                                                pct_start=0.1)
    best, best_state, bad = np.inf, None, 0
    for ep in range(T["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(tr)
        for i in range(0, len(perm), T["batch"]):
            b = torch.as_tensor(perm[i:i + T["batch"]], device=H.device)
            loss = model(H[b], M[b], A[b], K[b]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = sum(model(H[b], M[b], A[b], K[b]).sum().item()
                     for b in torch.as_tensor(va, device=H.device).split(1024)) / len(va)
        flag = ""
        if vl < best - 1e-4:
            best, bad, flag = vl, 0, " *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or flag or bad >= T["patience"]:
            log(TAG, f"  epoch {ep:2d} train nll {tot / len(tr):.3f} val nll {vl:.3f} ({time.time() - t0:.1f}s){flag}")
        if bad >= T["patience"]:
            break
    model.load_state_dict(best_state)
    return model.eval(), best


def metrics(d: sp.SignalData, pred, cs_pred) -> dict:
    out = {}
    for k in sp.KINGDOMS:
        g = d.kingdom == k
        for c in range(1, len(sp.CLASSES)):
            if (d.cls[g] == c).sum() == 0:
                continue
            m1 = g & np.isin(d.cls, [0, c])
            r = {"n_true": int((d.cls[g] == c).sum()), "n_pred": int((pred[g] == c).sum()),
                 "MCC1": matthews_corrcoef(d.cls[m1] == c, pred[m1] == c),
                 "MCC2": matthews_corrcoef(d.cls[g] == c, pred[g] == c)}
            for tol in range(4):
                hit = g & (d.cls == c) & (pred == c) & (np.abs(cs_pred - d.cs) <= tol)
                r[f"CS_rec_{tol}"] = hit.sum() / max(r["n_true"], 1)
                r[f"CS_prec_{tol}"] = hit.sum() / max(r["n_pred"], 1)
            out[f"{k}|{sp.CLASSES[c]}"] = r
    return out


def print_metrics(res: dict):
    log(TAG, "per-type detection and cleavage-site metrics (CV test predictions; SignalP 6.0 in brackets)")
    print(f"  {'group':9s} {'type':10s} {'n':>5s} {'MCC1':>13s} {'MCC2':>13s} {'CS rec':>13s} {'CS prec':>13s}"
          f"  CS rec/prec +-1..3")
    for key, r in res.items():
        k, cname = key.split("|")
        ref = SP6.get((k, sp.CLASSES.index(cname)))
        f = lambda v, j: f"{v:.3f}" + (f" [{ref[j]:.3f}]" if ref else "        ")
        tol = " ".join(f"{r[f'CS_rec_{t}']:.2f}/{r[f'CS_prec_{t}']:.2f}" for t in (1, 2, 3))
        print(f"  {k:9s} {cname:10s} {r['n_true']:5d} {f(r['MCC1'], 0)} {f(r['MCC2'], 1)} "
              f"{f(r['CS_rec_0'], 2)} {f(r['CS_prec_0'], 3)}  {tol}", flush=True)


def show(seq, path, cs_true, cs_pred, cls_true, cls_pred, acc):
    """Sequence with the predicted regions bracketed, e.g. [n:MKK][h:LLAVLAL][c:SAQA]|DTP..."""
    tags = []
    for s in path[:len(seq)]:
        tags.append("." if sp.STATE_CLASS[s] == 0 else "p" if s == sp.SI["P_SPIII"] else "nhc"[sp.STATE_REGION[s]])
    out, i = "", 0
    while i < len(seq):
        j = i
        while j < len(seq) and tags[j] == tags[i]:
            j += 1
        out += seq[i:j] if tags[i] == "." else f"[{tags[i]}:{seq[i:j]}]"
        if tags[i] != "." and (j == len(seq) or tags[j] == "."):
            out += "|"
        i = j
    print(f"  {acc} true {sp.CLASSES[cls_true]} cs={cs_true} | pred {sp.CLASSES[cls_pred]} cs={cs_pred}\n    {out}",
          flush=True)


def profile(d: sp.SignalData, pred, cs_pred, summ, paths_, out: dict):
    """Physicochemical profile of the predicted regions (correctly typed SPs): length, mean Kyte-Doolittle
    hydrophobicity, net side-chain charge, fraction K/R and fraction D/E; mature 10 residues after the CS as a
    reference. Also the (-3, -1) small-residue rule at the predicted Sec/SPI cleavage site."""
    log(TAG, "tripartite profile of predicted regions (mean +- sd over correctly typed test SPs)")
    print(f"  {'type':10s} {'group':9s} {'n':>5s} {'region':6s} {'len':>10s} {'KD':>11s} {'charge':>11s} "
          f"{'K+R frac':>8s} {'D+E frac':>8s}")
    for c in (1, 2, 3, 4):
        for grp in (["EUKARYA", "BACTERIA"] if c == 1 else ["ALL"]):
            sel = (d.cls == c) & (pred == c) & (cs_pred > 0)
            if grp == "EUKARYA":
                sel &= d.kingdom == "EUKARYA"
            elif grp == "BACTERIA":
                sel &= np.isin(d.kingdom, ["NEGATIVE", "POSITIVE"])
            idx = np.where(sel)[0]
            if len(idx) < 3:
                continue
            rows = {}
            for r, name in enumerate(("n", "h", "c", "mature")):
                L, KDm, CH, KR, DE = [], [], [], [], []
                for i in idx:
                    s = d.seqs[i]
                    if name == "mature":
                        seg = s[cs_pred[i]:cs_pred[i] + 10]
                    else:
                        reg = sp.STATE_REGION[np.maximum(paths_[i][:len(s)], 0)]
                        seg = "".join(a for a, g, st in zip(s, reg, paths_[i]) if g == r and sp.STATE_CLASS[st] > 0)
                    if not seg:
                        continue
                    L.append(len(seg)); KDm.append(sp.kd_profile(seg).mean()); CH.append(sp.charge_profile(seg).sum())
                    KR.append(sum(a in "KR" for a in seg) / len(seg)); DE.append(sum(a in "DE" for a in seg) / len(seg))
                if not L:
                    continue
                rows[name] = {"n": len(L), "len": (np.mean(L), np.std(L)), "kd": (np.mean(KDm), np.std(KDm)),
                              "charge": (np.mean(CH), np.std(CH)), "kr": np.mean(KR), "de": np.mean(DE)}
                v = rows[name]
                print(f"  {sp.CLASSES[c]:10s} {grp:9s} {v['n']:5d} {name:6s} {v['len'][0]:5.1f}+-{v['len'][1]:4.1f} "
                      f"{v['kd'][0]:5.2f}+-{v['kd'][1]:4.2f} {v['charge'][0]:5.2f}+-{v['charge'][1]:4.2f} "
                      f"{v['kr']:8.3f} {v['de']:8.3f}", flush=True)
            out[f"{sp.CLASSES[c]}|{grp}"] = rows
    idx = np.where((d.cls == 1) & (pred == 1) & (cs_pred > 3))[0]
    small = set("AGSCT")
    m1 = np.mean([d.seqs[i][cs_pred[i] - 1] in small for i in idx])
    m3 = np.mean([d.seqs[i][cs_pred[i] - 3] in small for i in idx])
    bg = np.mean([a in small for i in idx for a in d.seqs[i]])
    log(TAG, f"Sec/SPI (-3,-1) rule at predicted CS: small residue (AGSCT) at -1 {m1:.3f}, at -3 {m3:.3f} "
             f"(background {bg:.3f}; n={len(idx)})")
    idx = np.where(np.isin(d.cls, [2, 4]) & (pred == d.cls) & (cs_pred > 0) & (cs_pred < np.array(list(map(len, d.seqs)))))[0]
    log(TAG, f"lipoproteins: Cys at +1 of predicted CS {np.mean([d.seqs[i][cs_pred[i]] == 'C' for i in idx]):.3f} "
             f"(n={len(idx)})")
    out["rule_-1_small"], out["rule_-3_small"], out["bg_small"] = m1, m3, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/signal.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    run = paths.RUNS / "signal" / a.tag
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    log(TAG, f"config {cfg}")
    d = sp.load()
    log(TAG, f"SP length p5/p50/p95 {np.percentile(d.cs[d.cls > 0], [5, 50, 95]).tolist()}; "
             f"sequence length p50 {np.median(list(map(len, d.seqs))):.0f}")

    esm = FrozenESM(cfg["esm"], dev, cfg["layer"])
    H, M = embed(esm, d.seqs)
    del esm
    torch.cuda.empty_cache()
    sets = [sp.region_sets(s, l, c, cfg["regions"]) for s, l, c in zip(d.seqs, d.labels, d.cls)]
    A = allowed_tensor(sets, sp.MAX_LEN, dev)
    K = start_ok_tensor(d.kingdom if cfg["kingdom_mask"] else [None] * len(d), dev)
    amb = np.array([(s.sum(1) > 1).sum() for s in sets])
    log(TAG, f"region sets ({cfg['regions']}): allowed {tuple(A.shape)}; ambiguous residues per SP "
             f"mean {amb[d.cls > 0].mean():.1f}")
    probe = SignalHead(**cfg["head"]).to(dev)
    log(TAG, f"head params {n_params(probe) / 1e6:.3f}M (conv + BiLSTM + {sp.S}-state CRF)")
    with torch.no_grad():
        feas = torch.cat([probe(H[b], M[b], A[b], K[b]) for b in torch.arange(len(d), device=dev).split(2048)])
    assert (feas < 1e3).all(), f"{int((feas >= 1e3).sum())} label sets infeasible under the grammar"
    log(TAG, f"all {len(d):,} label sets feasible under the grammar (untrained nll mean {feas.mean():.2f})")
    del probe

    heads, pred, cs_pred = {}, np.zeros(len(d), int), np.zeros(len(d), int)
    prob, summ, path = np.zeros((len(d), len(sp.CLASSES)), np.float32), np.zeros((len(d), 9), np.float32), \
        np.zeros((len(d), sp.MAX_LEN), int)
    for test in range(3):
        te = np.where(d.part == test)[0]
        inner = [p for p in range(3) if p != test]
        models = []
        for tr_p, va_p in itertools.permutations(inner):
            tr, va = np.where(d.part == tr_p)[0], np.where(d.part == va_p)[0]
            log(TAG, f"outer test {test} | train partition {tr_p} ({len(tr):,}) | early stop on {va_p} ({len(va):,})")
            m, vl = train_head(cfg, H, M, A, K, tr, va, cfg["train"]["seed"] + 10 * test + tr_p)
            heads[(test, va_p)] = m.state_dict()
            models.append(m)
        for b in np.array_split(te, max(1, len(te) // 1024)):
            bt = torch.as_tensor(b, device=dev)
            E, tr_, st = ensemble_potentials(models, H[bt], M[bt])
            r = decode(E, M[bt], tr_, st, K[bt], [d.seqs[i] for i in b])
            pred[b], cs_pred[b], prob[b], summ[b], path[b, :r["path"].shape[1]] = (r["pred"], r["cs"], r["prob"],
                                                                                  r["summary"], r["path"])
        acc = (pred[te] == d.cls[te]).mean()
        log(TAG, f"outer test {test}: class accuracy {acc:.4f} on {len(te):,}")

    conf = np.zeros((len(sp.CLASSES), len(sp.CLASSES)), int)
    np.add.at(conf, (d.cls, pred), 1)
    log(TAG, "confusion (rows true, cols pred: " + ", ".join(sp.CLASSES) + ")\n" + str(conf))
    res = metrics(d, pred, cs_pred)
    print_metrics(res)
    mean = {k: float(np.mean([r[k] for r in res.values()])) for k in ("MCC1", "MCC2", "CS_rec_0", "CS_prec_0")}
    log(TAG, f"macro over group x type: {mean}")

    log(TAG, "sample predictions ([n:...][h:...][c:...]| = predicted regions and cleavage site; p = Sec/SPIII)")
    rng = np.random.default_rng(0)
    for c in range(len(sp.CLASSES)):
        idx = np.where(d.cls == c)[0]
        ok = idx[(pred[idx] == c) & (cs_pred[idx] == d.cs[idx])]
        bad = idx[(pred[idx] != c) | (cs_pred[idx] != d.cs[idx])]
        for i in list(rng.choice(ok, min(2, len(ok)), replace=False)) + list(rng.choice(bad, min(1, len(bad)), replace=False)):
            show(d.seqs[i], path[i], d.cs[i], cs_pred[i], d.cls[i], pred[i], "ok " if i in ok else "ERR")
    prof = {}
    profile(d, pred, cs_pred, summ, path, prof)

    torch.save({"cfg": cfg, "heads": heads, "ids": d.ids.tolist(), "part": d.part.tolist()}, run / "models.pt")
    np.savez_compressed(run / "cv_pred.npz", ids=d.ids, prob=prob.astype(np.float16), pred=pred, cs=cs_pred,
                        summary=summ.astype(np.float16))
    save_json({"cfg": cfg, "metrics": res, "macro": mean, "confusion": conf.tolist(), "profile": prof,
               "sp6_reference": {f"{k}|{sp.CLASSES[c]}": v for (k, c), v in SP6.items()},
               "region_summary_fields": REGION_SUMMARY}, run / "metrics.json")
    log(TAG, f"saved {run}/models.pt, cv_pred.npz, metrics.json")


if __name__ == "__main__":
    main()
