"""Stage A: train the adapted encoder (ESM2 + LoRA or frozen, GO-grounded resampler, retrieval head) on one
cross-fitting fold of the training proteins, then cache Z / r / h for every protein needed downstream.

  python -m tracego.train.train_encoder --config configs/dev.yaml --fold 0 [key=value ...]
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.data.godefs import build as build_defs, latent_init
from tracego.models.adapter import AdaptedEncoder, ia_bce, soft_contrastive
from tracego.models.encoder import esm_tokenizer, length_batches, tokenize
from tracego.train.common import (dense_rows, fold_of, known_mask, load_config, log, n_params, save_json,
                                  term_weights, topk_csr, train_rows)

TAG = "encA"


def known_terms(K: np.ndarray, aspect_idx: np.ndarray, rows) -> torch.Tensor:
    return torch.as_tensor(K[rows][:, aspect_idx], dtype=torch.float32)


@torch.no_grad()
def encode(model, tok, seqs, lens, dev, max_tokens=60000, want=("Z", "r", "h"), logits=False):
    model.eval()
    out = {k: None for k in want}
    n = len(seqs)
    res = {}
    for bi in length_batches(lens, max_tokens):
        ids, am = tokenize(tok, [seqs[j] for j in bi], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(ids, am)
            if logits:
                o["p"] = torch.sigmoid(model.term_logits(o["Z"]).float())
        for k in list(want) + (["p"] if logits else []):
            v = o[k].float().cpu().numpy().astype(np.float16 if k != "p" else np.float32)
            if k not in res:
                res[k] = np.zeros((n,) + v.shape[1:], v.dtype)
            res[k][bi] = v
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    E = cfg["encoder"]
    run = paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{a.fold}"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0) + a.fold)
    rng = np.random.default_rng(cfg.get("seed", 0) + a.fold)
    dev = torch.device("cuda")

    b = Bundle()
    rows_all = train_rows(b, cfg["data"]["subset"], cfg.get("seed", 0))
    fold = fold_of(rows_all)
    rows = rows_all[fold == a.fold]
    log(TAG, f"config {cfg['name']} encoder {E['tag']} fold {a.fold}: training proteins {len(rows):,} "
             f"of {len(rows_all):,} (other fold {int((fold != a.fold).sum()):,} held out from this adapter)")
    Y = b.Y_t0
    aspect_idx = np.array(["FPC".index(x) for x in b.aspect])
    K = known_mask(Y, b.aspect)
    w_term = torch.as_tensor(term_weights(b.ia, E.get("ia_weight", True)), device=dev)
    seqs, lens = b.prot.seq.values, b.prot.length.values
    log(TAG, f"labels: vocab {Y.shape[1]:,}; mean terms/protein {np.diff(Y[rows].indptr).mean():.1f}; "
             f"known namespaces F/P/C {K[rows].mean(0).round(3).tolist()}; seq len p50 {np.median(lens[rows]):.0f}")
    log(TAG, f"sample protein {b.prot.id.values[rows[0]]}: {seqs[rows[0]][:50]}... terms "
             f"{b.terms[Y[rows[0]].indices][:6].tolist()}")

    defs = build_defs(b.vocab)
    use_def = E.get("def_init", True)
    lat = latent_init(defs, b.aspect, E["n_latents"]) if use_def else None
    model = AdaptedEncoder(E["esm"], Y.shape[1], term_init=defs if use_def else None, latent_init=lat,
                           lora_r=E["lora_r"], n_latents=E["n_latents"], depth=E.get("depth", 2),
                           lora_from_layer=E.get("lora_from_layer", 0)).to(dev)
    tok = esm_tokenizer(E["esm"])
    n_esm = sum(p.numel() for p in model.enc.parameters())
    n_tr_esm = sum(p.numel() for p in model.enc.parameters() if p.requires_grad)
    log(TAG, f"params: ESM {n_esm / 1e6:.1f}M (trainable {n_tr_esm / 1e6:.3f}M, LoRA r={E['lora_r']}); resampler "
             f"{sum(p.numel() for p in model.resampler.parameters()) / 1e6:.2f}M; total trainable "
             f"{n_params(model) / 1e6:.2f}M; def-text init {use_def}")

    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
    groups = [{"params": other, "lr": E["lr_head"]}]
    if enc_params:
        groups.append({"params": enc_params, "lr": E["lr_lora"]})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    epochs = E["epochs"]
    steps_per_epoch = len(length_batches(lens[rows], E["max_tokens"], E["max_batch"], True, rng))
    total = epochs * steps_per_epoch
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log(TAG, f"optimiser: {epochs} epochs x {steps_per_epoch} steps; warmup {warm}")

    tv = b.rows("tv")
    from tracego.eval.score import Evaluator
    ev = Evaluator("tv")
    Qn = E.get("queue", 4096)
    q_r = torch.zeros(0, model.ret[-1].out_features, device=dev)
    q_Y = torch.zeros(0, Y.shape[1], device=dev)
    hist = []
    step = 0
    for ep in range(epochs):
        t0 = time.time()
        tot = {"cls": 0.0, "ret": 0.0, "n": 0}
        for bi in length_batches(lens[rows], E["max_tokens"], E["max_batch"], True, rng):
            br = rows[bi]
            ids, am = tokenize(tok, [seqs[j] for j in br], dev)
            Yb = dense_rows(Y, br, dev)
            known = known_terms(K, aspect_idx, br).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(ids, am)
                lg = model.term_logits(o["Z"]).float()
            l_cls = ia_bce(lg, Yb, known, w_term)
            r = o["r"].float()
            rk = torch.cat([r, q_r])
            Yk = torch.cat([Yb, q_Y])
            sm = torch.zeros(len(r), len(rk), dtype=torch.bool, device=dev)
            sm[:, :len(r)] = torch.eye(len(r), dtype=torch.bool, device=dev)
            l_ret = soft_contrastive(r, Yb, rk, Yk, w_term, model.log_tau.exp().clamp(0.02, 1), sm)
            loss = l_cls + E["ret_weight"] * l_ret
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            q_r = torch.cat([r.detach(), q_r])[:Qn]
            q_Y = torch.cat([Yb, q_Y])[:Qn]
            tot["cls"] += l_cls.item() * len(br)
            tot["ret"] += l_ret.item() * len(br)
            tot["n"] += len(br)
            step += 1
            if step % 200 == 0:
                log(TAG, f"ep {ep} step {step}/{total} cls {tot['cls'] / tot['n']:.4f} ret {tot['ret'] / tot['n']:.4f} "
                         f"tau {model.log_tau.exp().item():.3f} batch {len(br)} lr {sched.get_last_lr()[0]:.2e}")
        # per-epoch selection metric on TV v2: the resampler's classifier head (references not needed)
        enc = encode(model, tok, seqs[tv], lens[tv], dev, want=("r",), logits=True)
        cur = ev.curves(topk_csr(enc["p"]), b.prot.id.values[tv])
        s = ev.summary(cur)
        rec = {"epoch": ep, "cls_loss": tot["cls"] / tot["n"], "ret_loss": tot["ret"] / tot["n"],
               "tv_head": s, "seconds": time.time() - t0}
        hist.append(rec)
        log(TAG, f"epoch {ep} done in {time.time() - t0:.0f}s: loss cls {rec['cls_loss']:.4f} ret {rec['ret_loss']:.4f}; "
                 f"TV v2 head-only IA-Fmax MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} mean {s['mean']:.4f} "
                 f"test-like {s['mean_testlike']:.4f}")
    state = {k: v for k, v in model.state_dict().items() if ("lora_" in k or not k.startswith("enc."))}
    torch.save({"state": state, "cfg": cfg, "fold": a.fold}, run / "model.pt")
    save_json({"history": hist, "n_train": len(rows)}, run / "history.json")
    log(TAG, f"saved {len(state)} tensors -> {run / 'model.pt'}")
    if not a.no_cache:
        cache(model, tok, b, cfg, run, dev)


def cache_rows(b, cfg) -> np.ndarray:
    """Every row the decoder needs: training proteins (subset), TV queries, and at full scale val, test, fresh24."""
    rows = [train_rows(b, cfg["data"]["subset"], cfg.get("seed", 0)), b.rows("tv")]
    if not cfg["data"]["subset"]:
        rows += [b.rows("val"), b.rows("test"), b.rows("fr24p")]
    return np.unique(np.concatenate(rows))


def cache(model, tok, b, cfg, run: Path, dev):
    t0 = time.time()
    rows = cache_rows(b, cfg)
    seqs, lens = b.prot.seq.values, b.prot.length.values
    log(TAG, f"caching Z / r / h for {len(rows):,} proteins")
    res = encode(model, tok, seqs[rows], lens[rows], dev)
    np.save(run / "rows.npy", rows)
    for k, v in res.items():
        np.save(run / f"{k}.npy", v)
        log(TAG, f"  {k}: {v.shape} {v.dtype} ({v.nbytes / 1e9:.2f} GB); sample {v.reshape(len(v), -1)[0, :4]}")
    log(TAG, f"cache done in {time.time() - t0:.0f}s -> {run}")


if __name__ == "__main__":
    main()
