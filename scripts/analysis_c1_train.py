"""Phase C1: retrain resampler + term head on a frozen fold-0 encoder; fix latent collapse with competitive
slot-attention and a diversity penalty. Residue states are computed on the fly (no full-set residue cache).

Trains two heads on the same 20k subset and GO-definition init for a matched comparison:
  competitive   CompetitiveResampler (softmax over latents) + diversity loss
  control       stock Resampler (same data / init / epochs; still collapses)

Also scores the original fold-0 adapter (collapsed) as a third baseline. Selection metric: TV v2 term-head
IA-Fmax. Checkpoints and history under runs/analysis/c1/.

  CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c1_train.py
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracego.analysis.common import OUT, disk_free_gb, load_fold_encoder, log, out_dir
from tracego.analysis.slot_resampler import CompetitiveResampler, diversity_loss, latent_corr
from tracego.data.bundle import Bundle
from tracego.data.godefs import build as build_defs, latent_init
from tracego.models.adapter import ia_bce
from tracego.models.encoder import esm_tokenizer, length_batches, tokenize
from tracego.models.resampler import Resampler
from tracego.train.common import dense_rows, known_mask, n_params, save_json, term_weights, topk_csr, train_rows

TAG = "c1train"
DIV_W = 0.25
EPOCHS = 3
SUBSET = 20000
MAX_TOKENS = 48000
MAX_BATCH = 64
LR = 1e-3
SEED = 0


class TraceHead(nn.Module):
    """Resampler + term head sharing the frozen encoder's residue states. Matches AdaptedEncoder's term_logits."""

    def __init__(self, d_in: int, n_terms: int, resampler: nn.Module, term_init, d: int = 512):
        super().__init__()
        self.resampler = resampler
        self.cls_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        te = torch.as_tensor(term_init, dtype=torch.float32)
        te = te / te.norm(dim=1, keepdim=True)
        self.term_emb = nn.Parameter(te)
        self.term_bias = nn.Parameter(torch.full((n_terms,), -4.0))
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, H, mask, need_weights=False):
        Z, ws = self.resampler(H, mask, need_weights)
        return Z, ws

    def term_logits(self, Z):
        q = F.normalize(self.cls_proj(Z), dim=-1)
        e = F.normalize(self.term_emb, dim=-1)
        return self.scale * (q @ e.T).amax(1) + self.term_bias


def freeze_encoder(enc: nn.Module):
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def encode_residues(enc, tok, seqs, lens, dev, max_tokens=MAX_TOKENS):
    """Yield (batch_indices_into_seqs, H [B,L,d], mask [B,L]) on the fly. Caller must not cache all H."""
    for bi in length_batches(lens, max_tokens, MAX_BATCH, False):
        ids, am = tokenize(tok, [seqs[j] for j in bi], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = enc(ids, am)
            Hs = out.last_hidden_state
            m = enc.residue_mask(ids, am)
        yield bi, Hs, m


@torch.no_grad()
def tv_head_fmax(head, enc, tok, b, tv_rows, ev, dev):
    from tracego.analysis.slot_resampler import CompetitiveResampler  # noqa: F401
    seqs, lens = b.prot.seq.values, b.prot.length.values
    n, V = len(tv_rows), len(b.terms)
    P = np.zeros((n, V), np.float32)
    pos = {int(r): i for i, r in enumerate(tv_rows)}
    for bi, Hs, m in encode_residues(enc, tok, seqs[tv_rows], lens[tv_rows], dev):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Z, _ = head(Hs, m)
            lg = head.term_logits(Z).float()
        P[bi] = torch.sigmoid(lg).cpu().numpy()
    cur = ev.curves(topk_csr(P), b.prot.id.values[tv_rows])
    return ev.summary(cur)


@torch.no_grad()
def collapse_diag(head, enc, tok, seqs, lens, rows, dev, n=64):
    """Mean off-diagonal attention-map correlation on a small sample (collapse diagnostic)."""
    pick = rows[:n]
    cors = []
    for bi, Hs, m in encode_residues(enc, tok, seqs[pick], lens[pick], dev, max_tokens=16000):
        Z, ws = head(Hs, m, need_weights=True)
        cors.append(latent_corr(ws[-1], m))
    return float(np.mean(cors)) if cors else float("nan")


def train_one(name: str, head: TraceHead, enc, tok, b, rows, Y, K, aspect_idx, w_term, ev, tv, od: Path, dev,
              competitive: bool):
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    seqs, lens = b.prot.seq.values, b.prot.length.values
    rng = np.random.default_rng(SEED)
    steps_ep = len(length_batches(lens[rows], MAX_TOKENS, MAX_BATCH, True, rng))
    total = EPOCHS * steps_ep
    warm = min(200, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    log(TAG, f"{name}: {n_params(head)/1e6:.2f}M trainable; {len(rows):,} proteins; {EPOCHS} epochs x {steps_ep} "
             f"steps; competitive={competitive} div_w={DIV_W if competitive else 0}")
    hist, step, best = [], 0, -1.0
    for ep in range(EPOCHS):
        t0 = time.time()
        head.train()
        freeze_encoder(enc)
        tot = {"cls": 0.0, "div": 0.0, "n": 0}
        for bi in length_batches(lens[rows], MAX_TOKENS, MAX_BATCH, True, rng):
            br = rows[bi]
            ids, am = tokenize(tok, [seqs[j] for j in br], dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = enc(ids, am)
                Hs = out.last_hidden_state
                m = enc.residue_mask(ids, am)
            Yb = dense_rows(Y, br, dev)
            known = torch.as_tensor(K[br][:, aspect_idx], dtype=torch.float32, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Z, ws = head(Hs, m, need_weights=competitive)
                lg = head.term_logits(Z).float()
            l_cls = ia_bce(lg, Yb, known, w_term)
            l_div = diversity_loss(ws[-1].float(), m) if competitive else torch.zeros((), device=dev)
            loss = l_cls + (DIV_W * l_div if competitive else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if not torch.isfinite(loss):
                log(TAG, f"{name} non-finite loss at step {step}; skipping batch (cls={l_cls.item()} div={float(l_div.detach())})")
                opt.zero_grad(set_to_none=True)
                continue
            tot["cls"] += l_cls.item() * len(br)
            tot["div"] += float(l_div.detach()) * len(br)
            tot["n"] += len(br)
            step += 1
            if step % 100 == 0:
                log(TAG, f"{name} ep {ep} step {step}/{total} cls {tot['cls']/tot['n']:.4f} "
                         f"div {tot['div']/tot['n']:.4f} lr {sched.get_last_lr()[0]:.2e}")
        head.eval()
        s = tv_head_fmax(head, enc, tok, b, tv, ev, dev)
        corr = collapse_diag(head, enc, tok, seqs, lens, rows, dev)
        rec = {"epoch": ep, "cls": tot["cls"] / tot["n"], "div": tot["div"] / tot["n"], "tv_head": s,
               "att_corr": corr, "seconds": time.time() - t0}
        hist.append(rec)
        log(TAG, f"{name} epoch {ep} done in {rec['seconds']:.0f}s: cls {rec['cls']:.4f} div {rec['div']:.4f}; "
                 f"att_corr {corr:.3f}; TV head MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} "
                 f"mean {s['mean']:.4f} test-like {s['mean_testlike']:.4f}")
        ckpt = {"state": head.state_dict(), "name": name, "epoch": ep, "tv": s, "att_corr": corr}
        torch.save(ckpt, od / f"{name}_ep{ep}.pt")
        if s["mean_testlike"] > best:
            best = s["mean_testlike"]
            torch.save(ckpt, od / f"{name}_best.pt")
            log(TAG, f"{name}: new best test-like {best:.4f} -> {od / (name + '_best.pt')}")
    save_json({"history": hist, "n_train": len(rows), "competitive": competitive}, od / f"{name}_history.json")
    return hist


def score_collapsed(adapter, tok, b, tv, ev, od, dev):
    """TV term-head Fmax + collapse diagnostic of the original fold-0 adapter (no retrain)."""
    from tracego.models.encoder import length_batches, tokenize
    seqs, lens = b.prot.seq.values, b.prot.length.values
    n, V = len(tv), len(b.terms)
    P = np.zeros((n, V), np.float32)
    cors = []
    with torch.no_grad():
        for bi in length_batches(lens[tv], MAX_TOKENS, MAX_BATCH, False):
            br = tv[bi]
            ids, am = tokenize(tok, [seqs[j] for j in br], dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = adapter(ids, am, need_weights=True)
                lg = adapter.term_logits(o["Z"]).float()
            P[bi] = torch.sigmoid(lg).cpu().numpy()
            cors.append(latent_corr(o["att"][-1], o["mask"]))
    cur = ev.curves(topk_csr(P), b.prot.id.values[tv])
    s = ev.summary(cur)
    corr = float(np.mean(cors))
    log(TAG, f"collapsed fold-0: att_corr {corr:.3f}; TV head MF {s['MF']:.4f} BP {s['BP']:.4f} CC {s['CC']:.4f} "
             f"mean {s['mean']:.4f} test-like {s['mean_testlike']:.4f}")
    save_json({"tv_head": s, "att_corr": corr}, od / "collapsed_tv.json")
    return s, corr


def main():
    od = out_dir("c1")
    log(TAG, f"disk free {disk_free_gb():.1f} GB; out {od}")
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    b = Bundle()
    rows = train_rows(b, SUBSET, SEED)
    tv = b.rows("tv")
    Y = b.Y_t0
    aspect_idx = np.array(["FPC".index(x) for x in b.aspect])
    K = known_mask(Y, b.aspect)
    w_term = torch.as_tensor(term_weights(b.ia, True), device=dev)
    log(TAG, f"20k subset {len(rows):,}; TV {len(tv):,}; vocab {Y.shape[1]:,}; disk {disk_free_gb():.1f} GB")

    from tracego.eval.score import Evaluator
    ev = Evaluator("tv")

    adapter = load_fold_encoder(0, b, dev, frozen=False)
    freeze_encoder(adapter.enc)
    tok = esm_tokenizer("facebook/esm2_t12_35M_UR50D")
    score_collapsed(adapter, tok, b, tv, ev, od, dev)

    defs = build_defs(b.vocab)
    lat = latent_init(defs, b.aspect, 16)
    d_in = adapter.enc.hidden

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="both", choices=("both", "control", "competitive"))
    args, _ = ap.parse_known_args()

    if args.only in ("both", "control") and not (od / "control_best.pt").exists():
        ctrl = TraceHead(d_in, Y.shape[1], Resampler(d_in, 512, 16, 2, init=lat), defs).to(dev)
        train_one("control", ctrl, adapter.enc, tok, b, rows, Y, K, aspect_idx, w_term, ev, tv, od, dev, False)
        del ctrl
        torch.cuda.empty_cache()
    elif (od / "control_best.pt").exists():
        log(TAG, f"control_best.pt exists; skip control retrain")

    if args.only in ("both", "competitive"):
        comp = TraceHead(d_in, Y.shape[1], CompetitiveResampler(d_in, 512, 16, 2, init=lat), defs).to(dev)
        train_one("competitive", comp, adapter.enc, tok, b, rows, Y, K, aspect_idx, w_term, ev, tv, od, dev, True)
    log(TAG, f"done; disk free {disk_free_gb():.1f} GB; artefacts in {od}")


if __name__ == "__main__":
    main()
