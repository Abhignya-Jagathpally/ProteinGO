"""Ablation 'classifier head only': the same cross-fitted inputs as the decoder (soft-prompt latents Z and the
neighbour vote) feed an MLP multi-label head with the same copy bias, trained with IA-weighted BCE. No instruction
text, no decoding.

  python -m tracego.train.train_classifier --config configs/full.yaml --tag clf [--benches tv,test,fr24p]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator, save_pred
from tracego.train import train_decoder as td
from tracego.train.common import known_mask, load_config, log, n_params, save_json, term_weights
from tracego.eval.predict import oov_vote

TAG = "clf"


class Classifier(nn.Module):
    def __init__(self, d_in, V, use_vote=True):
        super().__init__()
        self.att = nn.Linear(d_in, 1)
        self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(1024, V))
        self.use_vote = use_vote
        # same init as the decoder's membership head: starts exactly at the vote for voted terms, ~0 elsewhere
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -6.0)
        self.a = nn.Parameter(torch.ones(V))
        self.b = nn.Parameter(torch.full((V,), 6.0))

    def forward(self, Z, v):
        w = self.att(Z).softmax(1)
        x = (w * Z).sum(1)
        lg = self.mlp(x)
        if self.use_vote:
            lg = lg + self.a * torch.logit(v.clamp(1e-3, 1 - 1e-3)) * (v > 0) + self.b * (v > 0)
        return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="clf")
    ap.add_argument("--benches", default="tv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    E, R = cfg["encoder"], cfg["retrieval"]
    run = paths.RUNS / cfg["name"] / f"dec_{a.tag}"
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    b = Bundle()
    spaces = [td.Space(paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{f}",
                       paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{f}" / f"evidence_{R['space']}.npz") for f in (0, 1)]
    Y = b.Y_t0
    V = Y.shape[1]
    K = known_mask(Y, b.aspect)
    aidx = np.array(["FPC".index(x) for x in b.aspect])
    w = torch.as_tensor(term_weights(b.ia, cfg["decoder"]["ia_weight"]), device=dev)
    model = Classifier(spaces[0].Z.shape[-1], V).to(dev)
    log(TAG, f"classifier params {n_params(model) / 1e6:.2f}M; inputs Z {spaces[0].Z.shape[1:]} + vote over {V:,} terms")
    data = []
    for f in (0, 1):
        q = spaces[f].ev["train_q"]
        data.append((f, q, td.vote_matrix(spaces[f], "train")))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ev = Evaluator("tv")
    rng = np.random.default_rng(0)

    def batch_inputs(f, q, vt, idx):
        Z = torch.as_tensor(np.stack([spaces[f].Z[spaces[f].zpos[r]] for r in q[idx]]).astype(np.float32), device=dev)
        v = torch.as_tensor(vt[idx].toarray(), device=dev)
        return Z, v

    def predict(bench):
        out = None
        for f in (0, 1):
            q = spaces[f].ev[f"{bench}_q"]
            vt = td.vote_matrix(spaces[f], bench)
            P = np.zeros((len(q), V), np.float32)
            model.eval()
            with torch.no_grad():
                for s in range(0, len(q), 1024):
                    idx = np.arange(s, min(len(q), s + 1024))
                    Z, v = batch_inputs(f, q, vt, idx)
                    P[idx] = torch.sigmoid(model(Z, v)).cpu().numpy()
            model.train()
            M = td.to_vocab_csr({0: P[:, td_cols[0]], 1: P[:, td_cols[1]], 2: P[:, td_cols[2]]},
                                [torch.as_tensor(c) for c in td_cols], len(q), V)
            out = M if out is None else out + M
        return out / 2, b.ids[spaces[0].ev[f"{bench}_q"]]

    td_cols = [np.where(b.aspect == x)[0] for x in "FPC"]
    hist, best = [], (-1.0, -1, None)
    for ep in range(a.epochs):
        t0 = time.time()
        tot, n = 0.0, 0
        for f, q, vt in data:
            perm = rng.permutation(len(q))
            for s in range(0, len(q), 512):
                idx = perm[s:s + 512]
                Z, v = batch_inputs(f, q, vt, idx)
                yb = torch.as_tensor(Y[q[idx]].toarray(), device=dev)
                kb = torch.as_tensor(K[q[idx]][:, aidx], dtype=torch.float32, device=dev)
                l = F.binary_cross_entropy_with_logits(model(Z, v), yb, reduction="none")
                loss = (l * w * kb).sum() / kb.sum()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx); n += len(idx)
        P, ids = predict("tv")
        s = ev.summary(ev.curves(P, ids))
        hist.append({"epoch": ep, "loss": tot / n, "tv": s})
        if s["mean_testlike"] > best[0]:
            best = (s["mean_testlike"], ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
        log(TAG, f"epoch {ep} loss {tot / n:.4f} ({time.time() - t0:.0f}s); TV v2 vocab terms mean {s['mean']:.4f} "
                 f"test-like {s['mean_testlike']:.4f}")
    model.load_state_dict(best[2])
    log(TAG, f"kept epoch {best[1]} (TV v2 test-like {best[0]:.4f})")
    res = {}
    for bench in a.benches.split(","):
        P, ids = predict(bench)
        evb = Evaluator(bench)
        O = None
        for f in (0, 1):
            Of = oov_vote(b, evb, spaces[f].ev[f"{bench}_I"], spaces[f].ev[f"{bench}_S"],
                          "t0" if bench == "tv" else "full", R)
            O = Of if O is None else O + Of
        M = (evb.to_onto(P) + O / 2).tocsr()
        s = evb.summary(evb.curves(M, ids, onto=True))
        res[bench] = s
        save_pred(run / f"pred_{bench}_clf.npz", M, ids, evb.o.terms)
        log(TAG, f"{bench} (with OOV copy): {s}")
    torch.save(model.state_dict(), run / "model.pt")
    save_json({"history": hist, "best_epoch": best[1], "final": res}, run / "history.json")


if __name__ == "__main__":
    main()
