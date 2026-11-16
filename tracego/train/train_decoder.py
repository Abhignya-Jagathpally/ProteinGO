"""Stage B: instruction fine-tuning of the retrieval-conditioned GO decoder on cached encoder outputs.

  python -m tracego.train.train_decoder --config configs/dev.yaml [--tag main] [key=value ...]

Training queries are the v2 training proteins; each one is seen through the encoder fold that did not train on it
(cross-fitting), with neighbours from that fold's retrieval space. Selection metric per epoch: TV v2 IA-Fmax.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.data.instructions import Collator, InstructionSet, leaves_by_ns
from tracego.eval.score import Evaluator
from tracego.models.decoder import Hierarchy, TraceDecoder, set_targets
from tracego.train.common import load_config, log, n_params, save_json, term_weights, train_rows

TAG = "decB"


def term_init_ids(tok, vocab, n_tok: int = 48):
    out = []
    for n, d in zip(vocab.name, vocab.definition):
        a = tok(n, add_special_tokens=False)["input_ids"]
        bb = tok(d, add_special_tokens=False)["input_ids"]
        out.append((a + a + bb)[:n_tok] or [tok.unk_token_id])
    return out


class Space:
    """One cross-fitting fold: cached latents and the neighbour lists of every query set."""

    def __init__(self, run_dir, evid_file):
        self.rows = np.load(run_dir / "rows.npy")
        self.zpos = {int(r): i for i, r in enumerate(self.rows)}
        self.Z = np.load(run_dir / "Z.npy", mmap_mode="r")
        self.ev = dict(np.load(evid_file, allow_pickle=False))


def load_aux_toks(b, q_rows, tok_ids, encode, benches=("train", "tv")):
    """Optional Phase B aux evidence tokens from runs/aux/{signal,loc}_*.npz (decoder.aux_evidence)."""
    from tracego import paths
    out = [[] for _ in q_rows]
    id_to_i = {a: i for i, a in enumerate(b.ids[q_rows])}
    any_hit = False
    for kind in ("signal", "loc"):
        for bench in benches:
            p = paths.RUNS / "aux" / f"{kind}_{bench}.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            ids = z["ids"].astype(str)
            # feature summary: first few floats, encoded as a short text token sequence
            feats = z[[k for k in z.files if k != "ids"][0]]
            for j, a in enumerate(ids):
                if a not in id_to_i:
                    continue
                any_hit = True
                vec = np.asarray(feats[j]).ravel()[:8]
                txt = f" {kind}:" + ",".join(f"{x:.2f}" for x in vec)
                out[id_to_i[a]] += encode(txt)
    return out if any_hit else None


def build_sets(cfg, b, spaces, tok_ids, ref_leaves, prefix_cache, names):
    R = cfg["retrieval"]
    D = cfg["decoder"]
    out = {}
    for f, sp_ in enumerate(spaces):
        for nm in names:
            if f"{nm}_q" not in sp_.ev:
                continue
            q, I, S = sp_.ev[f"{nm}_q"], sp_.ev[f"{nm}_I"], sp_.ev[f"{nm}_S"]
            lv = ref_leaves["t0" if nm in ("train", "tv", "tvr") else "full"]
            hom = None
            if D.get("homology_token", False):
                if nm == "train" and "train_hom_bin" in sp_.ev:
                    hom = sp_.ev["train_hom_bin"]
                elif nm in ("tv", "test", "val"):
                    hom = np.full(len(q), -1, np.int8)  # plain high-id benches = unmasked / full-homology token
                elif nm == "tvr":
                    hom = sp_.ev["tvr_bin"] if "tvr_bin" in sp_.ev else None
                    if hom is None:
                        from tracego.eval.remote import RemoteMask
                        hom = RemoteMask().bin
                elif nm == "fr24p":
                    from tracego.eval.remote import bin_index
                    from tracego.eval.score import identity as id_of
                    hom = bin_index(id_of("fr24p").reindex(b.ids[q]).fillna(0).to_numpy()).astype(np.int8)
            aux = None
            if D.get("aux_evidence", False):
                aux = load_aux_toks(b, q, tok_ids, tok_ids["encode"],
                                   benches=("train",) if nm == "train" else ("tv", "tvr"))
            out[(nm, f)] = InstructionSet(tok_ids, b, q, I, S, lv, None, sp_.ev["sim_edges"], R["k"],
                                          D["evidence"], prefix_cache, hom_bin=hom, aux_toks=aux)
    return out


def vote_matrix(sp_, nm):
    z = sp_.ev
    return sp.csr_matrix((z[f"{nm}_vd"], z[f"{nm}_vi"], z[f"{nm}_vp"]), shape=tuple(z[f"{nm}_vs"]))


def member_loss(model, h, Yb, S_at, vote, ns, w_ns, anchor_l2=0.0):
    lg = model.member_logits(h, ns, None if vote is None else vote[:, None, :])
    l = F.binary_cross_entropy_with_logits(lg.float(), Yb[:, None, :].float().expand_as(lg), reduction="none")
    keep = (~S_at).float() * w_ns
    loss = (l * keep).sum() / keep.sum().clamp(min=1) * 10.0
    if anchor_l2 > 0:
        # vote-anchored membership: the protein-specific residual on top of the copy prior pays an L2 price, so the
        # decoder departs from the neighbour vote only where the training signal consistently supports it
        cols = model.ns_cols[ns].to(h.device)
        res = model.mem_proj(h) @ model.mem_emb[cols].T / (h.shape[-1] ** 0.5)
        loss = loss + anchor_l2 * res.float().pow(2).mean()
    return loss


def next_loss(lg, eos, legal, avail, eos_t, valid, constrained):
    lg = lg.float()
    if constrained:
        lg = lg.masked_fill(~legal, float("-inf"))
    else:
        lg = lg.masked_fill(~legal, float("-inf"))  # already-emitted terms are excluded in both settings
    full = torch.cat([lg, eos.float()[..., None]], -1)
    lp = full.log_softmax(-1)
    na = avail.sum(-1).clamp(min=1)
    l_terms = -(lp[..., :-1].masked_fill(~avail, 0)).sum(-1) / na
    l = torch.where(eos_t, -lp[..., -1], l_terms)
    return (l * valid).sum() / valid.sum().clamp(min=1)


def forward_batch(model, bt, ns, H, cfg, w_ns, dev):
    D = cfg["decoder"]
    Z = bt["Z"] if D.get("soft_prompt", True) else None
    vote = bt.get("vote") if D["copy_bias"] else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc_h, enc_m = model.encode(Z, bt["src"], bt["src_mask"])
        tgt = bt["tgt"]
        cols = model.ns_cols[ns].to(dev)
        dec_in = torch.cat([torch.zeros_like(tgt[:, :1]),
                            torch.where(tgt >= 0, model.go0 + cols[tgt.clamp(min=0)], torch.zeros_like(tgt))], 1)
        out = model.decode(dec_in, enc_h, enc_m)
        h = out.last_hidden_state
    S, legal, avail, eos_t, valid = set_targets(tgt, bt["Y"], H if D["constrained"] else None, bt["truncated"])
    lg, eos = model.next_logits(h.float(), ns, None if vote is None else vote[:, None, :])
    l_next = next_loss(lg, eos, legal, avail, eos_t, valid, D["constrained"])
    # membership at position 0 plus a few random valid prefix positions
    P = D["member_positions"]
    B, T1 = valid.shape
    lens = valid.sum(1)
    pos = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev),
                     (torch.rand(B, P - 1, device=dev) * lens[:, None]).long().clamp(max=T1 - 1)], 1)
    hs = h.float().gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
    Sat = S.gather(1, pos[..., None].expand(-1, -1, S.shape[-1]))
    l_mem = member_loss(model, hs, bt["Y"], Sat, vote, ns, w_ns, D.get("anchor_l2", 0.0))
    return l_next, l_mem


@torch.no_grad()
def predict(model, iset, ns, vote, coll, H, cfg, dev, readouts=("step0", "ccms", "mix"), max_steps=None,
            batch=256, trace=False):
    """Constrained greedy decoding with commit-conditioned membership scores. Returns {readout: [n, V_ns]}."""
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
    D = cfg["decoder"]
    model.eval()
    n = len(iset.q_rows)
    V = len(model.ns_cols[ns])
    max_steps = max_steps or D.get("max_decode", {0: 40, 1: 128, 2: 48}[ns])
    res = {r: np.zeros((n, V), np.float32) for r in readouts}
    gen = []
    cols = model.ns_cols[ns].to(dev)
    for s in range(0, n, batch):
        idx = list(range(s, min(n, s + batch)))
        bt = coll([(iset, i) for i in idx], ns, None, None if vote is None else vote[idx], dev)
        vb = bt.get("vote") if D["copy_bias"] else None
        Z = bt["Z"] if D.get("soft_prompt", True) else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc_h, enc_m = model.encode(Z, bt["src"], bt["src_mask"])
        B = len(idx)
        Sm = torch.zeros(B, V, dtype=torch.bool, device=dev)
        cnt = torch.zeros(B, V, device=dev)
        commit = torch.zeros(B, V, device=dev)
        final = torch.zeros(B, V, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        cur = torch.zeros(B, 1, dtype=torch.long, device=dev)
        past = EncoderDecoderCache(DynamicCache(), DynamicCache())
        mu0 = None
        steps = []
        for t in range(max_steps + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.decode(cur, enc_h, enc_m, past=past)
            past = o.past_key_values
            h = o.last_hidden_state[:, -1].float()
            mu = torch.sigmoid(model.member_logits(h, ns, vb).float())
            if mu0 is None:
                mu0 = mu.clone()
            lg, eos = model.next_logits(h, ns, vb)
            legal = H.frontier(Sm, cnt) if D["constrained"] else ~Sm
            lg = lg.float().masked_fill(~legal, float("-inf"))
            full = torch.cat([lg, eos.float()[:, None]], -1)
            if t == max_steps:
                full[:, :-1] = float("-inf")
            ch = full.argmax(-1)
            is_eos = (ch == V) & ~done
            final[is_eos] = mu[is_eos]
            go = (~done) & (ch < V)
            bi = torch.where(go)[0]
            tt = ch[bi]
            commit[bi, tt] = mu[bi, tt]
            Sm[bi, tt] = True
            if D["constrained"]:
                cnt[bi] += H.desc[tt].float()
            steps.append(torch.where(go, ch, -1))
            done = done | is_eos
            nxt = torch.where(go, model.go0 + cols[ch.clamp(max=V - 1)], torch.zeros_like(ch))
            cur = nxt[:, None]
            if done.all():
                break
        ccms = torch.where(Sm, commit, final)
        if "step0" in res:
            res["step0"][idx] = mu0.cpu().numpy()
        if "ccms" in res:
            res["ccms"][idx] = ccms.cpu().numpy()
        if "mix" in res:
            res["mix"][idx] = (0.5 * (ccms + mu0)).cpu().numpy()
        if "gen" in res:
            res["gen"][idx] = Sm.float().cpu().numpy()
        if trace:
            gen.extend(torch.stack(steps, 1).cpu().numpy().tolist())
    model.train()
    return (res, gen) if trace else res


def to_vocab_csr(dense_by_ns: dict, ns_cols, n, V, k=400, min_score=0.005):
    rows, cols, vals = [], [], []
    for ns, M in dense_by_ns.items():
        c = ns_cols[ns].numpy()
        kk = min(k, M.shape[1])
        idx = np.argpartition(-M, kk - 1, axis=1)[:, :kk]
        v = np.take_along_axis(M, idx, 1)
        m = v >= min_score
        r = np.repeat(np.arange(n), kk).reshape(n, kk)
        rows.append(r[m]); cols.append(c[idx[m]]); vals.append(v[m])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, V))


def evaluate_bench(model, sets, spaces, coll, Hs, cfg, b, ev, dev, readouts, bench="tv", max_steps=None, nfold=2):
    """IA-Fmax on a named evidence set (tv or tvr), averaging the two folds' score matrices (vocab terms)."""
    out = {}
    mats = {r: None for r in readouts}
    for f in range(nfold):
        iset = sets[(bench, f)]
        vote = vote_matrix(spaces[f], bench)
        dense = {r: {} for r in readouts}
        for ns in range(3):
            res = predict(model, iset, ns, vote, coll[f], Hs[ns], cfg, dev, readouts, max_steps)
            for r in readouts:
                dense[r][ns] = res[r]
        for r in readouts:
            M = to_vocab_csr(dense[r], model.ns_cols, len(iset.q_rows), model.n_terms)
            mats[r] = M if mats[r] is None else mats[r] + M
    ids = b.ids[sets[(bench, 0)].q_rows]
    curs = {}
    for r in readouts:
        P = mats[r] / nfold
        curs[r] = ev.curves(P, ids)
        out[r] = ev.summary(curs[r])
    return out, mats, curs


def evaluate_tv(model, sets, spaces, coll, Hs, cfg, b, ev, dev, readouts, max_steps=None, nfold=2):
    return evaluate_bench(model, sets, spaces, coll, Hs, cfg, b, ev, dev, readouts, "tv", max_steps, nfold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    D, E, R = cfg["decoder"], cfg["encoder"], cfg["retrieval"]
    run = paths.RUNS / cfg["name"] / f"dec_{a.tag}"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))
    dev = torch.device("cuda")
    t_all = time.time()
    b = Bundle()
    log(TAG, f"run {run.name}: decoder {D['t5']}; constrained {D['constrained']}; evidence {D['evidence']}; copy bias "
             f"{D['copy_bias']}; def init {D['def_init']}; IA weight {D['ia_weight']}; soft prompt "
             f"{D.get('soft_prompt', True)}")

    enc_dirs = [paths.RUNS / cfg["name"] / f"enc_{E['tag']}_f{f}" for f in (0, 1)]
    evid_suffix = R.get("evid_suffix", "")
    spaces = [Space(d, d / f"evidence_{R['space']}{evid_suffix}.npz") for d in enc_dirs]
    log(TAG, f"spaces: {[str(d.name) for d in enc_dirs]}; evidence evidence_{R['space']}{evid_suffix}.npz; "
             f"latents {spaces[0].Z.shape}; evidence keys "
             f"{sorted(k for k in spaces[0].ev if k.endswith('_q'))}")

    model = TraceDecoder(D["t5"], b.terms, b.aspect, d_latent=spaces[0].Z.shape[-1],
                         copy_bias=D["copy_bias"], homology=bool(D.get("homology_token", False))).to(dev)
    if D["def_init"]:
        init = term_init_ids(model.tok, b.vocab)
        with torch.no_grad():
            emb = model.t5.get_input_embeddings().weight
            head = model.t5.get_output_embeddings().weight
            for j, ids in enumerate(init):
                ii = torch.as_tensor(ids, device=dev)
                emb[model.go0 + j] = emb[ii].mean(0)
                head[model.go0 + j] = head[ii].mean(0)
            model.mem_emb.copy_(head[model.go0:model.go0 + model.n_terms])
        log(TAG, f"definition-text init of {len(init):,} GO tokens; e.g. {b.terms[0]} <- "
                 f"'{model.tok.decode(init[0])[:80]}'")
    if D.get("anchor", False) and model.copy is not None:
        # membership copy weight fixed at its init (a = 1, offset cancelling the bias): sigmoid(logit(vote) + residual)
        keep = torch.ones_like(model.copy.a)
        keep[:, 1] = 0
        model.copy.a.register_hook(lambda g: g * keep)
        model.copy.b.register_hook(lambda g: g * keep)
        log(TAG, f"membership anchored to the vote (copy weight frozen); residual L2 {D.get('anchor_l2', 0.0)}")
    log(TAG, f"tokenizer {len(model.tok):,} tokens; trainable params {n_params(model) / 1e6:.1f}M "
             f"(T5 {sum(p.numel() for p in model.t5.parameters()) / 1e6:.1f}M)")

    tok_ids = {"encode": lambda s: model.tok(s, add_special_tokens=False)["input_ids"], "nb": model.nb_id,
               "sim0": model.sim0, "tax0": model.tax0, "hom0": model.hom0, "go0": model.go0,
               "eos": model.tok.eos_token_id}
    t = time.time()
    ref_leaves = {}
    for which in ("t0", "full"):
        Yl = b.labels(which)
        rows = np.where(np.diff(Yl.indptr) > 0)[0]
        lv = leaves_by_ns(Yl[rows], b.anc, b.aspect, b.ia, D["leaves_per_nb"])
        ref_leaves[which] = {int(r): l for r, l in zip(rows, lv)}
    log(TAG, f"reference leaves built in {time.time() - t:.0f}s")
    names = ["train", "tv", "tvr"] + [n for n in ("val", "test", "fr24p") if f"{n}_q" in spaces[0].ev]
    names = [n for n in names if f"{n}_q" in spaces[0].ev]
    prefix_cache = {}
    t = time.time()
    sets = build_sets(cfg, b, spaces, tok_ids, ref_leaves, prefix_cache, names)
    s0 = sets[("train", 0)]
    ex = s0.src[0][0]
    log(TAG, f"instruction sets built in {time.time() - t:.0f}s: "
             f"{ {k: len(v.q_rows) for k, v in sets.items()} }")
    log(TAG, f"sample source (MF, protein {b.ids[s0.q_rows[0]]}, {len(ex)} tokens + {spaces[0].Z.shape[1]} soft): "
             f"{model.tok.decode(ex)[:400]}")
    src_len = np.array([len(x) for x in s0.src[1]])
    log(TAG, f"BP source length p50 {np.median(src_len):.0f} p99 {np.percentile(src_len, 99):.0f} max {src_len.max()}")

    # hierarchy per namespace
    Hs, parents = [], []
    for ns in range(3):
        cols = model.ns_cols[ns].numpy()
        H = Hierarchy(b.anc, cols, dev)
        Hs.append(H)
        c, p = H.parents_np
        d = {}
        for ci, pi in zip(c, p):
            d.setdefault(int(ci), []).append(int(pi))
        parents.append(d)
        log(TAG, f"namespace {'FPC'[ns]}: {H.V:,} terms; roots {(H.n_anc == 0).sum().item()}; "
                 f"mean in-vocab ancestors {H.n_anc.mean().item():.1f}")
    coll = [Collator(spaces[f].Z, spaces[f].zpos, model.ns_cols, parents, D["max_src"], D["max_tgt"],
                     D["constrained"], seed=cfg.get("seed", 0) + f) for f in (0, 1)]
    w_term = torch.as_tensor(term_weights(b.ia, D["ia_weight"]), device=dev)
    w_ns = [w_term[model.ns_cols[ns].to(dev)] for ns in range(3)]

    # training examples: (fold space, index in set, namespace) for every annotated namespace
    Yt = b.Y_t0
    ex = []
    for f in (0, 1):
        q = sets[("train", f)].q_rows
        for ns in range(3):
            cols = model.ns_cols[ns].numpy()
            has = np.asarray(Yt[q][:, cols].sum(1)).ravel() > 0
            ex += [(f, i, ns) for i in np.where(has)[0]]
    ex = np.array(ex)
    p_ex = None
    if D.get("shift_weight", False):
        # covariate-shift correction: training queries are drawn so their top-1 evidence-score distribution
        # matches the TV v2 queries' (the training set is redundancy-reduced, so its queries see far weaker
        # neighbours than temporally new proteins do). Uses TV retrieval scores only, never TV labels.
        top_tr = np.array([spaces[f].ev["train_S"][i, 0] for f, i, _ in ex])
        top_tv = np.concatenate([spaces[f].ev["tv_S"][:, 0] for f in (0, 1)])
        edges = np.unique(np.quantile(top_tv, np.linspace(0, 1, 21)[1:-1]))
        btr, btv = np.digitize(top_tr, edges), np.digitize(top_tv, edges)
        ptr = np.bincount(btr, minlength=len(edges) + 1) / len(btr)
        ptv = np.bincount(btv, minlength=len(edges) + 1) / len(btv)
        w = np.clip(ptv[btr] / np.maximum(ptr[btr], 1e-9), 0.05, 20.0)
        p_ex = w / w.sum()
        log(TAG, f"shift weights: train top-1 p50 {np.median(top_tr):.3f} vs TV {np.median(top_tv):.3f}; weight "
                 f"p10/p50/p90 {np.percentile(w, [10, 50, 90]).round(2).tolist()}; effective sample size "
                 f"{1 / (p_ex ** 2).sum():,.0f} of {len(ex):,}")
    votes = {(nm, f): vote_matrix(spaces[f], nm) for (nm, f) in sets}
    log(TAG, f"training examples {len(ex):,} (F {int((ex[:, 2] == 0).sum()):,} P {int((ex[:, 2] == 1).sum()):,} "
             f"C {int((ex[:, 2] == 2).sum()):,})")
    s, i, ns = sets[("train", int(ex[0, 0]))], int(ex[0, 1]), int(ex[0, 2])
    bt = coll[int(ex[0, 0])]([(s, i)], ns, Yt[[s.q_rows[i]]], votes[("train", int(ex[0, 0]))][[i]], dev)
    log(TAG, f"sample target ({'FPC'[ns]}, {b.ids[s.q_rows[i]]}): "
             f"{[b.terms[model.ns_cols[ns][j]] for j in bt['tgt'][0].tolist() if j >= 0][:12]} ... "
             f"(len {(bt['tgt'][0] >= 0).sum().item()}); vote nnz {(bt['vote'] > 0).sum().item()}")

    opt = torch.optim.AdamW(model.parameters(), lr=D["lr"], weight_decay=0.01)
    steps_per_epoch = sum(math.ceil(((ex[:, 2] == ns) & (ex[:, 0] == f)).sum() / (D["batch"] if ns != 1 else
                          max(16, D["batch"] // 2))) for f in (0, 1) for ns in range(3))
    total = D["epochs"] * steps_per_epoch
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, st / total))))
    ev = Evaluator("tv")
    hist, best = [], (-1, None)
    step = 0
    for ep in range(D["epochs"]):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(ex)) if p_ex is None else rng.choice(len(ex), len(ex), p=p_ex)
        # namespace-homogeneous batches
        batches = []
        for f in (0, 1):
            for ns in range(3):
                pn = perm[(ex[perm, 2] == ns) & (ex[perm, 0] == f)]
                bs = D["batch"] if ns != 1 else max(16, D["batch"] // 2)
                batches += [pn[j:j + bs] for j in range(0, len(pn), bs)]
        order = rng.permutation(len(batches))
        tot = {"next": 0.0, "mem": 0.0, "n": 0}
        for bi in order:
            bx = ex[batches[bi]]
            f, ns = int(bx[0, 0]), int(bx[0, 2])
            s = sets[("train", f)]
            rows_q = s.q_rows[bx[:, 1]]
            bt = coll[f]([(s, int(i)) for i in bx[:, 1]], ns, Yt[rows_q], votes[("train", f)][bx[:, 1]], dev)
            l_next, l_mem = forward_batch(model, bt, ns, Hs[ns], cfg, w_ns[ns], dev)
            loss = l_next + D.get("member_weight", 1.0) * l_mem
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tot["next"] += float(l_next) * len(bx)
            tot["mem"] += float(l_mem) * len(bx)
            tot["n"] += len(bx)
            if step % 500 == 0:
                log(TAG, f"ep {ep} step {step}/{total} next {tot['next'] / tot['n']:.4f} member "
                         f"{tot['mem'] / tot['n']:.4f} lr {sched.get_last_lr()[0]:.2e}")
        te = time.time()
        # select on TV-remote when present (Phase A1), else plain TV; always also report the other
        sel = "tvr" if ("tvr", 0) in sets else "tv"
        sc, mats, _ = evaluate_bench(model, sets, spaces, coll, Hs, cfg, b, ev, dev, ("step0",), sel, max_steps=1)
        metric = sc["step0"]["mean_testlike"]
        if metric > best[0]:
            best = (metric, ep,
                    {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        sc_other = None
        if sel == "tvr" and ("tv", 0) in sets:
            sc_other, _, _ = evaluate_bench(model, sets, spaces, coll, Hs, cfg, b, ev, dev, ("step0",), "tv",
                                            max_steps=1)
        rec = {"epoch": ep, "next": tot["next"] / tot["n"], "member": tot["mem"] / tot["n"], sel: sc,
               "train_s": time.time() - t0 - (time.time() - te), "eval_s": time.time() - te}
        if sc_other is not None:
            rec["tv"] = sc_other
        hist.append(rec)
        msg = (f"epoch {ep}: loss next {rec['next']:.4f} member {rec['member']:.4f}; train {rec['train_s']:.0f}s "
               f"eval {rec['eval_s']:.0f}s; {sel} (vocab) " +
               "; ".join(f"{r}: MF {v['MF']:.4f} BP {v['BP']:.4f} CC {v['CC']:.4f} mean {v['mean']:.4f} "
                         f"test-like {v['mean_testlike']:.4f}" for r, v in sc.items()))
        if sc_other is not None:
            v = sc_other["step0"]
            msg += (f"; TV mean {v['mean']:.4f} test-like {v['mean_testlike']:.4f}")
        log(TAG, msg)
    torch.save({"state": best[2], "cfg": cfg, "epoch": best[1]}, run / "model.pt")
    save_json({"history": hist, "best_epoch": best[1], "best_sel_testlike": best[0], "select": sel,
               "n_examples": int(len(ex)), "seconds": time.time() - t_all}, run / "history.json")
    log(TAG, f"saved epoch {best[1]} ({sel} step0 test-like {best[0]:.4f}) -> {run}; total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
