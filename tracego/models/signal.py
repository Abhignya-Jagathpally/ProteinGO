"""Signal-peptide head: frozen ESM2 residue states -> conv + BiLSTM -> emissions of a grammar-constrained CRF over the
16 region states of `tracego.data.signalp` (I/M/O, N/H/C for each of the four tripartite SP classes, P for Sec/SPIII).

The CRF is trained as a multi-tag CRF (SignalP 6.0): the numerator sums over every state path whose state at each
residue lies in that residue's allowed set, so the n/h and h/c borders are latent and learned. The SP class posterior is
exact: every path of class k starts in a class-k state, so p(k | x) = sum of the position-0 marginals of class k.
The cleavage site comes from the Viterbi path restricted to the predicted class (SP length = cleavage position).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracego.data import signalp as sp
from tracego.models.encoder import ESMEncoder, esm_tokenizer

NEG = -1e4


class FrozenESM:
    """ESM2 residue states (BOS/EOS dropped), fp16, no gradients."""

    def __init__(self, name: str, device, layer: int = -1):
        self.enc = ESMEncoder(name, lora_r=0).to(device).eval().half()
        self.tok = esm_tokenizer(name)
        self.device = device
        self.layer = layer
        self.dim = self.enc.hidden

    @torch.no_grad()
    def __call__(self, seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(list(seqs), return_tensors="pt", padding=True)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        out = self.enc.esm(input_ids=ids, attention_mask=am, output_hidden_states=self.layer != -1)
        H = out.last_hidden_state if self.layer == -1 else out.hidden_states[self.layer]
        lens = am.sum(1) - 2
        Lm = int(lens.max())
        H = H[:, 1:Lm + 1]
        mask = torch.arange(Lm, device=self.device)[None] < lens[:, None]
        return H * mask[..., None], mask


class RegionCRF(nn.Module):
    """Linear-chain CRF with hard grammar masks on transitions and start states."""

    def __init__(self):
        super().__init__()
        T, start = sp.grammar()
        self.register_buffer("tmask", torch.as_tensor(T))
        self.register_buffer("smask", torch.as_tensor(start))
        self.trans = nn.Parameter(torch.zeros(sp.S, sp.S))
        self.start = nn.Parameter(torch.zeros(sp.S))

    def potentials(self, trans=None, start=None):
        trans = self.trans if trans is None else trans
        start = self.start if start is None else start
        return trans.masked_fill(~self.tmask, NEG), start.masked_fill(~self.smask, NEG)

    @staticmethod
    def log_partition(E, mask, trans, start, allowed=None, start_ok=None):
        """log sum over paths. E [B, L, S] emissions, mask [B, L], allowed [B, L, S] bool (multi-tag numerator),
        start_ok [B, S] bool (per-sequence start restriction: kingdom or forced class)."""
        if allowed is not None:
            E = E.masked_fill(~allowed, NEG)
        a = start[None] + E[:, 0]
        if start_ok is not None:
            a = a.masked_fill(~start_ok, NEG)
        for t in range(1, E.shape[1]):
            nxt = torch.logsumexp(a[:, :, None] + trans[None], 1) + E[:, t]
            a = torch.where(mask[:, t, None], nxt, a)
        return torch.logsumexp(a, -1)

    def nll(self, E, mask, allowed, start_ok=None):
        trans, start = self.potentials()
        E = E.float()
        logZ = self.log_partition(E, mask, trans, start, None, start_ok)
        num = self.log_partition(E, mask, trans, start, allowed, start_ok)
        return logZ - num

    @staticmethod
    def marginals(E, mask, trans, start, start_ok=None):
        """[B, L, S] posterior state marginals via d logZ / d E."""
        with torch.enable_grad():
            E = E.float().detach().requires_grad_(True)
            logZ = RegionCRF.log_partition(E, mask, trans, start, None, start_ok)
            (g,) = torch.autograd.grad(logZ.sum(), E)
        return g * mask[..., None]

    @staticmethod
    def viterbi(E, mask, trans, start, start_ok=None) -> np.ndarray:
        """[B, L] best state path (-1 on padding)."""
        E = E.float()
        B, L, _ = E.shape
        a = start[None] + E[:, 0]
        if start_ok is not None:
            a = a.masked_fill(~start_ok, NEG)
        bp = []
        for t in range(1, L):
            sc = a[:, :, None] + trans[None]
            best, arg = sc.max(1)
            nxt = best + E[:, t]
            keep = mask[:, t, None]
            a = torch.where(keep, nxt, a)
            bp.append(torch.where(keep, arg, torch.arange(sp.S, device=E.device)[None].expand(B, -1)))
        last = a.argmax(-1)
        path = [last]
        for arg in reversed(bp):
            last = arg.gather(1, last[:, None])[:, 0]
            path.append(last)
        P = torch.stack(path[::-1], 1).cpu().numpy()
        P[~mask.cpu().numpy()] = -1
        return P


class SignalHead(nn.Module):
    def __init__(self, d_in: int = 480, d: int = 128, kernel: int = 5, lstm: int = 128, dropout: float = 0.25):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, d))
        self.conv = nn.Conv1d(d, d, kernel, padding=kernel // 2)
        self.lstm = nn.LSTM(2 * d, lstm, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * lstm, sp.S)
        self.crf = RegionCRF()

    def emissions(self, H, mask):
        x = self.inp(H.float())
        c = F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        x = torch.cat([x, c], -1) * mask[..., None]
        lens = mask.sum(1).cpu()
        pk = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        y, _ = self.lstm(pk)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        return self.out(self.drop(y))

    def forward(self, H, mask, allowed, start_ok=None):
        return self.crf.nll(self.emissions(H, mask), mask, allowed, start_ok)


@torch.no_grad()
def ensemble_potentials(models, H, mask):
    """Product-of-experts ensemble: averaged emissions, transitions and start scores of several heads."""
    E = torch.stack([m.emissions(H, mask).float() for m in models]).mean(0)
    pots = [m.crf.potentials() for m in models]
    trans = torch.stack([p[0] for p in pots]).mean(0)
    start = torch.stack([p[1] for p in pots]).mean(0)
    return E, trans, start


@torch.no_grad()
def decode(E, mask, trans, start, start_ok, seqs) -> dict:
    """Class posteriors, cleavage site and per-region summary from (ensembled) potentials."""
    M = RegionCRF.marginals(E, mask, trans, start, start_ok)
    cls_states = [torch.as_tensor(sp.CLS_STATES[c], device=E.device) for c in range(len(sp.CLASSES))]
    prob = torch.stack([M[:, 0, s].sum(-1) for s in cls_states], -1).clamp(0, 1)
    pred = prob.argmax(-1)
    ok = torch.zeros_like(start_ok)
    for c in range(len(sp.CLASSES)):
        ok[pred == c] = start_ok[pred == c] & torch.isin(torch.arange(sp.S, device=E.device), cls_states[c])[None]
    path = RegionCRF.viterbi(E, mask, trans, start, ok)
    region = sp.STATE_REGION[np.maximum(path, 0)]
    is_sp = (sp.STATE_CLASS[np.maximum(path, 0)] > 0) & (path >= 0)
    cs = np.where(is_sp.all(1), is_sp.sum(1), np.argmin(is_sp, 1))
    cs[pred.cpu().numpy() == 0] = 0
    summ = np.zeros((len(seqs), 9), np.float32)     # n/h/c length, mean KD, net charge
    for i, s in enumerate(seqs):
        if cs[i] == 0:
            continue
        kd, ch = sp.kd_profile(s), sp.charge_profile(s)
        for r in range(3):
            m = (region[i, :len(s)] == r) & is_sp[i, :len(s)]
            if m.any():
                summ[i, r] = m.sum()
                summ[i, 3 + r] = kd[m].mean()
                summ[i, 6 + r] = ch[m].sum()
    return {"prob": prob.cpu().numpy(), "pred": pred.cpu().numpy(), "cs": cs, "path": path, "summary": summ}


REGION_SUMMARY = ("n_len", "h_len", "c_len", "n_kd", "h_kd", "c_kd", "n_charge", "h_charge", "c_charge")


def start_ok_tensor(kingdoms, device) -> torch.Tensor:
    return torch.as_tensor(np.stack([sp.start_mask_for(k) for k in kingdoms]), device=device)


def allowed_tensor(sets: list[np.ndarray], L: int, device) -> torch.Tensor:
    A = np.zeros((len(sets), L, sp.S), bool)
    for i, a in enumerate(sets):
        A[i, :len(a)] = a
    return torch.as_tensor(A, device=device)


class SignalPredictor:
    """The nested-CV heads saved by scripts/run_signal.py. Proteins in the SignalP 6.0 set are predicted out-of-fold
    (the two heads whose outer test partition holds them); every other protein by all heads."""

    def __init__(self, run_dir, device):
        ck = torch.load(run_dir / "models.pt", map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.models = {}
        for key, sd in ck["heads"].items():
            m = SignalHead(**self.cfg["head"]).to(device).eval()
            m.load_state_dict(sd)
            self.models[key] = m
        self.part = dict(zip(ck["ids"], ck["part"]))
        self.device = device

    def groups(self, ids) -> dict:
        """{outer test partition or -1: positions}."""
        p = np.array([self.part.get(i, -1) for i in ids])
        return {int(g): np.where(p == g)[0] for g in np.unique(p)}

    @torch.no_grad()
    def predict(self, H, mask, ids, kingdoms, seqs) -> dict:
        out = None
        start_ok = start_ok_tensor(kingdoms, self.device)
        for g, pos in self.groups(ids).items():
            heads = [m for (test, _), m in self.models.items() if g < 0 or test == g]
            pt = torch.as_tensor(pos, device=self.device)
            E, trans, start = ensemble_potentials(heads, H[pt], mask[pt])
            r = decode(E, mask[pt], trans, start, start_ok[pt], [seqs[i] for i in pos])
            if out is None:
                out = {k: np.zeros((len(ids),) + v.shape[1:], v.dtype) for k, v in r.items() if k != "path"}
            for k in out:
                out[k][pos] = r[k]
        out["crossfit"] = np.array([self.part.get(i, -1) >= 0 for i in ids])
        return out


def features(r: dict) -> np.ndarray:
    """10-d signal feature vector for downstream heads: class probabilities (6), cleavage position / 70 and
    n/h/c lengths / 70."""
    return np.concatenate([r["prob"], r["cs"][:, None] / sp.MAX_LEN, r["summary"][:, :3] / sp.MAX_LEN], 1
                          ).astype(np.float32)
