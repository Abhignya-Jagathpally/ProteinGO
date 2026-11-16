"""Diagnostic on TV v2: identity along the embedding alignment. The residue matching j*(i) = argmax_j cos(code_i,
code_j) that MaxSim already computes is read as an alignment; a residue counts as identical when it lies on a
diagonal run, its match cosine exceeds tau, and the matched amino acids agree. No second search: the candidates and
the alignment both come from the one encoder pass.

  python scripts/probe_aaid.py
"""
import numpy as np
import torch

from tracego import paths
from tracego.data.bundle import Bundle
from tracego.eval.score import Evaluator
from tracego.models.retrieval import ResidueCodes, vote
from tracego.train.common import known_mask, log

TAG = "aaid"
AA = "ACDEFGHIKLMNPQRSTVWYXBZUO"


@torch.no_grad()
def feats(codes, aa, q_rows, cand_rows, taus):
    out = np.zeros(cand_rows.shape + (len(taus), 2), np.float32)
    for i, q in enumerate(q_rows):
        x = codes.get(q)
        C, m = codes.padded(cand_rows[i])
        A, _ = aa.padded(cand_rows[i])
        aq = aa.get(q)
        sim = torch.einsum("id,cjd->cij", x, C).float().masked_fill(~m[:, None, :], -1)
        vq, jq = sim.max(2)
        vc, jc = sim.max(1)
        eq_q = A.gather(1, jq) == aq[None]                       # [c, Lq]
        eq_c = aq[jc] == A                                       # [c, Lc]

        def chain(j):
            ch = torch.zeros_like(j, dtype=torch.bool)
            run = j[:, 1:] == j[:, :-1] + 1
            ch[:, 1:] |= run
            ch[:, :-1] |= run
            return ch
        cq, cc = chain(jq), chain(jc)
        nc = m.float().sum(1)
        for t_i, t in enumerate(taus):
            a = ((vq > t) & cq & eq_q).float().mean(1)
            b = ((vc > t) & cc & eq_c & m).float().sum(1) / nc
            out[i, :, t_i, 0] = torch.maximum(a, b).cpu().numpy()
            a = ((vq > t) & cq).float().mean(1)
            b = ((vc > t) & cc & m).float().sum(1) / nc
            out[i, :, t_i, 1] = torch.maximum(a, b).cpu().numpy()
    return out


class AACodes(ResidueCodes):
    def __init__(self, codes, b):
        self.off, self.pos, self.device = codes.off, codes.pos, codes.device
        rows = np.empty(len(codes.pos), np.int64)
        for r, i in codes.pos.items():
            rows[i] = r
        lut = np.full(256, AA.index("X"), np.int64)
        for k, c in enumerate(AA):
            lut[ord(c)] = k
        seqs = b.prot.seq.values
        flat = np.zeros(int(codes.off[-1]), np.int64)
        bad = 0
        for i, r in enumerate(rows):
            s = np.frombuffer(seqs[r].encode(), np.uint8)
            n = codes.off[i + 1] - codes.off[i]
            bad += len(s[:n]) != n
            flat[codes.off[i]:codes.off[i] + min(n, len(s))] = lut[s[:n]]
        log(TAG, f"amino-acid track built; length mismatches {bad}")
        self.X = torch.as_tensor(flat, device=self.device)[:, None]

    def get(self, row):
        i = self.pos[int(row)]
        return self.X[self.off[i]:self.off[i + 1], 0]

    def padded(self, rows):
        seg = [self.get(r) for r in rows]
        Lm = max(len(s) for s in seg)
        C = torch.full((len(seg), Lm), -1, dtype=self.X.dtype, device=self.device)
        m = torch.zeros(len(seg), Lm, dtype=torch.bool, device=self.device)
        for i, s in enumerate(seg):
            C[i, :len(s)] = s
            m[i, :len(s)] = True
        return C, m


def main():
    b = Bundle()
    ids = b.ids
    ref, tv = b.rows("train"), b.rows("tv")
    z = np.load(paths.CACHE / "probe_identity.npz")
    I = z["I"]
    codes = ResidueCodes(paths.RUNS / "full" / "codes_l8_d64.npz")
    aa = AACodes(codes, b)
    taus = (0.0, 0.5, 0.7, 0.8, 0.9)
    F = feats(codes, aa, tv, ref[I], taus)
    ev = Evaluator("tv")
    Yo = b.onto_labels("t0")[ref]
    has = known_mask(b.labels("t0")[ref], b.aspect)
    for t_i, t in enumerate(taus):
        for v, nm in ((0, "chain+aa"), (1, "chain")):
            s = F[:, :, t_i, v]
            o = np.argsort(-s, 1, kind="stable")
            I2, S2 = np.take_along_axis(I, o, 1), np.take_along_axis(s, o, 1)
            for T in (0.03, 0.05, 0.08):
                cur = ev.curves(vote(I2, S2, Yo, T, 50, aspect_cols=ev.o.ns, ns_has=has), ids[tv], onto=True)
                r = ev.summary(cur)
                log(TAG, f"{nm:9s} tau={t} T={T}: mean {r['mean']:.4f} test-like {r['mean_testlike']:.4f} "
                         f"(MF {r['MF']:.4f} BP {r['BP']:.4f} CC {r['CC']:.4f}) bins "
                         f"{[round(x[1], 3) for x in ev.per_bin(cur).values()]}")


if __name__ == "__main__":
    main()
