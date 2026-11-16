"""SignalP 6.0 public training data (Teufel et al. 2022, DTU Health Tech) and its weak region labels.

File format (3 lines per record, first 70 residues of each protein):
  >ACC|KINGDOM|TYPE|PARTITION      KINGDOM in ARCHAEA/EUKARYA/NEGATIVE/POSITIVE, TYPE in NO_SP/SP/LIPO/TAT/TATLIPO/PILIN,
  SEQUENCE                         PARTITION in 0/1/2 (homology partitions at 30% identity)
  LABELS                           S Sec/SPI, L Sec/SPII, T Tat/SPI or Tat/SPII, P Sec/SPIII, I inside, M TM, O outside

There are no ground-truth n/h/c borders, so (as in SignalP 6.0) every SP residue gets a *set* of allowed region states
and the CRF marginalises over every segmentation consistent with the sets and the N -> H -> C grammar:
  Sec SPs  : the first 2 residues are N, the last 3 are C, the centre of the most hydrophobic 7-residue window
             (Kyte-Doolittle) is H; residues in between are N|H (before the peak) or H|C (after it).
  Tat SPs  : everything up to the twin-arginine motif plus one residue is N, the rest follows the Sec rule.
  Lipo SPs : the last 3 residues are the lipobox (the C state of SPII types, cleaved before the +1 Cys);
             between the H peak and the lipobox only H.
  Sec/SPIII: the whole leader is one state P.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracego import paths

URL = "https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta"
RAW = paths.RUNS / "aux" / "raw" / "sp6_dtu.fasta"
MAX_LEN = 70

# global classes: index 0 = no SP; "5+1" probabilities in this order
CLASSES = ("NO_SP", "Sec/SPI", "Sec/SPII", "Tat/SPI", "Tat/SPII", "Sec/SPIII")
TYPE2CLS = {"NO_SP": 0, "SP": 1, "LIPO": 2, "TAT": 3, "TATLIPO": 4, "PILIN": 5}
KINGDOMS = ("ARCHAEA", "EUKARYA", "NEGATIVE", "POSITIVE")

# CRF states: mature/non-SP I, M, O, then N/H/C per SP class (C of the lipo classes = lipobox), then P (Sec/SPIII)
STATES = ("I", "M", "O",
          "N_SPI", "H_SPI", "C_SPI",
          "N_SPII", "H_SPII", "C_SPII",
          "N_TATI", "H_TATI", "C_TATI",
          "N_TATII", "H_TATII", "C_TATII",
          "P_SPIII")
S = len(STATES)
SI = {s: i for i, s in enumerate(STATES)}
STATE_CLASS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5])
# region of each state: -1 mature, 0 N, 1 H, 2 C (the P state of Sec/SPIII is counted as its N region)
STATE_REGION = np.array([-1, -1, -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
CLS_STATES = {c: np.where(STATE_CLASS == c)[0] for c in range(len(CLASSES))}
NHC = {1: ("N_SPI", "H_SPI", "C_SPI"), 2: ("N_SPII", "H_SPII", "C_SPII"), 3: ("N_TATI", "H_TATI", "C_TATI"),
       4: ("N_TATII", "H_TATII", "C_TATII")}

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
          P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
CHARGE = dict(K=1.0, R=1.0, D=-1.0, E=-1.0, H=0.1)   # side-chain net charge at pH 7


def log(msg: str):
    print(f"[signalp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grammar() -> tuple[np.ndarray, np.ndarray]:
    """(transition mask [S, S], start mask [S]). Regions follow N -> H -> C -> cleavage -> O; P -> O; the mature
    part alternates I <-> M <-> O (every transition seen in the data; no direct I <-> O)."""
    T = np.zeros((S, S), bool)
    for a, b in [("I", "I"), ("I", "M"), ("M", "M"), ("M", "I"), ("M", "O"), ("O", "O"), ("O", "M"),
                 ("P_SPIII", "P_SPIII"), ("P_SPIII", "O")]:
        T[SI[a], SI[b]] = True
    for n, h, c in NHC.values():
        for a, b in [(n, n), (n, h), (h, h), (h, c), (c, c), (c, "O")]:
            T[SI[a], SI[b]] = True
    start = np.zeros(S, bool)
    start[[SI["I"], SI["M"], SI["O"], SI["P_SPIII"]] + [SI[v[0]] for v in NHC.values()]] = True
    return T, start


def start_mask_for(kingdom: str | None) -> np.ndarray:
    """Eukaryotes only have Sec/SPI signal peptides (as the SignalP 6.0 server assumes); unknown -> everything."""
    _, start = grammar()
    if kingdom == "EUKARYA":
        start = start & np.isin(STATE_CLASS, [0, 1])
    return start


def kd_profile(seq: str) -> np.ndarray:
    return np.array([KD.get(a, 0.0) for a in seq], np.float32)


def charge_profile(seq: str) -> np.ndarray:
    return np.array([CHARGE.get(a, 0.0) for a in seq], np.float32)


def hydrophobic_peak(seq: str, lo: int, hi: int, w: int = 7) -> int:
    """Centre of the most hydrophobic w-window inside [lo, hi)."""
    kd = kd_profile(seq)
    if hi - lo <= w:
        return (lo + hi - 1) // 2
    win = np.convolve(kd[lo:hi], np.ones(w) / w, mode="valid")
    return lo + int(np.argmax(win)) + w // 2


def region_sets(seq: str, labels: str, cls: int, mode: str = "weak") -> np.ndarray:
    """[L, S] bool: allowed CRF states per residue. mode "weak" = SignalP-6.0-style rules, "latent" = grammar only."""
    L = len(seq)
    A = np.zeros((L, S), bool)
    for i, c in enumerate(labels):
        if c in "IMO":
            A[i, SI[c]] = True
    n = len(labels) - len(labels.lstrip("STLP"))     # SP length = cleavage position
    if cls == 0 or n == 0:
        return A
    if cls == 5:
        A[:n, SI["P_SPIII"]] = True
        return A
    N, H, C = (SI[x] for x in NHC[cls])
    lipo = cls in (2, 4)
    tat = cls in (3, 4)
    if mode == "latent":
        A[:n, [N, H, C]] = True
        A[0, [H, C]] = False                       # grammar: an SP starts in N
        if lipo:
            A[:n, C] = False
            A[n - 3:n, :] = False
            A[n - 3:n, C] = True
        return A
    c_len = 3
    n_end = 2                                      # residues [0, n_end) are N only
    if tat:
        m = re.search("RR", seq[:n])
        if m is not None:
            n_end = min(m.start() + 3, n - c_len - 1)
    peak = max(hydrophobic_peak(seq, n_end, n - c_len), n_end)
    peak = min(peak, n - c_len - 1)
    A[:n_end, N] = True
    if tat and n_end > 2:
        A[n_end:peak + 1, H] = True                # the motif fixes the n/h border
    else:
        A[n_end:peak, [N, H]] = True
        A[peak, H] = True
    if lipo:
        A[peak + 1:n - c_len, H] = True
    else:
        A[peak + 1:n - c_len, [H, C]] = True
    A[n - c_len:n, C] = True
    return A


@dataclass
class SignalData:
    ids: np.ndarray        # accession
    kingdom: np.ndarray    # str
    cls: np.ndarray        # int, CLASSES index
    part: np.ndarray       # int, 0..2
    seqs: list
    labels: list
    cs: np.ndarray         # true cleavage position (SP length), 0 for no SP

    def __len__(self):
        return len(self.ids)


def load(path: Path = RAW) -> SignalData:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download {URL}")
    lines = [x.strip() for x in open(path) if x.strip()]
    ids, king, cls, part, seqs, labs, cs = [], [], [], [], [], [], []
    for i in range(0, len(lines), 3):
        acc, k, t, p = lines[i][1:].split("|")
        s, lab = lines[i + 1][:MAX_LEN], lines[i + 2][:MAX_LEN]
        assert len(s) == len(lab), acc
        ids.append(acc), king.append(k), cls.append(TYPE2CLS[t]), part.append(int(p)), seqs.append(s), labs.append(lab)
        cs.append(len(lab) - len(lab.lstrip("STLP")))
    d = SignalData(np.array(ids), np.array(king), np.array(cls), np.array(part), seqs, labs, np.array(cs))
    log(f"{len(d):,} sequences from {path.name}; partitions {np.bincount(d.part).tolist()}")
    for k in KINGDOMS:
        m = d.kingdom == k
        log(f"  {k:9s} n={m.sum():6,d}  " + "  ".join(f"{CLASSES[c]}={int((d.cls[m] == c).sum())}"
                                                  for c in range(len(CLASSES))))
    return d


if __name__ == "__main__":
    d = load()
    for i in [int(np.where(d.cls == c)[0][0]) for c in range(len(CLASSES))]:
        A = region_sets(d.seqs[i], d.labels[i], d.cls[i])
        tag = "".join("".join(STATES[j][0] for j in np.where(r)[0]) if r.sum() == 1 else
                      "/" if r.sum() > 1 else "?" for r in A)
        log(f"{d.ids[i]} {CLASSES[d.cls[i]]} cs={d.cs[i]}\n  seq {d.seqs[i]}\n  lab {d.labels[i]}\n  set {tag}")
