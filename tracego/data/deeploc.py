"""DeepLoc 2.0 SwissProt training data (Thumuluri et al. 2022, DTU Health Tech): 28,303 eukaryotic proteins with
experimentally annotated (ECO:0000269) localizations in 10 multi-label classes and 5 homology partitions.

Sequences longer than ESM2's 1,022-residue window are cropped to the first and last 511 residues, so both N-terminal
targeting peptides and C-terminal signals (PTS1, KDEL, GPI) stay in view.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tracego import paths

URL = "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv"
RAW = paths.RUNS / "aux" / "raw" / "deeploc2_swissprot.csv"
LOCS = ("Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion", "Plastid",
        "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome")
# cellular-component GO term per location (B4 check; lysosome is_a lytic vacuole is_a vacuole)
GO = {"Cytoplasm": "GO:0005737", "Nucleus": "GO:0005634", "Extracellular": "GO:0005576",
      "Cell membrane": "GO:0005886", "Mitochondrion": "GO:0005739", "Plastid": "GO:0009536",
      "Endoplasmic reticulum": "GO:0005783", "Lysosome/Vacuole": "GO:0005773", "Golgi apparatus": "GO:0005794",
      "Peroxisome": "GO:0005777"}
MAX_RES = 1022


def log(msg: str):
    print(f"[deeploc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def crop(seq: str, max_res: int = MAX_RES) -> str:
    if len(seq) <= max_res:
        return seq
    h = max_res // 2
    return seq[:h] + seq[-(max_res - h):]


@dataclass
class LocData:
    ids: np.ndarray
    kingdom: np.ndarray
    part: np.ndarray
    seqs: list            # cropped
    Y: np.ndarray         # [n, 10] float32
    membrane: np.ndarray  # DeepLoc's separate membrane flag (not a location)

    def __len__(self):
        return len(self.ids)


def load(path: Path = RAW) -> LocData:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download {URL}")
    df = pd.read_csv(path, index_col=0)
    d = LocData(df.ACC.values.astype(str), df.Kingdom.values.astype(str), df.Partition.values.astype(int),
                [crop(s) for s in df.Sequence.values], df[list(LOCS)].values.astype(np.float32),
                df.Membrane.values.astype(np.float32))
    L = df.Sequence.str.len()
    log(f"{len(d):,} proteins; partitions {np.bincount(d.part).tolist()}; kingdoms "
        f"{pd.Series(d.kingdom).value_counts().to_dict()}")
    log(f"length p50 {L.median():.0f} p95 {L.quantile(.95):.0f}; cropped to {MAX_RES}: {(L > MAX_RES).sum():,}; "
        f"labels per protein {d.Y.sum(1).mean():.2f}")
    log("location counts: " + ", ".join(f"{l}={int(c)}" for l, c in zip(LOCS, d.Y.sum(0))))
    return d


if __name__ == "__main__":
    d = load()
    for i in range(3):
        log(f"sample {d.ids[i]} {d.kingdom[i]} part {d.part[i]} len {len(d.seqs[i])} "
            f"locs {[l for l, y in zip(LOCS, d.Y[i]) if y]} seq {d.seqs[i][:40]}...")
