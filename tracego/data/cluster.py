"""MMseqs2 clustering and a cluster-disjoint train/validation/test split."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def mmseqs_binary() -> Path:
    binary = Path(__file__).resolve().parents[2] / "third_party" / "mmseqs" / "bin" / "mmseqs"
    if not binary.exists():
        raise FileNotFoundError(f"mmseqs missing at {binary}")
    print(f"[mmseqs] binary={binary}")
    return binary


def cluster_proteins(frame: pd.DataFrame, work: Path, identity: float, coverage: float) -> pd.DataFrame:
    fasta = work / "kinase.fa"
    print(f"[mmseqs] writing fasta proteins={len(frame)} path={fasta}")
    with fasta.open("w") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.accession}\n{row.sequence}\n")
    cluster_tsv = work / "kinase_cluster.tsv"
    if not cluster_tsv.exists():
        base = work / "kinase_clusters"
        command = [
            str(mmseqs_binary()),
            "easy-cluster",
            str(fasta),
            str(base),
            str(work / "mmseqs_tmp"),
            "--min-seq-id",
            str(identity),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--threads",
            "8",
        ]
        print("[mmseqs] " + " ".join(command))
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("[mmseqs] output tail:\n" + result.stdout[-2000:])
        if result.returncode != 0:
            raise RuntimeError("MMseqs2 easy-cluster failed")
        produced = Path(str(base) + "_cluster.tsv")
        produced.replace(cluster_tsv)
    mapping = pd.read_csv(cluster_tsv, sep="\t", names=["representative", "accession"])
    print(f"[mmseqs] members={len(mapping)} clusters={mapping.representative.nunique()}")
    if len(mapping) != len(frame):
        raise RuntimeError(
            f"cluster rows {len(mapping)} != proteins {len(frame)}"
        )
    groups = dict(zip(mapping.accession, mapping.representative))
    frame = frame.copy()
    frame["cluster"] = frame.accession.map(groups)
    if frame.cluster.isna().any():
        raise RuntimeError("some accessions were not clustered")
    sizes = frame.cluster.value_counts()
    print(f"[mmseqs] largest clusters={sizes.head(5).to_dict()}")
    return frame


def assign_splits(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.copy()
    index = np.arange(len(frame))
    groups = frame.cluster.to_numpy()
    holdout = float(config["val_fraction"]) + float(config["test_fraction"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=int(config["seed"]))
    train_index, temp_index = next(splitter.split(index, groups=groups))
    test_share = float(config["test_fraction"]) / holdout
    second = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=int(config["seed"]) + 1)
    val_rel, test_rel = next(second.split(temp_index, groups=groups[temp_index]))
    val_index = temp_index[val_rel]
    test_index = temp_index[test_rel]
    frame["split"] = ""
    frame.loc[train_index, "split"] = "train"
    frame.loc[val_index, "split"] = "val"
    frame.loc[test_index, "split"] = "test"
    for name in ("train", "val", "test"):
        part = frame[frame.split == name]
        print(
            f"[split] {name} proteins={len(part)} "
            f"fraction={len(part) / len(frame):.3f} clusters={part.cluster.nunique()}"
        )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frame.loc[frame.split == left, "cluster"]) & set(
            frame.loc[frame.split == right, "cluster"]
        )
        if overlap:
            raise RuntimeError(f"clusters overlap between {left} and {right}")
    print("[split] cluster ids are disjoint across train, validation, and test")
    return frame
