"""Choose Gene Ontology targets from the training split only."""

import json
from collections import Counter

import numpy as np
import pandas as pd


ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}
NAMESPACES = {"molecular_function", "biological_process", "cellular_component"}


def select_labels(frame: pd.DataFrame, terms: dict, ancestors, config: dict, work):
    train = frame.index[frame.split == "train"].to_numpy()
    counts = Counter(
        term
        for row in train
        for term in frame.at[row, "go_closed"]
        if term in terms and term not in ROOTS
    )
    n_train = len(train)
    low = int(config["min_train_positive"])
    high = int(n_train * float(config["max_train_prevalence"]))
    print(f"[labels] train_proteins={n_train} min_count={low} max_count={high}")
    valid = [
        term
        for term, count in counts.items()
        if low <= count <= high and terms[term]["namespace"] in NAMESPACES
    ]
    valid.sort(key=lambda term: (-counts[term], term))
    print(f"[labels] terms inside count window={len(valid)}")
    limit = int(config["top_go_terms"])
    quotas = {
        "molecular_function": max(8, limit // 2 - 4),
        "biological_process": max(4, limit // 4 - 2),
        "cellular_component": max(4, limit // 4 - 2),
    }
    selected = []
    for namespace, quota in quotas.items():
        pool = [term for term in valid if terms[term]["namespace"] == namespace]
        take = pool[:quota]
        print(f"[labels] {namespace} eligible={len(pool)} reserved={quota} taken={len(take)}")
        selected.extend(take)
    valid_set = set(valid)
    for child in list(selected):
        parents = sorted(
            (ancestors(child) - {child}) & valid_set,
            key=lambda term: (-counts[term], term),
        )
        for parent in parents:
            if parent not in selected and len(selected) < limit:
                selected.append(parent)
    for term in valid:
        if len(selected) >= limit:
            break
        if term not in selected:
            selected.append(term)
    if len(selected) < 5:
        raise RuntimeError(f"only {len(selected)} informative terms survived the filter")
    index = {term: position for position, term in enumerate(selected)}
    matrix = np.zeros((len(frame), len(selected)), dtype=np.float32)
    for row, go_set in enumerate(frame.go_closed):
        for term in go_set:
            if term in index:
                matrix[row, index[term]] = 1.0
    edges = sorted(
        {
            (index[child], index[parent])
            for child in selected
            for parent in (ancestors(child) - {child})
            if parent in index
        }
    )
    information = -np.log(np.clip(matrix[train].mean(axis=0), 1e-5, 1.0)).astype(np.float32)
    print(
        f"[labels] selected={len(selected)} edges={len(edges)} "
        f"train_density={float(matrix[train].mean()):.4f}"
    )
    namespaces = Counter(terms[term]["namespace"] for term in selected)
    print(f"[labels] namespaces={dict(namespaces)}")
    common = config["common_go"]
    print(f"[labels] shared family term {common} selected={common in index}")
    for term in selected[:12]:
        position = index[term]
        print(
            f"[labels] {term} {terms[term]['name'][:48]} "
            f"train_positive={int(matrix[train, position].sum())} "
            f"{terms[term]['namespace']}"
        )
    meta = [
        {
            "go_id": term,
            "name": terms[term]["name"],
            "namespace": terms[term]["namespace"],
            "train_positive": int(matrix[train, position].sum()),
            "ic": float(information[position]),
        }
        for position, term in enumerate(selected)
    ]
    path = work / "go_vocabulary.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"[labels] wrote {path}")
    return selected, matrix, edges, information
