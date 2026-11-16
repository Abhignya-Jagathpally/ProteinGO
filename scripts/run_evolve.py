"""Homology-prior residual on the kinase family.

The previous residue head did not beat the notebook's frozen ESM2 head.
This run uses the notebook's 50 percent identity cutoff and scores test F1
at the validation threshold. The prior is a train-only MMseqs2 transfer.
The trained piece only corrects that prior, using how the query embedding
and its charge, hydrophobicity, and length differ from those homologs.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracego.data.cluster import assign_splits, cluster_proteins, mmseqs_binary
from tracego.data.labels import ROOTS, NAMESPACES
from tracego.data.obo import ancestor_fn, close_annotations, download_obo, parse_obo
from tracego.data.uniprot import fetch_proteins
from tracego.eval.cafa import THRESHOLDS, evaluate_split, project_scores, protein_f1
from tracego.models.evolve import EvolutionaryResidual

KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}


def phys_vector(sequence: str) -> np.ndarray:
    length = max(len(sequence), 1)
    charge = sum(CHARGE.get(aa, 0.0) for aa in sequence) / length
    hydro = sum(KD.get(aa, 0.0) for aa in sequence) / length
    return np.array([charge, hydro, np.log(length)], dtype=np.float32)


def select_notebook_labels(frame, terms, ancestors, config, work):
    """Same frequency rule as the kinase notebook: common terms first, then ancestors."""
    from collections import Counter

    train = frame.index[frame.split == "train"].to_numpy()
    counts = Counter(
        term
        for row in train
        for term in frame.at[row, "go_closed"]
        if term in terms and term not in ROOTS
    )
    low = int(config["min_train_positive"])
    high = int(len(train) * float(config["max_train_prevalence"]))
    valid = [
        term
        for term, count in counts.items()
        if low <= count <= high and terms[term]["namespace"] in NAMESPACES
    ]
    valid.sort(key=lambda term: (-counts[term], term))
    limit = int(config["top_go_terms"])
    selected = list(valid[: max(8, limit // 2)])
    valid_set = set(valid)
    for child in list(selected):
        for parent in sorted(
            (ancestors(child) - {child}) & valid_set,
            key=lambda term: (-counts[term], term),
        ):
            if parent not in selected and len(selected) < limit:
                selected.append(parent)
    for term in valid:
        if len(selected) >= limit:
            break
        if term not in selected:
            selected.append(term)
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
        f"[labels] notebook rule selected={len(selected)} edges={len(edges)} "
        f"train_density={float(matrix[train].mean()):.4f}"
    )
    for term in selected[:8]:
        position = index[term]
        print(
            f"[labels] {term} {terms[term]['name'][:48]} "
            f"train_positive={int(matrix[train, position].sum())} {terms[term]['namespace']}"
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
    (work / "go_vocabulary.json").write_text(json.dumps(meta, indent=2))
    return selected, matrix, edges, information


def prepare(config, work: Path) -> tuple[pd.DataFrame, np.ndarray, list, np.ndarray, list]:
    source = ROOT / "runs" / "kinase"
    for name in ("uniprot_raw.tsv", "go-basic.obo"):
        destination = work / name
        if not destination.exists() and (source / name).exists():
            shutil.copy(source / name, destination)
            print(f"[data] copied {name} from the 30 percent run")
    frame = fetch_proteins(config, work)
    terms, alt_map = parse_obo(download_obo(work / "go-basic.obo"))
    ancestors = ancestor_fn(terms)
    print("[ontology] closing annotations under is_a and part_of")
    frame["go_closed"] = frame.go_direct.map(
        lambda direct: close_annotations(direct, alt_map, ancestors)
    )
    before = len(frame)
    common = config["common_go"]
    frame = frame[frame.go_closed.map(lambda go_set: common in go_set)].reset_index(drop=True)
    print(f"[ontology] proteins with {common} after true-path={len(frame)} / {before}")
    frame = cluster_proteins(
        frame, work, float(config["cluster_identity"]), float(config["cluster_coverage"])
    )
    frame = assign_splits(frame, config)
    frame[["accession", "cluster", "split", "seq_length", "reviewed"]].to_csv(
        work / "split_assignments.csv", index=False
    )
    selected, matrix, edges, information = select_notebook_labels(
        frame, terms, ancestors, config, work
    )
    namespaces = [terms[term]["namespace"] for term in selected]
    print(f"[labels] matrix={matrix.shape} edges={len(edges)}")
    return frame, matrix, edges, information, namespaces


def search_train(frame: pd.DataFrame, work: Path) -> pd.DataFrame:
    result = work / "train_hits.tsv"
    if result.exists():
        hits = pd.read_csv(result, sep="\t")
        if "query" not in hits.columns:
            hits = pd.read_csv(
                result, sep="\t", names=["query", "target", "pident", "evalue", "bits"]
            )
        print(f"[evolve] cached hits={len(hits)} path={result}")
        return hits
    train = frame[frame.split == "train"]
    query_fa = work / "query.fa"
    train_fa = work / "train.fa"
    print(f"[evolve] writing query={len(frame)} train_db={len(train)}")
    for path, rows in ((query_fa, frame), (train_fa, train)):
        with path.open("w") as handle:
            for row in rows.itertuples(index=False):
                handle.write(f">{row.accession}\n{row.sequence}\n")
    command = [
        str(mmseqs_binary()),
        "easy-search",
        str(query_fa),
        str(train_fa),
        str(result),
        str(work / "search_tmp"),
        "--max-seqs",
        "5",
        "--threads",
        "8",
        "--format-output",
        "query,target,pident,evalue,bits",
    ]
    print("[evolve] " + " ".join(command))
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("[evolve] search tail:\n" + completed.stdout[-1500:])
    if completed.returncode != 0:
        raise RuntimeError("MMseqs2 easy-search failed")
    hits = pd.read_csv(
        result, sep="\t", names=["query", "target", "pident", "evalue", "bits"]
    )
    hits.to_csv(result, sep="\t", index=False)
    print(f"[evolve] hits={len(hits)}")
    return hits


def homology_features(frame, hits, matrix):
    accession_row = {accession: index for index, accession in enumerate(frame.accession)}
    train_rows = set(np.flatnonzero(frame.split.to_numpy() == "train").tolist())
    prior = matrix[list(train_rows)].mean(axis=0).astype(np.float32)
    grouped = {key: table for key, table in hits.groupby("query")}
    knn = np.repeat(prior[None, :], len(frame), axis=0)
    best_identity = np.zeros(len(frame), dtype=np.float32)
    neighbor_rows = np.full(len(frame), -1, dtype=np.int32)
    neighbor_weight = np.zeros(len(frame), dtype=np.float32)
    used = 0
    for accession, row in accession_row.items():
        table = grouped.get(accession)
        if table is None:
            continue
        chosen = []
        for hit in table.itertuples(index=False):
            target = accession_row.get(hit.target)
            if target is None or target not in train_rows or target == row:
                continue
            chosen.append((target, float(hit.bits), float(hit.pident)))
        if not chosen:
            continue
        used += 1
        weights = np.array([max(item[1], 1e-6) for item in chosen], dtype=np.float32)
        labels = np.stack([matrix[item[0]] for item in chosen])
        knn[row] = (labels * weights[:, None]).sum(axis=0) / weights.sum()
        best_identity[row] = max(item[2] for item in chosen)
        top = int(np.argmax(weights))
        neighbor_rows[row] = chosen[top][0]
        neighbor_weight[row] = weights[top] / weights.sum()
    print(f"[evolve] proteins with a train homolog={used} / {len(frame)}")
    for name in ("train", "val", "test"):
        rows = frame.split.to_numpy() == name
        identities = best_identity[rows]
        print(
            f"[evolve] {name} best-hit identity "
            f"median={np.median(identities):.1f} "
            f"p90={np.quantile(identities, 0.9):.1f} "
            f"frac>30={float((identities >= 30).mean()):.3f} "
            f"frac>50={float((identities >= 50).mean()):.3f}"
        )
    return knn, best_identity, neighbor_rows, prior


def embed_pools(frame, config, device, work: Path) -> np.ndarray:
    cache = work / "esm_pool.npy"
    if cache.exists():
        pools = np.load(cache)
        print(f"[features] cached pools={pools.shape}")
        return pools
    from transformers import AutoModel, AutoTokenizer

    print(f"[features] frozen mean pools from {config['esm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["esm_name"])
    encoder = AutoModel.from_pretrained(config["esm_name"]).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    print(f"[features] frozen parameters={sum(p.numel() for p in encoder.parameters())}")
    pools = np.zeros((len(frame), int(encoder.config.hidden_size)), dtype=np.float32)
    batch = 32
    sequences = frame.sequence.tolist()
    for start in range(0, len(sequences), batch):
        chunk = sequences[start : start + batch]
        tokens = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["max_tokens"]),
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"][:, 1:].unsqueeze(-1).float()
        residue = hidden[:, 1:]
        pooled = (residue * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pools[start : start + len(chunk)] = pooled.cpu().numpy()
        if start == 0 or (start // batch) % 50 == 0:
            print(f"[features] pooled {start + len(chunk)} / {len(sequences)} shape={tuple(pooled.shape)}")
    np.save(cache, pools)
    print(f"[features] wrote {cache} shape={pools.shape}")
    del encoder
    torch.cuda.empty_cache()
    return pools


def build_delta(frame, pools, neighbor_rows):
    delta = np.zeros_like(pools)
    phys = np.zeros((len(frame), 3), dtype=np.float32)
    sequences = frame.sequence.tolist()
    copied = 0
    for row, neighbor in enumerate(neighbor_rows):
        query_phys = phys_vector(sequences[row])
        if neighbor < 0:
            phys[row] = query_phys
            continue
        delta[row] = pools[row] - pools[int(neighbor)]
        phys[row] = query_phys - phys_vector(sequences[int(neighbor)])
        copied += 1
    print(
        f"[evolve] delta rows={copied} delta_norm_mean={float(np.linalg.norm(delta, axis=1).mean()):.3f} "
        f"phys={phys.shape}"
    )
    return delta, phys


def f1_at_threshold(y_true, probs, edges, threshold) -> float:
    projected = project_scores(probs, edges)
    return protein_f1(y_true, projected >= threshold)


def sparse_columns(matrix, train_rows, prevalence=0.15):
    frequency = matrix[train_rows].mean(axis=0)
    columns = np.flatnonzero(frequency < prevalence)
    print(
        f"[sparse] prevalence<{prevalence} terms={len(columns)} "
        f"median_train_positive={float(np.median(matrix[train_rows][:, columns].sum(axis=0))):.0f}"
    )
    return columns


def fit_residual(model, tensors, y, edges, information, train_rows, val_rows, device, epochs, lr):
    weight = torch.tensor(information / information.mean(), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_pairs = edges
    dataset = TensorDataset(
        tensors["query"][train_rows],
        tensors["delta"][train_rows],
        tensors["phys"][train_rows],
        tensors["knn"][train_rows],
        torch.tensor(y[train_rows], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    best_state = None
    best_fmax = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for step, (query, delta, phys, knn, labels) in enumerate(loader, start=1):
            query, delta, phys, knn, labels = (
                query.to(device),
                delta.to(device),
                phys.to(device),
                knn.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            logits = model(query, delta, phys, knn)
            loss = (loss_fn(logits, labels) * weight).mean()
            if edge_pairs:
                probs = torch.sigmoid(logits)
                penalty = torch.stack(
                    [torch.relu(probs[:, child] - probs[:, parent]) for child, parent in edge_pairs]
                ).mean()
                loss = loss + 0.1 * penalty
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            seen += len(labels)
            if step == 1 or step % 20 == 0:
                print(f"[train] residual epoch={epoch} step={step} loss={float(loss.detach()):.4f}")
        probs = predict_residual(model, tensors, val_rows, device)
        metrics = evaluate_split(f"residual val epoch {epoch}", y[val_rows], probs, edges, information)
        print(f"[train] residual epoch={epoch} mean_loss={total / max(seen, 1):.4f}")
        if metrics["Fmax"] > best_fmax:
            best_fmax = metrics["Fmax"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"[train] residual new best val Fmax={best_fmax:.4f}")
    model.load_state_dict(best_state)
    print(f"[train] residual restored val Fmax={best_fmax:.4f}")
    return model


def fit_linear(pools, y, edges, information, train_rows, val_rows, device, epochs):
    dim = pools.shape[1]
    model = nn.Linear(dim, y.shape[1]).to(device)
    print(f"[model] mean-pool linear dim={dim} labels={y.shape[1]}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.tensor(pools[train_rows]),
        torch.tensor(y[train_rows]),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    best_state = None
    best_fmax = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (features, labels) in enumerate(loader, start=1):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            if edges:
                probs = torch.sigmoid(logits)
                penalty = torch.stack(
                    [torch.relu(probs[:, child] - probs[:, parent]) for child, parent in edges]
                ).mean()
                loss = loss + 0.1 * penalty
            loss.backward()
            optimizer.step()
            if step == 1 or step % 20 == 0:
                print(f"[train] mean-pool epoch={epoch} step={step} loss={float(loss.detach()):.4f}")
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(pools[val_rows], device=device))).cpu().numpy()
        metrics = evaluate_split(f"mean-pool val epoch {epoch}", y[val_rows], probs, edges, information)
        if metrics["Fmax"] > best_fmax:
            best_fmax = metrics["Fmax"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"[train] mean-pool new best val Fmax={best_fmax:.4f}")
    model.load_state_dict(best_state)
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(pools, device=device))).cpu().numpy()


@torch.no_grad()
def predict_residual(model, tensors, rows, device):
    model.eval()
    logits = model(
        tensors["query"][rows].to(device),
        tensors["delta"][rows].to(device),
        tensors["phys"][rows].to(device),
        tensors["knn"][rows].to(device),
    )
    return torch.sigmoid(logits).cpu().numpy()


def report(name, y_val, y_test, val_probs, test_probs, edges, information, namespaces, sparse):
    val_metrics = evaluate_split(f"{name} val", y_val, val_probs, edges, information, namespaces)
    threshold = val_metrics["F_threshold"]
    test_f1 = f1_at_threshold(y_test, test_probs, edges, threshold)
    test_metrics = evaluate_split(f"{name} test", y_test, test_probs, edges, information, namespaces)
    sparse_f1 = f1_at_threshold(
        y_test[:, sparse], test_probs[:, sparse], [], threshold
    )
    print(
        f"[compare] {name} val_threshold={threshold:.2f} "
        f"test_F1_at_val={test_f1:.4f} test_Fmax={test_metrics['Fmax']:.4f} "
        f"sparse_F1={sparse_f1:.4f}"
    )
    test_metrics["model"] = name
    test_metrics["test_F1_at_val_threshold"] = test_f1
    test_metrics["sparse_F1_at_val_threshold"] = sparse_f1
    test_metrics["val_Fmax"] = val_metrics["Fmax"]
    test_metrics["val_threshold"] = threshold
    return test_metrics


def report_identity_bins(name, y_test, probs, identities, edges, threshold):
    projected = project_scores(probs, edges)
    rows = []
    for label, mask in (
        ("best hit >=50%", identities >= 50),
        ("best hit <50%", identities < 50),
        ("best hit <30%", identities < 30),
    ):
        count = int(mask.sum())
        if count < 5:
            print(f"[remote] {name} {label} n={count}")
            continue
        score = protein_f1(y_test[mask], projected[mask] >= threshold)
        print(f"[remote] {name} {label} n={count} F1_at_val={score:.4f}")
        rows.append({"model": name, "bin": label, "n": count, "F1_at_val_threshold": score})
    return rows


def main():
    with (ROOT / "configs" / "kinase.yaml").open() as handle:
        config = yaml.safe_load(handle)
    config["cluster_identity"] = 0.50
    print(f"[run] {json.dumps(config)}")
    torch.manual_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device}")
    work = ROOT / "runs" / "evolve"
    work.mkdir(parents=True, exist_ok=True)
    frame, matrix, edges, information, namespaces = prepare(config, work)
    hits = search_train(frame, work)
    knn, identities, neighbors, prior = homology_features(frame, hits, matrix)
    pools = embed_pools(frame, config, device, work)
    delta, phys = build_delta(frame, pools, neighbors)
    rows = {name: np.flatnonzero(frame.split.to_numpy() == name) for name in ("train", "val", "test")}
    sparse = sparse_columns(matrix, rows["train"])
    tensors = {
        "query": torch.tensor(pools),
        "delta": torch.tensor(delta),
        "phys": torch.tensor(phys),
        "knn": torch.tensor(knn),
    }
    results = []
    frequency = np.repeat(prior[None, :], len(frame), axis=0)
    for name, probs in (("frequency", frequency), ("homology-knn", knn)):
        results.append(
            report(
                name,
                matrix[rows["val"]],
                matrix[rows["test"]],
                probs[rows["val"]],
                probs[rows["test"]],
                edges,
                information,
                namespaces,
                sparse,
            )
        )
    mean_probs = fit_linear(
        pools, matrix, edges, information, rows["train"], rows["val"], device, epochs=8
    )
    results.append(
        report(
            "mean-pool",
            matrix[rows["val"]],
            matrix[rows["test"]],
            mean_probs[rows["val"]],
            mean_probs[rows["test"]],
            edges,
            information,
            namespaces,
            sparse,
        )
    )
    model = EvolutionaryResidual(pools.shape[1], matrix.shape[1]).to(device)
    model = fit_residual(
        model, tensors, matrix, edges, information, rows["train"], rows["val"], device, epochs=8, lr=1e-3
    )
    residual_probs = predict_residual(model, tensors, np.arange(len(frame)), device)
    results.append(
        report(
            "evolutionary-residual",
            matrix[rows["val"]],
            matrix[rows["test"]],
            residual_probs[rows["val"]],
            residual_probs[rows["test"]],
            edges,
            information,
            namespaces,
            sparse,
        )
    )
    remote_rows = []
    test_identity = identities[rows["test"]]
    y_test = matrix[rows["test"]]
    for name, probs, metrics in (
        ("frequency", frequency, results[0]),
        ("homology-knn", knn, results[1]),
        ("mean-pool", mean_probs, results[2]),
        ("evolutionary-residual", residual_probs, results[3]),
    ):
        remote_rows.extend(
            report_identity_bins(
                name,
                y_test,
                probs[rows["test"]],
                test_identity,
                edges,
                metrics["val_threshold"],
            )
        )
    remote = pd.DataFrame(remote_rows)
    remote_path = work / "remote_scores.csv"
    remote.to_csv(remote_path, index=False)
    print(f"[remote] wrote {remote_path}")
    print(remote.to_string(index=False))
    table = pd.DataFrame(results)
    out = work / "benchmark_scores.csv"
    table.to_csv(out, index=False)
    print(f"[run] wrote {out}")
    columns = [
        "model",
        "val_Fmax",
        "test_F1_at_val_threshold",
        "Fmax",
        "Smin",
        "AUPR",
        "AUC",
        "MCC",
        "sparse_F1_at_val_threshold",
        "Fmax_molecular_function",
    ]
    print(table[[column for column in columns if column in table.columns]].to_string(index=False))
    print("[run] evolutionary residual comparison finished")
    print(f"[evolve] test best-hit identity median={np.median(identities[rows['test']]):.1f}")


if __name__ == "__main__":
    main()
