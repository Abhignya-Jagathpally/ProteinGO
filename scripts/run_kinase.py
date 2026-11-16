"""Real PF00069 kinase run: frozen ESM2 mean pool versus a residue-local head."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracego.data.cluster import assign_splits, cluster_proteins
from tracego.data.labels import select_labels
from tracego.data.obo import ancestor_fn, close_annotations, download_obo, parse_obo
from tracego.data.uniprot import fetch_proteins
from tracego.eval.cafa import evaluate_split, project_scores
from tracego.models.heads import MeanPoolMLP, ResidueLocalHead


def load_config() -> dict:
    path = ROOT / "configs" / "kinase.yaml"
    print(f"[run] config {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    print(f"[run] {json.dumps(config)}")
    return config


class KinaseSet(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: np.ndarray, indices: np.ndarray):
        self.frame = frame.reset_index(drop=True)
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])
        return self.frame.at[row, "sequence"], self.labels[row], self.frame.at[row, "accession"]


def collate(batch):
    sequences, labels, accessions = zip(*batch)
    return list(sequences), torch.tensor(np.stack(labels), dtype=torch.float32), list(accessions)


def embed_batch(sequences, tokenizer, encoder, max_tokens, device):
    tokens = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        hidden = encoder(**tokens).last_hidden_state
    # Drop the BOS token. The attention mask still marks padding and EOS.
    residue = hidden[:, 1:]
    mask = tokens["attention_mask"][:, 1:].float()
    return residue, mask


def hierarchy_penalty(logits, edges):
    if not edges:
        return logits.new_zeros(())
    probs = torch.sigmoid(logits)
    losses = [torch.relu(probs[:, child] - probs[:, parent]) for child, parent in edges]
    return torch.stack(losses).mean()


def predict(module, loader, tokenizer, encoder, config, device):
    module.eval()
    chunks = []
    with torch.no_grad():
        for sequences, labels, _accessions in loader:
            residue, mask = embed_batch(
                sequences, tokenizer, encoder, int(config["max_tokens"]), device
            )
            chunks.append(torch.sigmoid(module(residue, mask)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def fit(name, module, train_loader, val_loader, tokenizer, encoder, edges, config, device, y_val, information, namespaces):
    print(
        f"[train] {name} epochs={config['epochs']} lr={config['lr']} "
        f"hierarchy_lambda={config['hierarchy_lambda']}"
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=float(config["lr"]))
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_fmax = -1.0
    for epoch in range(1, int(config["epochs"]) + 1):
        module.train()
        total = 0.0
        seen = 0
        for step, (sequences, labels, _accessions) in enumerate(train_loader, start=1):
            residue, mask = embed_batch(
                sequences, tokenizer, encoder, int(config["max_tokens"]), device
            )
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = module(residue, mask)
            loss = loss_fn(logits, labels) + float(config["hierarchy_lambda"]) * hierarchy_penalty(
                logits, edges
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(sequences)
            seen += len(sequences)
            if step == 1 or step % 50 == 0:
                print(
                    f"[train] {name} epoch={epoch} step={step} "
                    f"loss={float(loss.detach()):.4f} batch={len(sequences)} "
                    f"residue={tuple(residue.shape)}"
                )
        probs = predict(module, val_loader, tokenizer, encoder, config, device)
        metrics = evaluate_split(
            f"{name} val epoch {epoch}", y_val, probs, edges, information, namespaces
        )
        print(f"[train] {name} epoch={epoch} mean_loss={total / max(seen, 1):.4f}")
        if metrics["Fmax"] > best_fmax:
            best_fmax = metrics["Fmax"]
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            print(f"[train] {name} new best val Fmax={best_fmax:.4f}")
    module.load_state_dict(best_state)
    print(f"[train] {name} restored best val Fmax={best_fmax:.4f}")
    return module


def subset(frame, labels, name):
    indices = np.flatnonzero(frame.split.to_numpy() == name)
    print(f"[data] loader {name} proteins={len(indices)}")
    return DataLoader(
        KinaseSet(frame, labels, indices),
        batch_size=int(config_batch()),
        shuffle=name == "train",
        collate_fn=collate,
    )


def config_batch():
    return int(ACTIVE["batch_size"])


ACTIVE = {}


def frequency_baseline(y_train, y_target):
    prior = y_train.mean(axis=0)
    print(f"[baseline] train prevalence shape={prior.shape} mean={float(prior.mean()):.4f}")
    return np.repeat(prior[None, :], len(y_target), axis=0)


def main():
    global ACTIVE
    config = load_config()
    ACTIVE = config
    torch.manual_seed(int(config["seed"]))
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"[run] device={device}")
    work = ROOT / "runs" / "kinase"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[run] work={work}")

    frame = fetch_proteins(config, work)
    obo = download_obo(work / "go-basic.obo")
    terms, alt_map = parse_obo(obo)
    ancestors = ancestor_fn(terms)
    print("[ontology] closing annotations under is_a and part_of")
    frame["go_closed"] = frame.go_direct.map(
        lambda direct: close_annotations(direct, alt_map, ancestors)
    )
    before = len(frame)
    common = config["common_go"]
    frame = frame[frame.go_closed.map(lambda go_set: common in go_set)].reset_index(drop=True)
    print(f"[ontology] proteins with {common} after true-path={len(frame)} / {before}")
    if len(frame) < int(config["min_records"]):
        raise RuntimeError("too few proteins retain the shared GO term")

    frame = cluster_proteins(
        frame,
        work,
        identity=float(config["cluster_identity"]),
        coverage=float(config["cluster_coverage"]),
    )
    frame = assign_splits(frame, config)
    frame[["accession", "cluster", "split", "seq_length", "reviewed"]].to_csv(
        work / "split_assignments.csv", index=False
    )
    print(f"[split] wrote {work / 'split_assignments.csv'}")

    selected, matrix, edges, information = select_labels(frame, terms, ancestors, config, work)
    namespaces = [terms[term]["namespace"] for term in selected]
    print(f"[labels] matrix={matrix.shape} edges={len(edges)}")
    train_rows = np.flatnonzero(frame.split.to_numpy() == "train")
    val_rows = np.flatnonzero(frame.split.to_numpy() == "val")
    test_rows = np.flatnonzero(frame.split.to_numpy() == "test")

    from transformers import AutoModel, AutoTokenizer

    print(f"[features] loading frozen {config['esm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["esm_name"])
    encoder = AutoModel.from_pretrained(config["esm_name"]).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    print(f"[features] frozen parameters={sum(p.numel() for p in encoder.parameters())}")

    train_loader = subset(frame, matrix, "train")
    val_loader = subset(frame, matrix, "val")
    test_loader = subset(frame, matrix, "test")
    dim = int(encoder.config.hidden_size)
    print(f"[features] hidden_size={dim} labels={matrix.shape[1]}")

    results = []
    for split_name, rows in (("val", val_rows), ("test", test_rows)):
        baseline = frequency_baseline(matrix[train_rows], matrix[rows])
        metrics = evaluate_split(
            f"frequency {split_name}",
            matrix[rows],
            baseline,
            edges,
            information,
            namespaces,
        )
        metrics["model"] = "frequency"
        results.append(metrics)

    for name, module in (
        ("mean-pool", MeanPoolMLP(dim, matrix.shape[1])),
        ("residue-local", ResidueLocalHead(dim, matrix.shape[1])),
    ):
        module = module.to(device)
        module = fit(
            name,
            module,
            train_loader,
            val_loader,
            tokenizer,
            encoder,
            edges,
            config,
            device,
            matrix[val_rows],
            information,
            namespaces,
        )
        for split_name, loader, rows in (
            ("val", val_loader, val_rows),
            ("test", test_loader, test_rows),
        ):
            probs = predict(module, loader, tokenizer, encoder, config, device)
            metrics = evaluate_split(
                f"{name} {split_name}",
                matrix[rows],
                probs,
                edges,
                information,
                namespaces,
            )
            metrics["model"] = name
            results.append(metrics)
        del module
        torch.cuda.empty_cache()

    table = pd.DataFrame(results)
    out = work / "benchmark_scores.csv"
    table.to_csv(out, index=False)
    print(f"[run] wrote {out}")
    print(table.to_string(index=False))
    print("[run] kinase comparison finished")


if __name__ == "__main__":
    main()
