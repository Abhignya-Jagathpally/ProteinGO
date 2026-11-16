"""Smoke forward pass: mean-pool MLP versus a residue-local head."""

import sys
from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracego.data.ontology import filter_terms, propagate_records
from tracego.data.smoke import load_smoke_records
from tracego.eval.metrics import protein_f1
from tracego.features.extract import extract_features
from tracego.models.heads import MeanPoolMLP, ResidueLocalHead


def load_config() -> dict:
    path = ROOT / "configs" / "small.yaml"
    print(f"[run] config {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    print(f"[run] config keys={sorted(config)}")
    return config


def pad_batch(features, labels, device):
    width = features[0].residue.shape[-1]
    length = max(item.residue.shape[0] for item in features)
    residue = torch.zeros(len(features), length, width)
    mask = torch.zeros(len(features), length)
    for index, item in enumerate(features):
        size = item.residue.shape[0]
        residue[index, :size] = item.residue
        mask[index, :size] = 1.0
    print(
        f"[run] padded residue={tuple(residue.shape)} mask={tuple(mask.shape)} "
        f"labels={tuple(labels.shape)}"
    )
    return residue.to(device), mask.to(device), labels.to(device)


def label_matrix(records, label_names):
    index = {name: position for position, name in enumerate(label_names)}
    matrix = torch.zeros(len(records), len(label_names))
    for row, record in enumerate(records):
        for term in record.go_terms:
            if term in index:
                matrix[row, index[term]] = 1.0
    print(f"[run] label names={label_names}")
    print(f"[run] label matrix positives={int(matrix.sum())} shape={tuple(matrix.shape)}")
    return matrix


def fit(name, module, residue, mask, labels, steps, lr):
    print(f"[train] start {name} steps={steps} lr={lr}")
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    module.train()
    last = None
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = module(residue, mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        print(f"[train] {name} step={step} loss={last:.4f} logits={tuple(logits.shape)}")
    module.eval()
    with torch.no_grad():
        logits = module(residue, mask)
    f1 = protein_f1(logits, labels)
    print(f"[train] done {name} final_loss={last:.4f} smoke_f1={f1:.4f}")
    return f1


def main():
    config = load_config()
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"[run] device={device}")
    torch.manual_seed(int(config["seed"]))

    records = load_smoke_records(int(config["smoke_n"]))
    records = propagate_records(records)
    train_index = list(range(max(len(records) - 1, 1)))
    kept, _dropped = filter_terms(
        records,
        train_index,
        max_frequency=float(config["max_term_frequency"]),
        min_count=int(config["min_term_count"]),
    )
    if not kept:
        raise RuntimeError("term filter kept nothing; lower max_term_frequency")
    features = extract_features(
        records,
        esm_name=config["esm_name"],
        max_length=int(config["max_length"]),
        hidden_fallback=int(config["hidden_fallback"]),
        device=device,
    )
    label_names = kept
    labels = label_matrix(records, label_names)
    residue, mask, labels = pad_batch(features, labels, device)
    dim = residue.shape[-1]
    n_labels = labels.shape[-1]

    mean_head = MeanPoolMLP(dim, n_labels).to(device)
    local_head = ResidueLocalHead(dim, n_labels).to(device)
    mean_f1 = fit(
        "mean-pool",
        mean_head,
        residue,
        mask,
        labels,
        int(config["train_steps"]),
        float(config["lr"]),
    )
    local_f1 = fit(
        "residue-local",
        local_head,
        residue,
        mask,
        labels,
        int(config["train_steps"]),
        float(config["lr"]),
    )
    print(
        f"[run] smoke comparison mean_pool_f1={mean_f1:.4f} "
        f"residue_local_f1={local_f1:.4f}"
    )
    print("[run] smoke finished. These F1 numbers are not a benchmark.")


if __name__ == "__main__":
    main()
