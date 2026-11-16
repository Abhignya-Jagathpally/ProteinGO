"""Frozen ESM2 residue states, with a deterministic fallback."""

from dataclasses import dataclass

import torch

from tracego.data.smoke import SmokeRecord


@dataclass
class SequenceFeatures:
    accession: str
    residue: torch.Tensor  # [L, D]
    pooled: torch.Tensor  # [D]
    source: str


def _fallback_embed(sequence: str, dim: int) -> torch.Tensor:
    """Hash amino acids into a fixed vector. Prints that ESM2 was not used."""
    table = torch.zeros(len(sequence), dim)
    for index, amino in enumerate(sequence):
        generator = torch.Generator().manual_seed(ord(amino) + index * 17)
        table[index] = torch.randn(dim, generator=generator)
    return table


def _load_esm(name: str, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    print(f"[features] loading frozen encoder {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(
        f"[features] encoder frozen parameters="
        f"{sum(p.numel() for p in model.parameters())}"
    )
    return tokenizer, model


def extract_features(
    records: list[SmokeRecord],
    esm_name: str,
    max_length: int,
    hidden_fallback: int,
    device: torch.device,
) -> list[SequenceFeatures]:
    tokenizer = model = None
    source = "fallback-hash"
    try:
        tokenizer, model = _load_esm(esm_name, device)
        source = esm_name
    except Exception as error:
        print(f"[features] ESM2 load failed ({type(error).__name__}: {error})")
        print(f"[features] using hash fallback dim={hidden_fallback}")

    features: list[SequenceFeatures] = []
    for record in records:
        sequence = record.sequence
        limit = max_length - 2
        if len(sequence) > limit:
            print(
                f"[features] {record.accession} truncated "
                f"{len(sequence)} -> {limit} residues"
            )
            sequence = sequence[:limit]
        if tokenizer is None or model is None:
            residue = _fallback_embed(sequence, hidden_fallback).to(device)
        else:
            tokens = tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state[0]
            # Drop BOS and EOS so the length matches residues that were tokenized.
            residue = hidden[1:-1]
            if residue.shape[0] != len(sequence):
                print(
                    f"[features] length mismatch {record.accession}: "
                    f"tokens={residue.shape[0]} residues={len(sequence)}; truncating"
                )
                limit = min(residue.shape[0], len(sequence))
                residue = residue[:limit]
        pooled = residue.mean(dim=0)
        print(
            f"[features] {record.accession} source={source} "
            f"residue={tuple(residue.shape)} pooled={tuple(pooled.shape)}"
        )
        features.append(
            SequenceFeatures(record.accession, residue.cpu(), pooled.cpu(), source)
        )
    return features
