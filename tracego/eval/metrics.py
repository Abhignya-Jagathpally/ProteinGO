"""Protein-centric F1 at one threshold. Fmax arrives with the real split."""

import torch


def protein_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= threshold
    true_positive = (predicted & labels.bool()).sum(dim=1).float()
    predicted_positive = predicted.sum(dim=1).float().clamp(min=1.0)
    actual_positive = labels.sum(dim=1).float().clamp(min=1.0)
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    score = float(f1.mean())
    print(
        f"[eval] protein F1 at {threshold:.2f} = {score:.4f} "
        f"on batch={tuple(labels.shape)}"
    )
    return score
