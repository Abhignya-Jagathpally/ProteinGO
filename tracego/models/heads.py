"""Two readouts of the same frozen residue states."""

import torch
from torch import nn


class MeanPoolMLP(nn.Module):
    """Baseline: one vector per protein, then a linear label head."""

    def __init__(self, dim: int, n_labels: int):
        super().__init__()
        self.scorer = nn.Linear(dim, n_labels)
        print(f"[model] MeanPoolMLP dim={dim} labels={n_labels}")

    def forward(self, residue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(residue.dtype)
        pooled = (residue * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.scorer(pooled)


class ResidueLocalHead(nn.Module):
    """Proposed stub: a residue convolution before pooling.

    Molecular-function signal that lives on a short motif can survive this
    layer. It is averaged away by MeanPoolMLP.
    """

    def __init__(self, dim: int, n_labels: int, kernel: int = 5):
        super().__init__()
        padding = kernel // 2
        self.local = nn.Conv1d(dim, dim, kernel_size=kernel, padding=padding)
        self.scorer = nn.Linear(dim, n_labels)
        print(f"[model] ResidueLocalHead dim={dim} kernel={kernel} labels={n_labels}")

    def forward(self, residue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Conv1d wants [batch, channels, length].
        hidden = self.local(residue.transpose(1, 2)).transpose(1, 2)
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = hidden.sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.scorer(pooled)
