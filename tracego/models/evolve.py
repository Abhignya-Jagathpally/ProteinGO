"""Evolutionary residual: homology prior plus the deviation from the nearest training homolog."""

import torch
from torch import nn


class EvolutionaryResidual(nn.Module):
    """Start at the train-only homology prior and learn a correction.

    The correction sees the query embedding, its difference from the
    identity-weighted training homologs, and a short physicochemical delta.
    The last layer is zero so the untrained model copies the prior.
    """

    def __init__(self, embed_dim: int, n_labels: int, phys_dim: int = 3):
        super().__init__()
        width = embed_dim * 2 + phys_dim
        self.correct = nn.Sequential(
            nn.Linear(width, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_labels),
        )
        nn.init.zeros_(self.correct[-1].weight)
        nn.init.zeros_(self.correct[-1].bias)
        print(
            f"[model] EvolutionaryResidual embed={embed_dim} phys={phys_dim} "
            f"labels={n_labels} last_layer=zero"
        )

    def forward(self, query, delta, phys, knn):
        correction = self.correct(torch.cat([query, delta, phys], dim=-1))
        prior = torch.logit(knn.clamp(1e-4, 1.0 - 1e-4))
        return prior + correction
