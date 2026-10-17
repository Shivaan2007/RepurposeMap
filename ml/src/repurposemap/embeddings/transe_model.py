"""The TransE knowledge-graph embedding model (Bordes et al., 2013).

``score(h, r, t) = -distance(e_h + e_r, e_t)`` -- a higher score means a more
plausible triple. Implemented as a small ``torch.nn.Module``: one entity
embedding table, one relation embedding table, scored by L1 or L2 distance.
Deliberately simple -- no R-GCN, no attention, no additional KGE library
beyond PyTorch itself, per the approved plan.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

Distance = Literal["l1", "l2"]


class TransE(nn.Module):
    """Entity + relation embeddings, scored by negative L1/L2 distance.

    Entity embeddings are unit-normalized after every optimizer step (call
    ``normalize_entity_embeddings()``) -- the standard TransE regularizer;
    without it, embedding norms can grow unboundedly to trivially shrink the
    margin loss. Relation embeddings are **not** normalized, per the
    approved plan.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 128,
        distance: Distance = "l2",
    ) -> None:
        super().__init__()
        if distance not in ("l1", "l2"):
            raise ValueError(f"distance must be 'l1' or 'l2', got {distance!r}")
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.distance: Distance = distance
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)

        # Standard TransE initialization (Bordes et al., 2013): uniform in
        # [-6/sqrt(dim), 6/sqrt(dim)], entities then normalized to unit norm.
        bound = 6.0 / (embedding_dim**0.5)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        self.normalize_entity_embeddings()

    def normalize_entity_embeddings(self) -> None:
        """L2-unit-normalize every entity embedding row, in place, no grad."""
        with torch.no_grad():
            norms = self.entity_embeddings.weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            self.entity_embeddings.weight.div_(norms)

    def _distance(self, diff: torch.Tensor) -> torch.Tensor:
        if self.distance == "l1":
            return diff.abs().sum(dim=-1)
        return diff.norm(p=2, dim=-1)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``-distance(e_h + e_r, e_t)``. Higher is more plausible."""
        diff = self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
        return -self._distance(diff)

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(h, r, t)
