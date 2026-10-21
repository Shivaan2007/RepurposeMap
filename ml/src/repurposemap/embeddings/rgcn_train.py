"""Training utilities for the R-GCN model: config and a toy training loop.

Mirrors ``transe_train.py``'s scope exactly -- ``train_toy`` is exercised
only against tiny synthetic graphs, as a sanity check that the pieces
(indexing, triples, message-passing edges, negative sampling, the model,
the loss) fit together and actually learn something. It is not the real-
PrimeKG training pipeline (see ``rgcn_train_gpu.py``).

Reuses ``margin_ranking_loss``/``seed_everything`` from ``transe_train.py``
directly rather than duplicating them -- both are already model-agnostic.

**Training-loop shape differs from TransE's, and this is the one thing to
get right when reading this file.** TransE scores each triple
independently, so its toy loop does a normal step-per-batch SGD. R-GCN
needs entity embeddings that come from running message passing over the
*whole* graph first. Computing those embeddings once, then calling
``optimizer.step()`` after every mini-batch while reusing that same
embeddings tensor, would be wrong: after the first batch's ``step()``
mutates the model's parameters, the previously computed embeddings are
stale, but the autograd graph needed to recompute them cleanly for the next
batch is gone (a graph is normally freed after its one ``backward()``).
Retaining the graph and stepping anyway would train against knowingly stale
representations.

So this loop does the opposite: run the encoder exactly once per epoch,
score every mini-batch against that single embeddings tensor, accumulate
the batch losses into one epoch-level objective, then exactly one
``backward()``/``optimizer.step()`` per epoch -- one coherent autograd
graph, discarded cleanly at the end of the epoch, never reused into the
next one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler
from repurposemap.embeddings.rgcn_model import RGCN, build_message_passing_edges
from repurposemap.embeddings.transe_train import margin_ranking_loss, seed_everything


@dataclass(frozen=True)
class RGCNTrainingConfig:
    """Every R-GCN training knob, with no hidden defaults -- same discipline
    as ``TransETrainingConfig``, with ``hidden_dim``/``num_bases``/
    ``num_layers`` in place of ``embedding_dim``/``distance`` (R-GCN has no
    distance metric; it uses DistMult)."""

    hidden_dim: int = 64
    num_bases: int = 8
    num_layers: int = 2
    margin: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 1024
    negatives_per_positive: int = 2
    epochs: int = 100
    seed: int = 42
    max_negative_retries: int = 10

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be at least 1, got {self.hidden_dim}")
        if self.num_bases < 1:
            raise ValueError(f"num_bases must be at least 1, got {self.num_bases}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")
        if self.margin <= 0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.negatives_per_positive < 1:
            raise ValueError(f"negatives_per_positive must be at least 1, got {self.negatives_per_positive}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.max_negative_retries < 1:
            raise ValueError(f"max_negative_retries must be at least 1, got {self.max_negative_retries}")

    def to_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_bases": self.num_bases,
            "num_layers": self.num_layers,
            "margin": self.margin,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "negatives_per_positive": self.negatives_per_positive,
            "epochs": self.epochs,
            "seed": self.seed,
            "max_negative_retries": self.max_negative_retries,
        }


@dataclass
class RGCNTrainingResult:
    """What ``train_toy`` returns: the trained model, per-epoch loss
    history, and how often the negative sampler had to fall back."""

    model: RGCN
    losses: list[float]
    n_type_aware_fallbacks: int
    n_unresolved_collisions: int


def train_toy(
    triples: np.ndarray,
    n_entities: int,
    n_relations: int,
    sampler: TypeAwareNegativeSampler,
    config: RGCNTrainingConfig,
) -> RGCNTrainingResult:
    """Train an R-GCN model on a small triple set (synthetic graphs only).

    Deterministic under ``config.seed``: seeds Python/NumPy/Torch once at
    the start, and batch shuffling uses an explicit, seeded
    ``torch.Generator`` rather than global RNG state -- same discipline as
    ``transe_train.train_toy``.

    Exactly one encoder forward pass and one optimizer step per epoch (see
    this module's docstring for why) -- not one per mini-batch.
    """
    seed_everything(config.seed)
    model = RGCN(n_entities, n_relations, hidden_dim=config.hidden_dim, num_bases=config.num_bases, num_layers=config.num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    edge_h_np, edge_r_np, edge_t_np = build_message_passing_edges(triples, n_relations)
    edge_h = torch.as_tensor(edge_h_np, dtype=torch.long)
    edge_r = torch.as_tensor(edge_r_np, dtype=torch.long)
    edge_t = torch.as_tensor(edge_t_np, dtype=torch.long)

    h = torch.as_tensor(triples[:, 0], dtype=torch.long)
    r = torch.as_tensor(triples[:, 1], dtype=torch.long)
    t = torch.as_tensor(triples[:, 2], dtype=torch.long)

    n = len(triples)
    batch_size = min(config.batch_size, n)
    generator = torch.Generator().manual_seed(config.seed)

    losses: list[float] = []
    for _epoch in range(config.epochs):
        entity_embeddings = model.encode(edge_h, edge_r, edge_t, n_entities)

        permutation = torch.randperm(n, generator=generator)
        batch_losses: list[torch.Tensor] = []

        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_h, batch_r, batch_t = h[idx], r[idx], t[idx]
            batch_n = len(idx)

            # Same placeholder-on-unresolved-slot convention as
            # transe_train.train_toy: neg_h/neg_t start as a copy of the
            # positive endpoints, harmless wherever valid_mask marks a slot
            # False.
            neg_h = batch_h.clone().unsqueeze(1).repeat(1, config.negatives_per_positive)
            neg_t = batch_t.clone().unsqueeze(1).repeat(1, config.negatives_per_positive)
            valid_mask = torch.ones(batch_n, config.negatives_per_positive, dtype=torch.bool)
            for i in range(batch_n):
                for j in range(config.negatives_per_positive):
                    corrupted = sampler.corrupt(int(batch_h[i]), int(batch_r[i]), int(batch_t[i]))
                    if corrupted is None:
                        valid_mask[i, j] = False
                        continue
                    nh, _nr, nt = corrupted
                    neg_h[i, j] = nh
                    neg_t[i, j] = nt
            neg_r = batch_r.unsqueeze(1).repeat(1, config.negatives_per_positive)

            pos_scores = model.score(entity_embeddings, batch_h, batch_r, batch_t)
            neg_scores = model.score(
                entity_embeddings, neg_h.reshape(-1), neg_r.reshape(-1), neg_t.reshape(-1)
            ).reshape(batch_n, config.negatives_per_positive)

            batch_losses.append(margin_ranking_loss(pos_scores, neg_scores, config.margin, valid_mask=valid_mask))

        epoch_loss = torch.stack(batch_losses).mean()
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        losses.append(float(epoch_loss.item()))

    return RGCNTrainingResult(
        model=model,
        losses=losses,
        n_type_aware_fallbacks=sampler.n_type_aware_fallbacks,
        n_unresolved_collisions=sampler.n_unresolved_collisions,
    )
