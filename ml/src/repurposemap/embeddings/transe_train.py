"""Training utilities for the TransE model: config, margin ranking loss,
deterministic seeding, and a toy training loop.

Stage A scope: ``train_toy`` is exercised only against tiny synthetic
graphs, as a sanity check that the pieces (indexing, triples, negative
sampling, the model, the loss) fit together and actually learn something.
It is not the real-PrimeKG training pipeline -- that is a later stage of
this milestone, and will need batching/checkpointing machinery this
function does not have (e.g. streaming triples rather than holding every
negative in memory at once).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler
from repurposemap.embeddings.transe_model import TransE

Distance = Literal["l1", "l2"]


@dataclass(frozen=True)
class TransETrainingConfig:
    """Every TransE training knob, with no hidden defaults.

    Deliberately a separate dataclass from
    ``repurposemap.evaluation.split.EvaluationConfig`` -- that config's
    exact key set is locked by an existing test
    (``test_evaluation_config_to_dict_has_no_hidden_fields``), and its knobs
    (path length, candidates, time budgets) are meaningless for a learned
    embedding model.
    """

    embedding_dim: int = 128
    distance: Distance = "l2"
    margin: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 1024
    negatives_per_positive: int = 2
    epochs: int = 100
    seed: int = 42
    max_negative_retries: int = 10

    def __post_init__(self) -> None:
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be at least 1, got {self.embedding_dim}")
        if self.distance not in ("l1", "l2"):
            raise ValueError(f"distance must be 'l1' or 'l2', got {self.distance!r}")
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
            "embedding_dim": self.embedding_dim,
            "distance": self.distance,
            "margin": self.margin,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "negatives_per_positive": self.negatives_per_positive,
            "epochs": self.epochs,
            "seed": self.seed,
            "max_negative_retries": self.max_negative_retries,
        }


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy, and Torch (CPU + CUDA) from one seed.

    CUDA seeding is a no-op without a GPU; it is still called so the same
    function seeds correctly once Stage D moves to one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The standard TransE margin loss (Bordes et al., 2013).

    ``pos_scores``/``neg_scores`` are TransE's ``-distance`` scores (higher
    is better), so in distance terms the loss is
    ``max(0, margin + dist_pos - dist_neg)``, which in score terms is
    ``max(0, margin - pos_scores + neg_scores)``. ``neg_scores`` may hold
    several negatives per positive (one extra trailing dimension), which
    broadcasts against ``pos_scores`` correctly.

    ``valid_mask`` (same shape as ``neg_scores``), when given, zeroes out
    and excludes from the mean any negative slot the sampler could not
    resolve (``TypeAwareNegativeSampler.corrupt`` returned ``None`` for it)
    -- such a slot holds a placeholder value, never a real negative, and
    must not influence the loss. ``None`` (the default) means every slot is
    valid, matching the unmasked behaviour used by Stage A's direct
    loss-formula test.
    """
    losses = torch.clamp(margin - pos_scores.unsqueeze(-1) + neg_scores, min=0.0)
    if valid_mask is None:
        return losses.mean()
    mask = valid_mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass
class TrainingResult:
    """What ``train_toy`` returns: the trained model, loss history, and how
    often the negative sampler had to fall back (visibility, not just a
    final number)."""

    model: TransE
    losses: list[float]
    n_type_aware_fallbacks: int
    n_unresolved_collisions: int


def train_toy(
    triples: np.ndarray,
    n_entities: int,
    n_relations: int,
    sampler: TypeAwareNegativeSampler,
    config: TransETrainingConfig,
) -> TrainingResult:
    """Train a TransE model on a small triple set (synthetic graphs only, Stage A).

    Deterministic under ``config.seed``: seeds Python/NumPy/Torch once at
    the start, and batch shuffling uses an explicit, seeded
    ``torch.Generator`` rather than global RNG state.

    When ``sampler.corrupt`` cannot resolve a valid negative for a slot (it
    returns ``None``), that slot is skipped -- excluded from the loss via
    ``valid_mask`` -- rather than training on a known positive mislabeled as
    a negative. See ``TypeAwareNegativeSampler.corrupt``.
    """
    seed_everything(config.seed)
    model = TransE(n_entities, n_relations, embedding_dim=config.embedding_dim, distance=config.distance)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    h = torch.as_tensor(triples[:, 0], dtype=torch.long)
    r = torch.as_tensor(triples[:, 1], dtype=torch.long)
    t = torch.as_tensor(triples[:, 2], dtype=torch.long)

    n = len(triples)
    batch_size = min(config.batch_size, n)
    generator = torch.Generator().manual_seed(config.seed)

    losses: list[float] = []
    for _epoch in range(config.epochs):
        permutation = torch.randperm(n, generator=generator)
        epoch_loss_total = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_h, batch_r, batch_t = h[idx], r[idx], t[idx]
            batch_n = len(idx)

            # neg_h/neg_t start as a copy of the positive triple's own
            # endpoints -- a harmless placeholder wherever the sampler
            # cannot resolve a valid negative (valid_mask marks those slots
            # False, so margin_ranking_loss excludes them; the placeholder
            # itself is never trained on as if it were a real negative).
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

            pos_scores = model.score(batch_h, batch_r, batch_t)
            neg_scores = model.score(
                neg_h.reshape(-1), neg_r.reshape(-1), neg_t.reshape(-1)
            ).reshape(batch_n, config.negatives_per_positive)

            loss = margin_ranking_loss(pos_scores, neg_scores, config.margin, valid_mask=valid_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_entity_embeddings()

            epoch_loss_total += float(loss.item())
            n_batches += 1

        losses.append(epoch_loss_total / max(n_batches, 1))

    return TrainingResult(
        model=model,
        losses=losses,
        n_type_aware_fallbacks=sampler.n_type_aware_fallbacks,
        n_unresolved_collisions=sampler.n_unresolved_collisions,
    )
