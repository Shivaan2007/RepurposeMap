"""The toy R-GCN training loop: config validation, determinism, and that
training on a tiny synthetic graph actually reduces loss."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler  # noqa: E402
from repurposemap.embeddings.rgcn_train import RGCNTrainingConfig, train_toy  # noqa: E402


def _toy_triples():
    # 6 drugs, 6 diseases, one indication edge each -- same shape as
    # TransE's own toy fixtures (test_transe_train.py).
    return np.array([[i, 0, 6 + i] for i in range(6)], dtype=np.int64)


def _toy_sampler(seed: int = 0) -> TypeAwareNegativeSampler:
    from repurposemap.embeddings.negative_sampling import TypePools

    type_of_index = tuple(["drug"] * 6 + ["disease"] * 6)
    pools = {"drug": tuple(range(6)), "disease": tuple(range(6, 12))}
    type_pools = TypePools(pools_by_type=pools, type_of_index=type_of_index)
    return TypeAwareNegativeSampler(type_pools, n_entities=12, n_relations=1, positive_triples=_toy_triples(), seed=seed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0},
        {"num_bases": 0},
        {"num_layers": 0},
        {"margin": 0.0},
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"negatives_per_positive": 0},
        {"epochs": 0},
        {"max_negative_retries": 0},
    ],
)
def test_rgcn_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RGCNTrainingConfig(**kwargs)


def test_rgcn_training_config_to_dict_has_no_hidden_fields():
    config = RGCNTrainingConfig()
    assert set(config.to_dict()) == {
        "hidden_dim", "num_bases", "num_layers", "margin", "learning_rate",
        "batch_size", "negatives_per_positive", "epochs", "seed", "max_negative_retries",
    }


def test_toy_training_reduces_loss():
    config = RGCNTrainingConfig(hidden_dim=8, num_bases=2, margin=1.0, learning_rate=1e-2, negatives_per_positive=2, epochs=15, seed=1)
    result = train_toy(_toy_triples(), n_entities=12, n_relations=1, sampler=_toy_sampler(), config=config)

    assert all(np.isfinite(loss) for loss in result.losses)
    assert result.losses[-1] < result.losses[0]


def test_toy_training_is_deterministic_under_fixed_seed():
    config = RGCNTrainingConfig(hidden_dim=4, num_bases=1, margin=1.0, learning_rate=1e-2, negatives_per_positive=1, epochs=3, seed=7)
    result_a = train_toy(_toy_triples(), n_entities=12, n_relations=1, sampler=_toy_sampler(seed=7), config=config)
    result_b = train_toy(_toy_triples(), n_entities=12, n_relations=1, sampler=_toy_sampler(seed=7), config=config)

    assert result_a.losses == pytest.approx(result_b.losses)


def test_toy_training_reports_fallback_counts():
    config = RGCNTrainingConfig(hidden_dim=4, num_bases=1, epochs=2, seed=1)
    result = train_toy(_toy_triples(), n_entities=12, n_relations=1, sampler=_toy_sampler(), config=config)

    assert result.n_type_aware_fallbacks >= 0
    assert result.n_unresolved_collisions >= 0


def test_toy_training_parameters_actually_update():
    seed_everything_config = RGCNTrainingConfig(hidden_dim=4, num_bases=1, learning_rate=1e-1, epochs=1, seed=2)
    from repurposemap.embeddings.transe_train import seed_everything

    seed_everything(seed_everything_config.seed)
    from repurposemap.embeddings.rgcn_model import RGCN

    initial_weights = RGCN(12, 1, hidden_dim=4, num_bases=1, num_layers=2).encoder.entity_embeddings.weight.detach().clone()

    result = train_toy(_toy_triples(), n_entities=12, n_relations=1, sampler=_toy_sampler(seed=2), config=seed_everything_config)

    weights_after = result.model.encoder.entity_embeddings.weight.detach()
    assert torch.isfinite(weights_after).all()
    assert not torch.allclose(weights_after, initial_weights)
