"""TransE toy training: loss behavior, learning sanity, determinism (Stage A).

Requires torch; skipped (not failed) in an environment without it installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from graph_builders import primekg_style_graph  # noqa: E402
from repurposemap.embeddings.entity_index import build_entity_index, build_relation_index  # noqa: E402
from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler, build_type_pools  # noqa: E402
from repurposemap.embeddings.triples import build_triples  # noqa: E402
from repurposemap.embeddings.transe_train import (  # noqa: E402
    TransETrainingConfig,
    margin_ranking_loss,
    seed_everything,
    train_toy,
)

INDICATION = ("indication", "indication")


def _toy_graph():
    # 6 drugs, 6 diseases, each drug indicated for exactly one disease, in a
    # fixed pattern -- enough signal for TransE to learn something real,
    # enough type diversity (6 of each) for type-aware negatives to always
    # have alternatives.
    drugs = [f"D{i}" for i in range(6)]
    diseases = [f"DIS{i}" for i in range(6)]
    types = {name: "drug" for name in drugs} | {name: "disease" for name in diseases}
    edges = [(*INDICATION, drugs[i], diseases[i]) for i in range(6)]
    return primekg_style_graph(types, edges)


def _toy_setup(seed: int = 42):
    graph = _toy_graph()
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)
    type_pools = build_type_pools(graph, entity_index)
    triple_set = build_triples(graph, entity_index, relation_index)
    sampler = TypeAwareNegativeSampler(
        type_pools,
        n_entities=len(entity_index),
        n_relations=len(relation_index),
        positive_triples=triple_set.triples,
        seed=seed,
    )
    return triple_set, entity_index, relation_index, sampler


def test_margin_ranking_loss_matches_the_formula_by_hand():
    pos = torch.tensor([-1.0, -2.0])  # scores = -distance
    neg = torch.tensor([[-3.0], [-1.5]])
    margin = 1.0
    loss = margin_ranking_loss(pos, neg, margin)
    # max(0, margin - pos + neg) per row, then mean.
    expected = torch.tensor(
        [max(0.0, margin - (-1.0) + (-3.0)), max(0.0, margin - (-2.0) + (-1.5))]
    ).mean()
    assert torch.allclose(loss, expected)


def test_seed_everything_makes_subsequent_randomness_deterministic():
    seed_everything(123)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(123)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_training_loss_decreases_over_a_toy_run():
    triple_set, _entity_index, _relation_index, sampler = _toy_setup()
    config = TransETrainingConfig(embedding_dim=8, epochs=30, batch_size=6, negatives_per_positive=2, seed=42)

    result = train_toy(triple_set.triples, n_entities=12, n_relations=1, sampler=sampler, config=config)

    assert len(result.losses) == config.epochs
    assert all(np.isfinite(loss) for loss in result.losses)
    # Not monotonic every single epoch, but the end must be clearly below the start.
    early_mean = np.mean(result.losses[:5])
    late_mean = np.mean(result.losses[-5:])
    assert late_mean < early_mean


def test_known_positive_triples_score_better_than_negatives_after_training():
    triple_set, entity_index, relation_index, sampler = _toy_setup()
    config = TransETrainingConfig(embedding_dim=8, epochs=50, batch_size=6, negatives_per_positive=4, seed=42)

    result = train_toy(triple_set.triples, n_entities=12, n_relations=1, sampler=sampler, config=config)
    model = result.model

    h = torch.as_tensor(triple_set.triples[:, 0], dtype=torch.long)
    r = torch.as_tensor(triple_set.triples[:, 1], dtype=torch.long)
    t = torch.as_tensor(triple_set.triples[:, 2], dtype=torch.long)
    with torch.no_grad():
        positive_scores = model.score(h, r, t)

        # Controlled negatives: every positive drug paired with a disease it
        # was never indicated for (a fixed wrong offset, not a random draw).
        wrong_t = torch.as_tensor(np.roll(triple_set.triples[:, 2], shift=1), dtype=torch.long)
        negative_scores = model.score(h, r, wrong_t)

    assert positive_scores.mean().item() > negative_scores.mean().item()


def test_training_is_deterministic_under_a_fixed_seed():
    triple_set, _entity_index, _relation_index, sampler_a = _toy_setup(seed=5)
    _triple_set_b, _entity_index_b, _relation_index_b, sampler_b = _toy_setup(seed=5)
    config = TransETrainingConfig(embedding_dim=6, epochs=10, batch_size=6, negatives_per_positive=2, seed=5)

    result_a = train_toy(triple_set.triples, n_entities=12, n_relations=1, sampler=sampler_a, config=config)
    result_b = train_toy(triple_set.triples, n_entities=12, n_relations=1, sampler=sampler_b, config=config)

    assert result_a.losses == pytest.approx(result_b.losses)
    for p_a, p_b in zip(result_a.model.parameters(), result_b.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b)


def test_training_result_reports_negative_sampler_fallback_counts():
    triple_set, _entity_index, _relation_index, sampler = _toy_setup()
    config = TransETrainingConfig(embedding_dim=4, epochs=3, batch_size=6, negatives_per_positive=2, seed=42)

    result = train_toy(triple_set.triples, n_entities=12, n_relations=1, sampler=sampler, config=config)

    assert result.n_type_aware_fallbacks == sampler.n_type_aware_fallbacks
    assert result.n_unresolved_collisions == sampler.n_unresolved_collisions


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_dim", 0),
        ("distance", "l3"),
        ("margin", 0.0),
        ("learning_rate", 0.0),
        ("batch_size", 0),
        ("negatives_per_positive", 0),
        ("epochs", 0),
        ("max_negative_retries", 0),
    ],
)
def test_transe_training_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        TransETrainingConfig(**{field: value})
