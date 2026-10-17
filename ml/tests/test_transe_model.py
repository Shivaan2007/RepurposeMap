"""The TransE model: forward-pass shape, distance modes, normalization (Stage A).

Requires torch; skipped (not failed) in an environment without it installed,
so the rest of the suite stays green regardless.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from repurposemap.embeddings.transe_model import TransE  # noqa: E402


def test_forward_pass_has_one_score_per_triple():
    model = TransE(n_entities=10, n_relations=3, embedding_dim=8)
    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 1, 0])
    t = torch.tensor([3, 4, 5])

    scores = model(h, r, t)

    assert scores.shape == (3,)


def test_l1_and_l2_distance_both_produce_finite_scores():
    for distance in ("l1", "l2"):
        model = TransE(n_entities=5, n_relations=2, embedding_dim=4, distance=distance)
        scores = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        assert torch.isfinite(scores).all()


def test_invalid_distance_raises():
    with pytest.raises(ValueError, match="distance must be"):
        TransE(n_entities=5, n_relations=2, embedding_dim=4, distance="l3")


def test_entity_embeddings_are_unit_norm_after_construction_and_after_normalize_call():
    model = TransE(n_entities=6, n_relations=2, embedding_dim=4)
    norms = model.entity_embeddings.weight.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    # Perturb the weights (simulating an optimizer step), then confirm
    # normalize_entity_embeddings restores unit norm.
    with torch.no_grad():
        model.entity_embeddings.weight.mul_(3.0)
    model.normalize_entity_embeddings()
    norms_after = model.entity_embeddings.weight.norm(p=2, dim=1)
    assert torch.allclose(norms_after, torch.ones(6), atol=1e-5)


def test_relation_embeddings_are_not_normalized():
    model = TransE(n_entities=6, n_relations=2, embedding_dim=4)
    with torch.no_grad():
        model.relation_embeddings.weight.mul_(5.0)
    norms = model.relation_embeddings.weight.norm(p=2, dim=1)
    # No normalize_relation_embeddings method exists, and nothing else should
    # have touched these weights -- still scaled by 5x, not unit norm.
    assert not torch.allclose(norms, torch.ones(2), atol=1e-3)
    assert not hasattr(model, "normalize_relation_embeddings")


def test_score_matches_the_transe_distance_formula_by_hand():
    model = TransE(n_entities=2, n_relations=1, embedding_dim=3, distance="l2")
    with torch.no_grad():
        model.entity_embeddings.weight[0] = torch.tensor([1.0, 0.0, 0.0])
        model.entity_embeddings.weight[1] = torch.tensor([0.0, 1.0, 0.0])
        model.relation_embeddings.weight[0] = torch.tensor([0.0, 0.0, 0.0])

    score = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    expected = -((1.0**2 + 1.0**2) ** 0.5)  # -||e_h + 0 - e_t||_2
    assert torch.allclose(score, torch.tensor([expected]), atol=1e-5)
