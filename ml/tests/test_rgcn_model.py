"""The R-GCN encoder + DistMult decoder: shapes, basis decomposition,
self-loop/isolated-node behavior, and the message-passing edge-construction
isolation boundary."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from repurposemap.embeddings.rgcn_model import (  # noqa: E402
    RGCN,
    RGCNEncoder,
    build_message_passing_edges,
)


def test_build_message_passing_edges_doubles_count_and_ranges():
    triples = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 0]], dtype=np.int64)
    n_relations = 2

    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)

    assert len(edge_h) == len(edge_r) == len(edge_t) == 2 * len(triples)
    # Forward half is the original triples, untouched, relation IDs < n_relations.
    assert np.array_equal(edge_h[:3], triples[:, 0])
    assert np.array_equal(edge_r[:3], triples[:, 1])
    assert np.array_equal(edge_t[:3], triples[:, 2])
    assert edge_r[:3].max() < n_relations
    # Inverse half: swapped endpoints, relation IDs shifted into [n_relations, 2*n_relations).
    assert np.array_equal(edge_h[3:], triples[:, 2])
    assert np.array_equal(edge_t[3:], triples[:, 0])
    assert np.array_equal(edge_r[3:], triples[:, 1] + n_relations)
    assert edge_r[3:].min() >= n_relations
    assert edge_r.max() < 2 * n_relations


def test_build_message_passing_edges_never_mutates_input():
    triples = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int64)
    original = triples.copy()

    build_message_passing_edges(triples, n_relations=2)

    assert np.array_equal(triples, original)


def test_build_message_passing_edges_does_not_corrupt_real_relation_ids():
    # The forward half's relation IDs must exactly match the input's own
    # relation column -- no accidental shift, no accidental doubling
    # applied to the real (non-inverse) edges.
    triples = np.array([[0, 5, 1], [1, 7, 2]], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations=10)
    assert list(edge_r[:2]) == [5, 7]


def test_rgcn_encoder_forward_shape():
    n_entities, n_relations = 6, 2
    triples = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 5]], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)
    edge_h_t = torch.as_tensor(edge_h, dtype=torch.long)
    edge_r_t = torch.as_tensor(edge_r, dtype=torch.long)
    edge_t_t = torch.as_tensor(edge_t, dtype=torch.long)

    encoder = RGCNEncoder(n_entities, n_relations, hidden_dim=8, num_bases=2, num_layers=2)
    output = encoder(edge_h_t, edge_r_t, edge_t_t, n_entities)

    assert output.shape == (n_entities, 8)
    assert torch.isfinite(output).all()


def test_isolated_node_still_gets_a_nonzero_representation_via_self_loop():
    # Node index 5 has no edges at all (not in `triples`); only the
    # self-loop transform should produce its representation, and it must
    # not be all-zero or NaN.
    n_entities, n_relations = 6, 1
    triples = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)
    edge_h_t = torch.as_tensor(edge_h, dtype=torch.long)
    edge_r_t = torch.as_tensor(edge_r, dtype=torch.long)
    edge_t_t = torch.as_tensor(edge_t, dtype=torch.long)

    encoder = RGCNEncoder(n_entities, n_relations, hidden_dim=4, num_bases=1, num_layers=1)
    output = encoder(edge_h_t, edge_r_t, edge_t_t, n_entities)

    isolated_row = output[5]
    assert torch.isfinite(isolated_row).all()
    assert not torch.allclose(isolated_row, torch.zeros_like(isolated_row))


def test_no_divide_by_zero_nan_even_with_a_relation_no_node_receives():
    # Relation id 1 appears nowhere in `triples` -- the encoder must not
    # divide by a zero in-degree for it (it should simply skip relations
    # with no edges, not attempt to normalize an empty gather).
    n_entities, n_relations = 4, 2
    triples = np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)
    edge_h_t = torch.as_tensor(edge_h, dtype=torch.long)
    edge_r_t = torch.as_tensor(edge_r, dtype=torch.long)
    edge_t_t = torch.as_tensor(edge_t, dtype=torch.long)

    encoder = RGCNEncoder(n_entities, n_relations, hidden_dim=4, num_bases=2, num_layers=2)
    output = encoder(edge_h_t, edge_r_t, edge_t_t, n_entities)

    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_basis_decomposition_matches_hand_computed_weight():
    from repurposemap.embeddings.rgcn_model import _BasisRelationLayer

    layer = _BasisRelationLayer(in_dim=3, out_dim=3, n_relations_mp=4, num_bases=2)
    relation_id = 1

    expected = torch.einsum("b,bio->io", layer.coefficients[relation_id], layer.bases)
    actual = layer.relation_weight(relation_id)

    assert torch.allclose(expected, actual)


def test_rgcn_decoder_relation_table_sized_to_real_relations_only():
    n_entities, n_relations = 5, 3
    model = RGCN(n_entities, n_relations, hidden_dim=4, num_bases=2, num_layers=1)

    assert model.relation_embeddings.num_embeddings == n_relations
    # The encoder's own relation dimension is doubled for inverses; the
    # decoder's is not -- the two must never be conflated.
    assert model.encoder.n_relations_mp == 2 * n_relations


def test_distmult_score_shape_and_formula():
    n_entities, n_relations = 4, 1
    model = RGCN(n_entities, n_relations, hidden_dim=4, num_bases=1, num_layers=1)
    entity_embeddings = torch.randn(n_entities, 4)

    h = torch.tensor([0, 1])
    r = torch.tensor([0, 0])
    t = torch.tensor([2, 3])
    scores = model.score(entity_embeddings, h, r, t)

    assert scores.shape == (2,)
    expected_0 = (entity_embeddings[0] * model.relation_embeddings(torch.tensor(0)) * entity_embeddings[2]).sum()
    assert torch.allclose(scores[0], expected_0)


def test_encode_does_not_recompute_inside_score():
    # score() must take embeddings as an argument and never call
    # self.encode(...)/self.encoder(...) itself -- enforced by constructing
    # the model, computing embeddings once, mutating the encoder's weights
    # afterward, and confirming score() still uses the (now stale, but
    # explicitly passed) embeddings rather than silently recomputing fresh
    # ones from the mutated encoder.
    n_entities, n_relations = 4, 1
    triples = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)
    model = RGCN(n_entities, n_relations, hidden_dim=4, num_bases=1, num_layers=1)

    embeddings_before = model.encode(
        torch.as_tensor(edge_h, dtype=torch.long), torch.as_tensor(edge_r, dtype=torch.long), torch.as_tensor(edge_t, dtype=torch.long), n_entities
    ).detach()
    score_before = model.score(embeddings_before, torch.tensor([0]), torch.tensor([0]), torch.tensor([1])).item()

    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)

    score_with_stale_embeddings = model.score(embeddings_before, torch.tensor([0]), torch.tensor([0]), torch.tensor([1])).item()
    assert score_before == pytest.approx(score_with_stale_embeddings)


def test_gradient_checkpointing_matches_non_checkpointed_gradients():
    # Gradient checkpointing must not change what is computed -- only
    # when activations are materialized. Proves checkpointed and
    # non-checkpointed encoders produce the same forward output and the
    # same gradients w.r.t. every encoder parameter, from identical
    # initial weights.
    n_entities, n_relations = 10, 2
    triples = np.array([[i, i % n_relations, (i + 1) % n_entities] for i in range(8)], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)
    edge_h_t = torch.as_tensor(edge_h, dtype=torch.long)
    edge_r_t = torch.as_tensor(edge_r, dtype=torch.long)
    edge_t_t = torch.as_tensor(edge_t, dtype=torch.long)

    torch.manual_seed(0)
    encoder_checkpointed = RGCNEncoder(n_entities, n_relations, hidden_dim=6, num_bases=2, num_layers=2, use_gradient_checkpointing=True)
    torch.manual_seed(0)
    encoder_plain = RGCNEncoder(n_entities, n_relations, hidden_dim=6, num_bases=2, num_layers=2, use_gradient_checkpointing=False)

    # Same initial weights by construction (same seed) -- confirm before trusting the gradient comparison.
    for p_a, p_b in zip(encoder_checkpointed.parameters(), encoder_plain.parameters(), strict=True):
        assert torch.equal(p_a, p_b)

    out_checkpointed = encoder_checkpointed(edge_h_t, edge_r_t, edge_t_t, n_entities)
    out_plain = encoder_plain(edge_h_t, edge_r_t, edge_t_t, n_entities)
    assert torch.allclose(out_checkpointed, out_plain, atol=1e-6)

    out_checkpointed.sum().backward()
    out_plain.sum().backward()

    for p_a, p_b in zip(encoder_checkpointed.parameters(), encoder_plain.parameters(), strict=True):
        assert torch.allclose(p_a.grad, p_b.grad, atol=1e-5)


def test_forward_sampled_matches_forward_when_subgraph_is_the_full_graph():
    # Sanity equivalence: if the "sampled" subgraph happens to BE the full
    # graph (local_to_global = identity order), forward_sampled must
    # produce the identical output forward() would.
    n_entities, n_relations = 6, 2
    triples = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 5]], dtype=np.int64)
    edge_h, edge_r, edge_t = build_message_passing_edges(triples, n_relations)
    edge_h_t = torch.as_tensor(edge_h, dtype=torch.long)
    edge_r_t = torch.as_tensor(edge_r, dtype=torch.long)
    edge_t_t = torch.as_tensor(edge_t, dtype=torch.long)

    encoder = RGCNEncoder(n_entities, n_relations, hidden_dim=8, num_bases=2, num_layers=2, use_gradient_checkpointing=False)
    full_output = encoder(edge_h_t, edge_r_t, edge_t_t, n_entities)

    local_to_global = torch.arange(n_entities, dtype=torch.long)
    sampled_output = encoder.forward_sampled(edge_h_t, edge_r_t, edge_t_t, local_to_global)

    assert torch.allclose(full_output, sampled_output, atol=1e-6)


def test_forward_sampled_gathers_only_the_sampled_rows():
    # A 2-node subgraph sampled from a larger entity table must use
    # exactly those 2 rows as its layer-0 input, not the full table.
    n_entities, n_relations = 10, 1
    encoder = RGCNEncoder(n_entities, n_relations, hidden_dim=4, num_bases=1, num_layers=1, use_gradient_checkpointing=False)

    local_to_global = torch.tensor([3, 7], dtype=torch.long)
    edge_h_t = torch.tensor([0], dtype=torch.long)
    edge_r_t = torch.tensor([0], dtype=torch.long)
    edge_t_t = torch.tensor([1], dtype=torch.long)

    output = encoder.forward_sampled(edge_h_t, edge_r_t, edge_t_t, local_to_global)
    assert output.shape == (2, 4)

    # Mutate entity 3's global embedding and confirm it changes local row 0's result.
    with torch.no_grad():
        encoder.entity_embeddings.weight[3] += 100.0
    output_after = encoder.forward_sampled(edge_h_t, edge_r_t, edge_t_t, local_to_global)
    assert not torch.allclose(output, output_after)


def test_encode_sampled_gradients_flow_into_the_global_embedding_table():
    n_entities, n_relations = 10, 1
    model = RGCN(n_entities, n_relations, hidden_dim=4, num_bases=1, num_layers=1)

    local_to_global = torch.tensor([3, 7], dtype=torch.long)
    edge_h_t = torch.tensor([0], dtype=torch.long)
    edge_r_t = torch.tensor([0], dtype=torch.long)
    edge_t_t = torch.tensor([1], dtype=torch.long)

    entity_embeddings = model.encode_sampled(edge_h_t, edge_r_t, edge_t_t, local_to_global)
    entity_embeddings.sum().backward()

    full_grad = model.encoder.entity_embeddings.weight.grad
    assert full_grad is not None
    # Only the sampled global rows (3 and 7) should have received gradient.
    touched = {3, 7}
    for i in range(n_entities):
        if i in touched:
            assert not torch.allclose(full_grad[i], torch.zeros(4))
        else:
            assert torch.allclose(full_grad[i], torch.zeros(4))
