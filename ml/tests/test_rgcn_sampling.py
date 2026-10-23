"""Relation-aware neighborhood sampling: adjacency construction, the
relation-balanced fanout policy, local/global index mapping, determinism,
and -- the hard invariant -- that a held-out indication edge can never
reappear through neighborhood sampling, because the adjacency is built
only from an already leakage-safe triple array."""

from __future__ import annotations

import numpy as np
import pytest

from repurposemap.embeddings.rgcn_sampling import (
    RelationAwareNeighborSampler,
    SamplingConfig,
    _sample_neighbors_relation_balanced,
    _sample_neighbors_relation_balanced_arrays,
    build_relation_adjacency,
)


def test_build_relation_adjacency_outgoing_and_incoming():
    triples = np.array([[0, 1, 2], [0, 1, 3], [2, 5, 0]], dtype=np.int64)
    adjacency = build_relation_adjacency(triples)

    assert adjacency.outgoing[0][1] == (2, 3)
    assert adjacency.outgoing[2][5] == (0,)
    assert adjacency.incoming[2][1] == (0,)
    assert adjacency.incoming[3][1] == (0,)
    assert adjacency.incoming[0][5] == (2,)
    assert 1 not in adjacency.outgoing  # entity 1 never appears as a head or tail


def test_sample_neighbors_relation_balanced_caps_total_at_fanout():
    rng = np.random.default_rng(0)
    candidates = {0: [(i, 1) for i in range(20)], 1: [(i, 0) for i in range(20, 40)]}
    sampled = _sample_neighbors_relation_balanced(candidates, fanout=6, rng=rng)
    assert len(sampled) == 6


def test_sample_neighbors_relation_balanced_splits_evenly_across_relations():
    rng = np.random.default_rng(0)
    # relation 0 has only 2 candidates (a high-degree relation would have
    # many) -- relation 1 has 100. With an even fanout of 10, relation 0
    # must not be starved, and relation 1 must not dominate past its even
    # share before relation 0's small pool is exhausted.
    candidates = {0: [(i, 1) for i in range(2)], 1: [(i, 0) for i in range(2, 102)]}
    sampled = _sample_neighbors_relation_balanced(candidates, fanout=10, rng=rng)
    rel0_count = sum(1 for r, _n, _d in sampled if r == 0)
    rel1_count = sum(1 for r, _n, _d in sampled if r == 1)
    assert rel0_count == 2  # relation 0's entire (small) pool is included
    assert rel0_count + rel1_count == 10


def test_sample_neighbors_relation_balanced_never_exceeds_available_candidates():
    rng = np.random.default_rng(0)
    candidates = {0: [(1, 1), (2, 1)]}
    sampled = _sample_neighbors_relation_balanced(candidates, fanout=50, rng=rng)
    assert len(sampled) == 2  # cannot sample more than exist, even with a huge fanout


def test_sample_neighbors_relation_balanced_empty_candidates_returns_empty():
    rng = np.random.default_rng(0)
    assert _sample_neighbors_relation_balanced({}, fanout=10, rng=rng) == []


def _arrays_from_candidates(candidates_by_relation: dict[int, list[tuple[int, int]]]):
    """Build the CSR-style (rel_ids, offsets, neighbors, directions) arrays
    ``_sample_neighbors_relation_balanced_arrays`` expects, the same way
    ``build_relation_adjacency`` does, so these tests exercise exactly the
    hot-path function's real calling convention."""
    rel_ids_sorted = sorted(candidates_by_relation)
    group_sizes = np.array([len(candidates_by_relation[r]) for r in rel_ids_sorted], dtype=np.int64)
    offsets = np.zeros(len(rel_ids_sorted) + 1, dtype=np.int64)
    np.cumsum(group_sizes, out=offsets[1:])
    total = int(offsets[-1])
    neighbors = np.empty(total, dtype=np.int64)
    directions = np.empty(total, dtype=np.int64)
    pos = 0
    for r, n in zip(rel_ids_sorted, group_sizes.tolist(), strict=True):
        if n:
            arr = np.asarray(candidates_by_relation[r], dtype=np.int64)
            neighbors[pos : pos + n] = arr[:, 0]
            directions[pos : pos + n] = arr[:, 1]
        pos += n
    return np.array(rel_ids_sorted, dtype=np.int64), offsets, neighbors, directions


def test_sample_neighbors_relation_balanced_arrays_caps_total_at_fanout():
    rng = np.random.default_rng(0)
    candidates = {0: [(i, 1) for i in range(20)], 1: [(i, 0) for i in range(20, 40)]}
    sampled = _sample_neighbors_relation_balanced_arrays(*_arrays_from_candidates(candidates), fanout=6, rng=rng)
    assert len(sampled) == 6


def test_sample_neighbors_relation_balanced_arrays_splits_evenly_across_relations():
    rng = np.random.default_rng(0)
    candidates = {0: [(i, 1) for i in range(2)], 1: [(i, 0) for i in range(2, 102)]}
    sampled = _sample_neighbors_relation_balanced_arrays(*_arrays_from_candidates(candidates), fanout=10, rng=rng)
    rel0_count = sum(1 for r, _n, _d in sampled if r == 0)
    rel1_count = sum(1 for r, _n, _d in sampled if r == 1)
    assert rel0_count == 2  # relation 0's entire (small) pool is included
    assert rel0_count + rel1_count == 10


def test_sample_neighbors_relation_balanced_arrays_never_exceeds_available_candidates():
    rng = np.random.default_rng(0)
    candidates = {0: [(1, 1), (2, 1)]}
    sampled = _sample_neighbors_relation_balanced_arrays(*_arrays_from_candidates(candidates), fanout=50, rng=rng)
    assert len(sampled) == 2  # cannot sample more than exist, even with a huge fanout


def test_sample_neighbors_relation_balanced_arrays_empty_returns_empty():
    rng = np.random.default_rng(0)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    assert _sample_neighbors_relation_balanced_arrays(*empty, fanout=10, rng=rng) == []


def test_build_relation_adjacency_combined_matches_outgoing_and_incoming():
    """The precomputed CSR-style ``combined`` field must carry exactly the
    same (neighbor, direction) candidates as the existing ``outgoing``/
    ``incoming`` dicts, just pre-merged and pre-flattened -- not a
    different or lossy view of the same adjacency."""
    triples = np.array([[0, 1, 2], [0, 1, 3], [2, 5, 0]], dtype=np.int64)
    adjacency = build_relation_adjacency(triples)

    rel_ids, offsets, neighbors, directions = adjacency.combined[0]
    reconstructed: dict[int, set[tuple[int, int]]] = {}
    for i, r in enumerate(rel_ids.tolist()):
        start, end = int(offsets[i]), int(offsets[i + 1])
        reconstructed[r] = set(zip(neighbors[start:end].tolist(), directions[start:end].tolist(), strict=True))

    # Entity 0: outgoing (0,1,2) and (0,1,3) -> relation 1, direction 1, neighbors {2,3};
    # incoming (2,5,0) -> relation 5, direction 0, neighbor {2}.
    assert reconstructed[1] == {(2, 1), (3, 1)}
    assert reconstructed[5] == {(2, 0)}


def _drug_disease_graph(n_pairs: int = 20, n_proteins: int = 5):
    """indication edges drug_i -(0)-> disease_i, plus a few target edges
    drug_0 -(1)-> protein_j, so there is a second relation to balance against."""
    triples = []
    for i in range(n_pairs):
        triples.append([i, 0, n_pairs + i])  # drug_i -> disease_i, relation 0 = indication
    for j in range(n_proteins):
        triples.append([0, 1, 2 * n_pairs + j])  # drug_0 -> protein_j, relation 1 = target
    return np.array(triples, dtype=np.int64)


def test_sampled_subgraph_includes_all_seed_entities():
    triples = _drug_disease_graph()
    adjacency = build_relation_adjacency(triples)
    sampler = RelationAwareNeighborSampler(adjacency, n_relations=2, config=SamplingConfig(fanout_hop1=5, fanout_hop2=5), seed=0)

    seeds = [0, 20, 5, 25]
    subgraph = sampler.sample(seeds)

    for s in seeds:
        assert s in subgraph.local_to_global.tolist()


def test_sampled_subgraph_isolated_seed_entity_still_included():
    triples = _drug_disease_graph()
    adjacency = build_relation_adjacency(triples)
    sampler = RelationAwareNeighborSampler(adjacency, n_relations=2, config=SamplingConfig(fanout_hop1=5, fanout_hop2=5), seed=0)

    isolated_entity = 999  # never appears in any triple
    subgraph = sampler.sample([isolated_entity])

    assert isolated_entity in subgraph.local_to_global.tolist()
    assert subgraph.n_local_nodes == 1
    assert len(subgraph.edge_h) == 0


def test_sampled_subgraph_edges_are_doubled_for_message_passing():
    triples = _drug_disease_graph()
    adjacency = build_relation_adjacency(triples)
    sampler = RelationAwareNeighborSampler(adjacency, n_relations=2, config=SamplingConfig(fanout_hop1=20, fanout_hop2=20), seed=0)

    subgraph = sampler.sample([0, 20])

    assert len(subgraph.edge_h) == 2 * subgraph.stats["n_raw_edges"]
    assert subgraph.edge_r.max() < 4  # 2 * n_relations=2


def test_to_local_raises_for_an_id_never_included():
    triples = _drug_disease_graph()
    adjacency = build_relation_adjacency(triples)
    sampler = RelationAwareNeighborSampler(adjacency, n_relations=2, config=SamplingConfig(fanout_hop1=5, fanout_hop2=5), seed=0)
    subgraph = sampler.sample([0, 20])

    with pytest.raises(ValueError, match="not present in sampled subgraph"):
        subgraph.to_local(np.array([99999]))


def test_to_local_correctly_maps_present_ids():
    triples = _drug_disease_graph()
    adjacency = build_relation_adjacency(triples)
    sampler = RelationAwareNeighborSampler(adjacency, n_relations=2, config=SamplingConfig(fanout_hop1=20, fanout_hop2=20), seed=0)
    subgraph = sampler.sample([0, 20])

    local_idx = subgraph.to_local(np.array([0, 20]))
    assert subgraph.local_to_global[local_idx[0]] == 0
    assert subgraph.local_to_global[local_idx[1]] == 20


def test_sampling_is_deterministic_under_fixed_seed():
    triples = _drug_disease_graph(n_pairs=50)
    adjacency = build_relation_adjacency(triples)
    config = SamplingConfig(fanout_hop1=5, fanout_hop2=5)

    sampler_a = RelationAwareNeighborSampler(adjacency, n_relations=2, config=config, seed=7)
    sampler_b = RelationAwareNeighborSampler(adjacency, n_relations=2, config=config, seed=7)

    seeds = [0, 50, 10, 60]
    subgraph_a = sampler_a.sample(seeds)
    subgraph_b = sampler_b.sample(seeds)

    assert np.array_equal(subgraph_a.local_to_global, subgraph_b.local_to_global)
    assert np.array_equal(subgraph_a.edge_h, subgraph_b.edge_h)
    assert np.array_equal(subgraph_a.edge_r, subgraph_b.edge_r)
    assert np.array_equal(subgraph_a.edge_t, subgraph_b.edge_t)


def test_sampler_state_round_trip_reproduces_identical_draws():
    triples = _drug_disease_graph(n_pairs=50)
    adjacency = build_relation_adjacency(triples)
    config = SamplingConfig(fanout_hop1=5, fanout_hop2=5)

    sampler = RelationAwareNeighborSampler(adjacency, n_relations=2, config=config, seed=3)
    sampler.sample([0, 1, 2])  # advance RNG state
    state = sampler.get_state()

    subgraph_before = sampler.sample([10, 60])

    sampler.set_state(state)
    subgraph_after = sampler.sample([10, 60])

    assert np.array_equal(subgraph_before.local_to_global, subgraph_after.local_to_global)
    assert np.array_equal(subgraph_before.edge_h, subgraph_after.edge_h)


def test_held_out_indication_edge_never_reappears_through_sampling():
    # 100 (drug, disease) indication pairs. Diseases 90-99 are "held out"
    # (locked-test style) -- their indication edges are EXCLUDED from the
    # triple array the adjacency is built from, exactly mirroring how
    # tuning_triples/final_triples already exclude held-out indications
    # before this module ever sees them.
    n_pairs = 100
    held_out_diseases = set(range(90, 100))
    all_triples = [[i, 0, n_pairs + i] for i in range(n_pairs)]
    leakage_safe_triples = np.array(
        [t for t in all_triples if (t[2] - n_pairs) not in held_out_diseases], dtype=np.int64
    )
    held_out_pairs = {(i, n_pairs + i) for i in range(n_pairs) if i in held_out_diseases}

    adjacency = build_relation_adjacency(leakage_safe_triples)
    config = SamplingConfig(fanout_hop1=100, fanout_hop2=100)  # deliberately generous fanout
    sampler = RelationAwareNeighborSampler(adjacency, n_relations=1, config=config, seed=0)

    # Seed broadly, including the held-out drug/disease ids themselves --
    # they might still be reachable as entities (e.g. via other relations
    # in a real graph), just never via the excluded indication edge.
    all_entity_ids = list(range(2 * n_pairs))
    subgraph = sampler.sample(all_entity_ids)

    # Reconstruct the actual sampled (h, t) pairs in the REAL (non-doubled)
    # relation space -- inverse message-passing edges have relation id
    # >= n_relations and are excluded here, since this test is about
    # whether the held-out FACT was ever sampled, not about the
    # structural message-passing edges derived from it.
    real_edges = {
        (int(subgraph.local_to_global[h]), int(subgraph.local_to_global[t]))
        for h, r, t in zip(subgraph.edge_h.tolist(), subgraph.edge_r.tolist(), subgraph.edge_t.tolist(), strict=True)
        if r < 1  # real relation space is [0, n_relations) = [0, 1) here
    }
    assert real_edges.isdisjoint(held_out_pairs), f"held-out pair(s) leaked through sampling: {real_edges & held_out_pairs}"
