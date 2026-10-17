"""Type-aware negative sampling, including bounded-retry fallback (Stage A)."""

from __future__ import annotations

from graph_builders import primekg_style_graph
from repurposemap.embeddings.entity_index import build_entity_index, build_relation_index
from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler, build_type_pools
from repurposemap.embeddings.positive_store import PositiveTripleStore, build_positive_store, encode_triple
from repurposemap.embeddings.triples import build_triples

TARGET = ("drug_protein", "target")
PPI = ("protein_protein", "ppi")


def _diverse_graph():
    # 3 drugs, 3 proteins, 1 disease -- enough type-diversity for ordinary
    # type-aware corruption to always succeed.
    types = {f"D{i}": "drug" for i in range(1, 4)} | {f"P{i}": "protein" for i in range(1, 4)} | {"DIS1": "disease"}
    edges = [(*TARGET, "D1", "P1"), (*TARGET, "D2", "P2"), (*PPI, "P1", "P2"), (*PPI, "P2", "P3")]
    return primekg_style_graph(types, edges)


def _sampler_for(graph, *, max_retries: int = 10, seed: int = 0) -> tuple[TypeAwareNegativeSampler, object, object]:
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)
    type_pools = build_type_pools(graph, entity_index)
    triple_set = build_triples(graph, entity_index, relation_index)
    sampler = TypeAwareNegativeSampler(
        type_pools,
        n_entities=len(entity_index),
        n_relations=len(relation_index),
        positive_triples=triple_set.triples,
        max_retries=max_retries,
        seed=seed,
    )
    return sampler, entity_index, type_pools


def test_corrupted_entity_matches_the_original_node_types():
    graph = _diverse_graph()
    sampler, entity_index, type_pools = _sampler_for(graph)
    h, r, t = entity_index.id_to_index["D1"], 0, entity_index.id_to_index["P1"]

    for _ in range(200):
        negative = sampler.corrupt(h, r, t)
        assert negative is not None  # this graph is diverse enough to never need to give up
        nh, nr, nt = negative
        assert nr == r
        if nh != h:
            assert type_pools.type_of_index[nh] == type_pools.type_of_index[h]
        if nt != t:
            assert type_pools.type_of_index[nt] == type_pools.type_of_index[t]


def test_known_positive_triples_are_never_returned_as_negatives():
    graph = _diverse_graph()
    sampler, entity_index, _ = _sampler_for(graph, seed=1)
    relation_index = build_relation_index(graph)
    triple_set = build_triples(graph, entity_index, relation_index)
    positives = {tuple(row) for row in triple_set.triples}

    h, r, t = tuple(triple_set.triples[0])
    for _ in range(500):
        negative = sampler.corrupt(h, r, t)
        assert negative is None or negative not in positives


def test_sparse_graph_produces_a_valid_negative_on_every_call():
    # A graph with real type diversity and few positives should basically
    # never need to give up -- confirms the fix to #1 did not turn the
    # common case into a stream of None results.
    graph = _diverse_graph()
    sampler, entity_index, _ = _sampler_for(graph, seed=3)
    h, r, t = entity_index.id_to_index["D1"], 0, entity_index.id_to_index["P1"]

    results = [sampler.corrupt(h, r, t) for _ in range(300)]
    assert all(result is not None for result in results)
    assert sampler.n_unresolved_collisions == 0


def test_corruption_is_deterministic_under_the_same_seed():
    graph = _diverse_graph()
    sampler_a, entity_index, _ = _sampler_for(graph, seed=7)
    sampler_b, _, _ = _sampler_for(graph, seed=7)
    h, r, t = entity_index.id_to_index["D1"], 0, entity_index.id_to_index["P1"]

    sequence_a = [sampler_a.corrupt(h, r, t) for _ in range(50)]
    sequence_b = [sampler_b.corrupt(h, r, t) for _ in range(50)]
    assert sequence_a == sequence_b


def test_fallback_counters_increase_and_no_known_positive_is_ever_returned_on_a_tiny_dense_graph():
    # Two drugs, one relation, every possible drug-drug-ish triple already a
    # "positive" -- a naive reject-and-resample loop over this pool could
    # spin forever, or (before the fix) could be forced to hand back a
    # known positive as a "negative". Neither may happen: it must terminate
    # quickly, and every call that cannot find a real negative must return
    # None rather than a collision.
    graph = primekg_style_graph(
        {"D1": "drug", "D2": "drug"},
        [("drug_drug", "synergistic interaction", "D1", "D2")],
    )
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)
    type_pools = build_type_pools(graph, entity_index)
    h, r, t = entity_index.id_to_index["D1"], 0, entity_index.id_to_index["D2"]

    # Every (h, r, t) combination over this 2-entity, 1-relation universe is
    # marked positive, so no corruption -- type-aware or uniform -- can ever
    # find a non-positive replacement.
    all_possible = {(a, r, b) for a in (h, t) for b in (h, t)}
    sampler = TypeAwareNegativeSampler(
        type_pools,
        n_entities=len(entity_index),
        n_relations=len(relation_index),
        positive_triples=all_possible,
        max_retries=5,
        seed=0,
    )

    results = [sampler.corrupt(h, r, t) for _ in range(20)]  # must return promptly every time, never hang

    assert all(result is None for result in results)  # never a known positive handed back as "valid"
    assert sampler.n_type_aware_fallbacks > 0
    assert sampler.n_unresolved_collisions == 20


def test_sampler_accepts_a_prebuilt_positive_triple_store_with_identical_semantics():
    # The real (Stage C/D scale) usage: build the store once from the FULL
    # leakage-safe training triple set, then reuse it across sampler
    # instances rather than rebuilding it from a raw array each time.
    graph = _diverse_graph()
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)
    type_pools = build_type_pools(graph, entity_index)
    triple_set = build_triples(graph, entity_index, relation_index)
    positives = {tuple(row) for row in triple_set.triples}

    store = build_positive_store(triple_set.triples, n_relations=len(relation_index), n_entities=len(entity_index))
    assert isinstance(store, PositiveTripleStore)

    sampler = TypeAwareNegativeSampler(
        type_pools, n_entities=len(entity_index), n_relations=len(relation_index), positive_triples=store, seed=2
    )
    h, r, t = tuple(triple_set.triples[0])
    for _ in range(300):
        negative = sampler.corrupt(h, r, t)
        assert negative is None or negative not in positives


def test_encode_triple_is_injective_for_distinct_rows():
    n_relations, n_entities = 3, 5
    codes = {
        encode_triple(h, r, t, n_relations, n_entities)
        for h in range(n_entities)
        for r in range(n_relations)
        for t in range(n_entities)
    }
    assert len(codes) == n_entities * n_relations * n_entities
