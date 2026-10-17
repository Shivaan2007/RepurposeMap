"""Triple construction for TransE, including the leakage-safety test (Stage A)."""

from __future__ import annotations

from graph_builders import primekg_style_graph
from repurposemap.embeddings.entity_index import build_entity_index, build_relation_index
from repurposemap.embeddings.triples import build_triples
from repurposemap.evaluation.training_graph import build_training_graph

TARGET = ("drug_protein", "target")
PPI = ("protein_protein", "ppi")
INDICATION = ("indication", "indication")


def test_build_triples_produces_one_triple_per_edge_including_parallel_relations():
    # D1 and P1 are linked by two distinct relations -- both must survive as
    # two distinct triples, never merged into one.
    graph = primekg_style_graph(
        {"D1": "drug", "P1": "protein", "P2": "protein"},
        [
            (*TARGET, "D1", "P1"),
            ("drug_protein", "enzyme", "D1", "P1"),
            (*PPI, "P1", "P2"),
        ],
    )
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)

    result = build_triples(graph, entity_index, relation_index)

    assert len(result) == 3
    assert result.n_self_loops_dropped == 0
    rows = {tuple(row) for row in result.triples}
    assert len(rows) == 3  # the two D1<->P1 relations are distinct triples


def test_build_triples_counts_and_drops_self_loops():
    graph = primekg_style_graph(
        {"D1": "drug", "P1": "protein"},
        [
            (*TARGET, "D1", "P1"),
            ("drug_drug", "synergistic interaction", "D1", "D1"),  # self-loop
        ],
    )
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)

    result = build_triples(graph, entity_index, relation_index)

    assert len(result) == 1
    assert result.n_self_loops_dropped == 1


def test_build_triples_from_an_empty_graph_is_an_empty_triple_set():
    graph = primekg_style_graph({"D1": "drug"}, [])
    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)

    result = build_triples(graph, entity_index, relation_index)

    assert len(result) == 0
    assert result.triples.shape == (0, 3)


def test_build_triples_from_the_leakage_safe_training_graph_excludes_held_out_indication_edges_both_directions():
    # PrimeKG stores an indication edge in both directions; the training
    # graph removes both, and build_triples must never see either one.
    graph = primekg_style_graph(
        {"D1": "drug", "DIS1": "disease", "P1": "protein"},
        [
            (*INDICATION, "D1", "DIS1"),
            (*INDICATION, "DIS1", "D1"),
            (*TARGET, "D1", "P1"),
        ],
    )
    training_graph = build_training_graph(graph, {"DIS1"})
    entity_index = build_entity_index(graph)  # index over every entity, not just the training graph's
    relation_index = build_relation_index(graph)

    result = build_triples(training_graph, entity_index, relation_index)

    indication_idx = relation_index.id_to_index[INDICATION]
    disease_idx = entity_index.id_to_index["DIS1"]
    for head, relation, tail in result.triples:
        if relation == indication_idx:
            assert head != disease_idx and tail != disease_idx
    # The non-indication edge survives; only the two indication edges are gone.
    assert len(result) == 1
