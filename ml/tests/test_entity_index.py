"""Deterministic entity/relation indexing for TransE (Stage A)."""

from __future__ import annotations

import pytest

from graph_builders import primekg_style_graph
from repurposemap.embeddings.entity_index import (
    build_entity_index,
    build_relation_index,
    resolve_relation_key,
)

TARGET = ("drug_protein", "target")
PPI = ("protein_protein", "ppi")
INDICATION = ("indication", "indication")


def _types() -> dict[str, str]:
    return {"D1": "drug", "D2": "drug", "P1": "protein", "P2": "protein", "DIS1": "disease"}


def _graph_forward_order():
    return primekg_style_graph(
        _types(),
        [
            (*TARGET, "D1", "P1"),
            (*PPI, "P1", "P2"),
            (*INDICATION, "D2", "DIS1"),
        ],
    )


def _graph_reverse_insertion_order():
    # Same nodes, same edges, built by inserting them in reverse order --
    # the resulting index must still be identical.
    types = _types()
    reversed_types = dict(reversed(list(types.items())))
    return primekg_style_graph(
        reversed_types,
        [
            (*INDICATION, "D2", "DIS1"),
            (*PPI, "P1", "P2"),
            (*TARGET, "D1", "P1"),
        ],
    )


def test_entity_index_is_sorted_by_node_id():
    index = build_entity_index(_graph_forward_order())
    assert index.index_to_id == tuple(sorted(index.index_to_id))


def test_entity_index_is_stable_across_insertion_order():
    forward = build_entity_index(_graph_forward_order())
    reverse = build_entity_index(_graph_reverse_insertion_order())
    assert forward.index_to_id == reverse.index_to_id
    assert forward.id_to_index == reverse.id_to_index


def test_entity_index_round_trips_to_a_dict():
    original = build_entity_index(_graph_forward_order())
    rebuilt = type(original).from_dict(original.to_dict())
    assert rebuilt == original


def test_entity_index_length_matches_node_count():
    index = build_entity_index(_graph_forward_order())
    assert len(index) == 5


def test_relation_index_is_sorted_and_deterministic():
    forward = build_relation_index(_graph_forward_order())
    reverse = build_relation_index(_graph_reverse_insertion_order())
    assert forward.index_to_id == tuple(sorted(forward.index_to_id))
    assert forward.index_to_id == reverse.index_to_id
    assert forward.id_to_index == reverse.id_to_index


def test_relation_index_round_trips_to_a_dict():
    original = build_relation_index(_graph_forward_order())
    rebuilt = type(original).from_dict(original.to_dict())
    assert rebuilt == original


def test_relation_index_keys_are_full_relation_display_pairs():
    index = build_relation_index(_graph_forward_order())
    assert INDICATION in index.id_to_index
    assert set(index.index_to_id) == {TARGET, PPI, INDICATION}


def test_resolve_relation_key_resolves_the_real_indication_pair():
    index = build_relation_index(_graph_forward_order())
    resolved = resolve_relation_key(index, "indication")
    assert index.index_to_id[resolved] == INDICATION


def test_resolve_relation_key_raises_when_no_pair_matches():
    index = build_relation_index(_graph_forward_order())
    with pytest.raises(ValueError, match="found \\[\\]"):
        resolve_relation_key(index, "contraindication")


def test_resolve_relation_key_raises_on_ambiguous_synthetic_relation_pairs():
    # Two different display_relation values sharing the same relation
    # string -- resolve_relation_key must refuse to guess which one.
    graph = primekg_style_graph(
        {"D1": "drug", "DIS1": "disease", "DIS2": "disease"},
        [
            ("indication", "indication", "D1", "DIS1"),
            ("indication", "strong indication", "D1", "DIS2"),
        ],
    )
    index = build_relation_index(graph)
    with pytest.raises(ValueError, match="expected exactly one"):
        resolve_relation_key(index, "indication")
