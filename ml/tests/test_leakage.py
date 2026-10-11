"""Reverse-edge leakage protection.

PrimeKG stores an indication edge in both directions: drug -> disease and a
separate disease -> drug edge for the same known treatment. A leakage check
that only looks at one direction can miss a real leak. These tests check
`find_remaining_test_pairs` directly, not through `assert_no_leakage`'s
combined result, so they fail on their own if the direction-blind version of
the function is ever reintroduced, independent of whether the other,
direction-agnostic check (`find_remaining_test_indications`) would also
happen to catch it.
"""

from __future__ import annotations

from graph_builders import primekg_style_graph

from repurposemap.evaluation.leakage import find_remaining_test_pairs
from repurposemap.evaluation.split import DiseaseSplit
from repurposemap.evaluation.indications import IndicationRecord
from repurposemap.evaluation.training_graph import build_training_graph

INDICATION = ("indication", "indication")


def _both_directions_graph():
    # A, B are drugs; D is the held-out test disease; E is a train disease, for contrast.
    types = {"A": "drug", "B": "drug", "D": "disease", "E": "disease"}
    edges = [
        (*INDICATION, "A", "D"),  # forward: drug -> disease
        (*INDICATION, "D", "A"),  # reverse: disease -> drug, same known treatment
        (*INDICATION, "B", "E"),  # an unrelated, train-side indication
        (*INDICATION, "E", "B"),
    ]
    return primekg_style_graph(types, edges)


def _split_holding_out_d():
    record = IndicationRecord(disease_id="D", drug_id="A", disease_name="D", drug_name="A")
    train_record = IndicationRecord(disease_id="E", drug_id="B", disease_name="E", drug_name="B")
    return DiseaseSplit(
        train_disease_ids=("E",),
        test_disease_ids=("D",),
        train_indications=(train_record,),
        test_indications=(record,),
    )


def test_raw_graph_has_the_indication_edge_in_both_directions():
    # Sanity check on the fixture itself: PrimeKG really does store both directions.
    graph = _both_directions_graph()
    assert graph.has_edge("A", "D")
    assert graph.has_edge("D", "A")


def test_find_remaining_test_pairs_catches_a_forward_only_leak():
    graph = _both_directions_graph()
    # Simulate a training graph that removed only the reverse edge, leaving the forward one.
    graph.remove_edge("D", "A", key=INDICATION)
    violations = find_remaining_test_pairs(graph, _split_holding_out_d())
    assert len(violations) == 1
    assert "drug -> disease" in violations[0]


def test_find_remaining_test_pairs_catches_a_reverse_only_leak():
    graph = _both_directions_graph()
    # Simulate a training graph that removed only the forward edge, leaving the reverse one.
    # This is exactly the case a forward-only check would miss.
    graph.remove_edge("A", "D", key=INDICATION)
    violations = find_remaining_test_pairs(graph, _split_holding_out_d())
    assert len(violations) == 1
    assert "disease -> drug" in violations[0]


def test_find_remaining_test_pairs_catches_both_directions_at_once():
    graph = _both_directions_graph()
    violations = find_remaining_test_pairs(graph, _split_holding_out_d())
    assert len(violations) == 2


def test_find_remaining_test_pairs_is_clean_on_the_real_training_graph():
    # The actual leakage-safe view, not a hand-edited graph: both directions must be gone.
    graph = _both_directions_graph()
    split = _split_holding_out_d()
    training_graph = build_training_graph(graph, split.test_disease_ids)

    assert not training_graph.has_edge("A", "D")
    assert not training_graph.has_edge("D", "A")
    assert find_remaining_test_pairs(training_graph, split) == []

    # The train-side pair (B, E) is untouched either direction.
    assert training_graph.has_edge("B", "E")
    assert training_graph.has_edge("E", "B")
