"""Synthetic-graph tests for the masked case-study harness. Never loads a
real graph or locked-test data -- every graph here is built by hand in the
test itself."""

from __future__ import annotations

import networkx as nx
import pytest

from repurposemap.evaluation.case_study import (
    CaseStudySpec,
    MaskingNotConfirmedError,
    build_case_study_result,
    confirm_masked,
    mask_known_indication,
)


def _synthetic_graph() -> nx.MultiDiGraph:
    """A tiny synthetic graph with a known indication edge in both
    directions, plus an unrelated gene-link path and a second, unrelated
    indication pair -- large enough to prove masking is targeted, not a
    blanket wipe."""
    graph = nx.MultiDiGraph()
    graph.add_node("drug:A", node_type="drug", name="Demo Drug A")
    graph.add_node("disease:X", node_type="disease", name="Demo Disease X")
    graph.add_node("gene:G1", node_type="gene/protein", name="Demo Gene G1")
    graph.add_node("drug:B", node_type="drug", name="Demo Drug B")
    graph.add_node("disease:Y", node_type="disease", name="Demo Disease Y")

    graph.add_edge("drug:A", "disease:X", relation="indication")
    graph.add_edge("disease:X", "drug:A", relation="indication")
    graph.add_edge("drug:A", "gene:G1", relation="target")
    graph.add_edge("gene:G1", "disease:X", relation="associated_with")
    graph.add_edge("drug:B", "disease:Y", relation="indication")
    graph.add_edge("disease:Y", "drug:B", relation="indication")
    return graph


def _spec() -> CaseStudySpec:
    return CaseStudySpec(
        disease_id="disease:X",
        disease_name="Demo Disease X",
        known_drug_id="drug:A",
        known_drug_name="Demo Drug A",
        ranking_method="demo_ranker",
    )


def test_known_indication_edge_is_removed_in_both_directions():
    graph = _synthetic_graph()
    spec = _spec()
    assert graph.has_edge("drug:A", "disease:X")
    assert graph.has_edge("disease:X", "drug:A")

    masked = mask_known_indication(graph, spec)

    assert not masked.has_edge("drug:A", "disease:X")
    assert not masked.has_edge("disease:X", "drug:A")
    assert confirm_masked(masked, spec) is True


def test_masking_does_not_touch_unrelated_edges_or_mutate_the_input():
    graph = _synthetic_graph()
    spec = _spec()

    masked = mask_known_indication(graph, spec)

    # unrelated indication pair and the gene path survive untouched
    assert masked.has_edge("drug:B", "disease:Y")
    assert masked.has_edge("disease:Y", "drug:B")
    assert masked.has_edge("drug:A", "gene:G1")
    assert masked.has_edge("gene:G1", "disease:X")
    # the original graph passed in is never mutated
    assert graph.has_edge("drug:A", "disease:X")
    assert graph.has_edge("disease:X", "drug:A")


def test_masked_edge_cannot_leak_back_via_a_parallel_edge():
    """A second, unmasked parallel ``indication`` edge between the same
    two nodes (e.g. added after masking, or missed by a bug) must still
    be caught by confirm_masked -- masking is not just "did we call the
    masking function once", it is "is the edge actually gone"."""
    graph = _synthetic_graph()
    spec = _spec()
    masked = mask_known_indication(graph, spec)

    # Simulate a leak: a parallel indication edge slips back in.
    masked.add_edge("drug:A", "disease:X", relation="indication")

    assert confirm_masked(masked, spec) is False
    with pytest.raises(MaskingNotConfirmedError):
        build_case_study_result(masked, spec, rank=3)


def test_result_cannot_be_built_without_masking_confirmation():
    graph = _synthetic_graph()
    spec = _spec()

    with pytest.raises(MaskingNotConfirmedError):
        build_case_study_result(graph, spec, rank=1)


def test_result_builds_cleanly_once_masking_is_confirmed():
    graph = _synthetic_graph()
    spec = _spec()
    masked = mask_known_indication(graph, spec)

    result = build_case_study_result(
        masked,
        spec,
        rank=7,
        limitations=("small synthetic graph, not a real evaluation",),
    )

    assert result.spec is spec
    assert result.rank == 7
    assert result.limitations == ("small synthetic graph, not a real evaluation",)
    assert result.explanation_paths == ()
    assert result.provenance == {}


def test_confirm_masked_is_false_when_graph_was_never_masked():
    graph = _synthetic_graph()
    spec = _spec()
    assert confirm_masked(graph, spec) is False
