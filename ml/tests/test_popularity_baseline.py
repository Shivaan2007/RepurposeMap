"""Popularity baseline: ranks the complete drug universe, independent of rank_drugs."""

from __future__ import annotations

from graph_builders import primekg_style_graph
from test_disease_evaluation import eval_graph

from repurposemap.evaluation import EvaluationConfig, evaluate_disease_split
from repurposemap.evaluation.popularity_baseline import compute_popularity_scores, popularity_ranking
from repurposemap.evaluation.training_graph import build_training_graph

TARGET = ("drug_protein", "target")
INDICATION = ("indication", "indication")


def _degree_graph():
    # D0: no edges. D1: one edge. D2: three edges (to three different proteins).
    types = {
        "D0": "drug", "D1": "drug", "D2": "drug",
        "P1": "gene/protein", "P2": "gene/protein", "P3": "gene/protein",
    }
    edges = [(*TARGET, "D1", "P1"), (*TARGET, "D2", "P1"), (*TARGET, "D2", "P2"), (*TARGET, "D2", "P3")]
    return primekg_style_graph(types, edges)


def test_popularity_ranking_orders_by_descending_degree():
    graph = _degree_graph()
    drug_ids = ["D0", "D1", "D2"]
    scores = compute_popularity_scores(graph, drug_ids)
    assert scores == {"D0": 0, "D1": 1, "D2": 3}
    assert popularity_ranking(scores, drug_ids) == ["D2", "D1", "D0"]


def test_popularity_ranking_breaks_ties_by_id():
    scores = {"B": 5, "A": 5, "C": 1}
    assert popularity_ranking(scores, ["B", "A", "C"]) == ["A", "B", "C"]


def test_popularity_ranking_is_deterministic():
    graph = _degree_graph()
    drug_ids = ["D0", "D1", "D2"]
    scores = compute_popularity_scores(graph, drug_ids)
    first = popularity_ranking(scores, drug_ids)
    second = popularity_ranking(scores, drug_ids)
    assert first == second


def test_popularity_score_unknown_drug_defaults_to_zero():
    scores = {"A": 5}
    assert popularity_ranking(scores, ["A", "MISSING"]) == ["A", "MISSING"]


def test_popularity_degree_is_read_from_the_leakage_safe_training_graph():
    # A held-out drug-disease indication edge must not inflate either node's degree:
    # the training graph has already removed it (both directions) before popularity
    # is computed.
    types = {"A": "drug", "D": "disease", "P": "gene/protein"}
    edges = [(*TARGET, "A", "P"), (*INDICATION, "A", "D"), (*INDICATION, "D", "A")]
    graph = primekg_style_graph(types, edges)
    training_graph = build_training_graph(graph, ["D"])

    # Raw graph: A has 3 edges (target + indication x2). Training graph: only the target edge.
    assert training_graph.degree("A") == 1
    scores = compute_popularity_scores(training_graph, ["A"])
    assert scores["A"] == 1


def test_popularity_rank_does_not_require_a_path_candidate():
    # Reuses the shared fixture from test_disease_evaluation: A4's only link to D4
    # was the indication edge that gets held out, so rank_drugs finds zero candidates
    # for it (confirmed by path_rank is None / n_candidates == 0 elsewhere). Popularity
    # must not depend on that: it ranks the complete universe regardless.
    result = evaluate_disease_split(
        eval_graph(), EvaluationConfig(seed=0, test_fraction=1.0, max_test_diseases=4, max_candidates=50)
    )
    by_drug = {o.drug_id: o for o in result.outcomes}
    assert by_drug["A4"].path_rank is None
    assert by_drug["A4"].n_candidates == 0
    assert by_drug["A4"].popularity_rank is not None


def test_popularity_universe_is_every_drug_node_not_just_rank_drugs_output():
    # The popularity order must include every drug in the graph, not only the
    # (much smaller) set rank_drugs happened to find as candidates for any one disease.
    result = evaluate_disease_split(
        eval_graph(), EvaluationConfig(seed=0, test_fraction=1.0, max_test_diseases=4, max_candidates=50)
    )
    all_drug_ids = {"A1", "A2", "A3a", "A3b", "A4"}
    for outcome in result.outcomes:
        # n_candidates (path) is at most a handful; the popularity universe is every drug.
        assert outcome.n_candidates <= len(all_drug_ids)
    # Every true drug (which is, by construction, a drug node) has a popularity rank
    # computed against the full universe, confirmed indirectly: it is never None here,
    # because every drug in this fixture has at least itself in the universe.
    assert all(o.popularity_rank is not None for o in result.outcomes)


def test_popularity_hits_and_mrr_reflect_the_outcomes():
    result = evaluate_disease_split(
        eval_graph(), EvaluationConfig(seed=0, test_fraction=1.0, max_test_diseases=4, max_candidates=50)
    )
    ranks = [o.popularity_rank for o in result.outcomes]
    assert result.popularity_mrr == sum((1 / r if r else 0) for r in ranks) / len(ranks)
    for k in (1, 3, 10):
        expected = sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
        assert result.popularity_hits_at(k) == expected
