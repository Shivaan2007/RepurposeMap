"""Drug-ranking baseline and evaluation harness, on tiny PrimeKG-style graphs."""

import pytest

from graph_builders import primekg_style_graph
from repurposemap.evaluation import (
    HeldOutPair,
    evaluate_baseline,
    indication_pairs,
    rank_of,
    select_pairs,
)
from repurposemap.evaluation.baseline import PairOutcome
from repurposemap.ranking import rank_drugs
from repurposemap.graph import PathStep
from repurposemap.scoring import (
    PRIMEKG_RELATION_POLICY,
    DrugTargetCounts,
    NeighbourCounts,
    RelationCategory,
    RelationPolicy,
    drug_promiscuity_penalty,
    score_path,
)
from repurposemap.scoring.promiscuity import PROMISCUOUS_TARGET_THRESHOLD, VERY_PROMISCUOUS_TARGET_THRESHOLD

TARGET = ("drug_protein", "target")
GENE_DISEASE = ("disease_protein", "associated with")
INDICATION = ("indication", "indication")
DRUG_DRUG = ("drug_drug", "synergistic interaction")

TYPES = {
    "A": "drug",
    "B": "drug",
    "I": "drug",
    "X": "drug",
    "P1": "gene/protein",
    "P2": "gene/protein",
    "P3": "gene/protein",
    "D": "disease",
    "E": "disease",
}


def build_graph(extra_edges=()):
    edges = [
        (*TARGET, "A", "P1"),
        (*TARGET, "A", "P2"),
        (*TARGET, "B", "P1"),
        (*TARGET, "I", "P3"),
        (*GENE_DISEASE, "P1", "D"),
        (*GENE_DISEASE, "P2", "D"),
        (*GENE_DISEASE, "P3", "D"),
        (*INDICATION, "I", "D"),
        (*INDICATION, "A", "E"),
        (*DRUG_DRUG, "X", "A"),
        *extra_edges,
    ]
    return primekg_style_graph(TYPES, edges)


def rank(graph, disease="D", **options):
    policy = options.pop("policy", RelationPolicy())
    return rank_drugs(
        graph, disease, neighbour_counts=NeighbourCounts(graph, policy), policy=policy, **options
    )


# Ranking

def test_drug_with_more_supporting_paths_ranks_higher():
    ranking = rank(build_graph())
    names = [candidate.drug_id for candidate in ranking.ranked]
    assert names[0] == "A"


def test_score_is_sum_of_top_three_path_totals():
    graph = build_graph()
    ranking = rank(graph)
    drug_a = next(c for c in ranking.ranked if c.drug_id == "A")
    # Drug A has two valid paths, A-P1-D and A-P2-D. Each is preferred, preferred, length 2.
    two_hop_path = [
        PathStep(
            source="A", source_type="drug", relation="drug_protein",
            target="P1", target_type="gene/protein", display_relation="target",
        ),
        PathStep(
            source="P1", source_type="gene/protein", relation="disease_protein",
            target="D", target_type="disease", display_relation="associated with",
        ),
    ]
    expected_each = score_path(two_hop_path, RelationPolicy(), {}).total_score
    assert drug_a.n_qualifying_paths == 2
    assert drug_a.score == pytest.approx(2 * expected_each)


def test_ranking_is_deterministic():
    graph = build_graph()
    first = [(c.drug_id, c.score) for c in rank(graph).ranked]
    second = [(c.drug_id, c.score) for c in rank(graph).ranked]
    assert first == second


def test_best_path_is_the_explanation_and_is_valid():
    ranking = rank(build_graph())
    for candidate in ranking.ranked:
        assert candidate.best_path.score.valid
        assert candidate.best_path.steps[0].source == candidate.drug_id


def test_drug_connected_only_by_drug_drug_is_not_a_candidate():
    ranking = rank(build_graph())
    assert "X" not in [c.drug_id for c in ranking.ranked]


def test_disease_with_only_an_excluded_link_has_no_candidates():
    # E is linked only by A's indication, which the policy excludes.
    ranking = rank(build_graph(), disease="E")
    assert ranking.ranked == ()
    assert ranking.n_candidates == 0


def test_isolated_disease_gives_no_candidates():
    graph = primekg_style_graph({"D": "disease", "Q": "disease"}, [])
    ranking = rank(graph, disease="Q")
    assert ranking.ranked == ()
    assert ranking.n_candidates == 0


def test_non_disease_node_is_rejected():
    with pytest.raises(ValueError, match="not a disease node"):
        rank(build_graph(), disease="A")


def test_unknown_disease_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        rank(build_graph(), disease="NOPE")


def test_candidate_cap_keeps_nearest_drugs_and_reports_it():
    ranking = rank(build_graph(), max_candidates=1)
    assert ranking.candidate_cap_hit
    assert ranking.n_candidates == 1
    assert [c.drug_id for c in ranking.ranked] == ["A"]


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        rank(build_graph(), max_path_length=0)
    with pytest.raises(ValueError):
        rank(build_graph(), paths_per_drug=0)


# Direct-edge leakage

def test_direct_indication_edge_is_never_used_as_evidence():
    ranking = rank(build_graph())
    for candidate in ranking.ranked:
        for step in candidate.best_path.steps:
            assert step.relation != "indication"


def test_hidden_pair_removes_a_direct_edge_that_the_policy_allows():
    # Make indication count as acceptable, so only hidden_pairs can stop the direct hop.
    policy = RelationPolicy(
        {**PRIMEKG_RELATION_POLICY, ("indication", "indication"): RelationCategory.ACCEPTABLE}
    )
    graph = build_graph()
    visible = rank(graph, policy=policy)
    hidden = rank(graph, policy=policy, hidden_pairs=[("I", "D"), ("D", "I")])
    best_visible = next(c for c in visible.ranked if c.drug_id == "I").best_path
    best_hidden = next(c for c in hidden.ranked if c.drug_id == "I").best_path
    assert len(best_visible.steps) == 1
    assert len(best_hidden.steps) == 2


# Evaluation

def test_held_out_drug_rank_and_metrics():
    graph = build_graph()
    counts = NeighbourCounts(graph, RelationPolicy())
    pair = HeldOutPair("I", "I", "D", "D")
    result = evaluate_baseline(graph, [pair], neighbour_counts=counts, seed=7)
    outcome = result.outcomes[0]
    # With I's indication hidden, the drugs ranked for D are A (score 3.0) and then B and I (1.5 each, so
    # ordered by name). I ranks third.
    assert outcome.rank == 3
    assert outcome.hit_at_k
    assert outcome.reciprocal_rank == pytest.approx(1 / 3)
    assert result.hits_at_10 == pytest.approx(1.0)
    assert result.mean_reciprocal_rank == pytest.approx(1 / 3)
    assert result.seed == 7


def test_unranked_held_out_drug_scores_zero():
    graph = build_graph()
    counts = NeighbourCounts(graph, RelationPolicy())
    pair = HeldOutPair("X", "X", "D", "D")
    result = evaluate_baseline(graph, [pair], neighbour_counts=counts)
    assert result.outcomes[0].rank is None
    assert not result.outcomes[0].hit_at_k
    assert result.mean_reciprocal_rank == 0.0


def test_rank_of_is_one_based_and_none_when_absent():
    ranking = rank(build_graph())
    assert rank_of("A", ranking) == 1
    assert rank_of("NOT_THERE", ranking) is None


def test_outcome_metrics_for_known_ranks():
    pair = HeldOutPair("a", "a", "d", "d")
    assert PairOutcome(pair, rank=1, n_ranked=5).reciprocal_rank == 1.0
    assert PairOutcome(pair, rank=10, n_ranked=20).hit_at_k
    assert not PairOutcome(pair, rank=11, n_ranked=20).hit_at_k
    assert PairOutcome(pair, rank=None, n_ranked=20).reciprocal_rank == 0.0


def test_indication_pairs_are_drug_to_disease_only():
    assert indication_pairs(build_graph()) == [("A", "E"), ("I", "D")]


def test_select_pairs_is_reproducible_for_a_seed():
    graph = build_graph()
    assert select_pairs(graph, 1, seed=3) == select_pairs(graph, 1, seed=3)


def test_select_pairs_returns_all_when_asked_for_more():
    assert len(select_pairs(build_graph(), 50, seed=0)) == 2


def test_select_pairs_rejects_a_graph_without_indications():
    graph = primekg_style_graph({"A": "drug", "D": "disease"}, [(*TARGET, "A", "D")])
    with pytest.raises(ValueError):
        select_pairs(graph, 1, seed=0)


# Drug-side promiscuity penalty



@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (1, 0.0),
        (PROMISCUOUS_TARGET_THRESHOLD, 0.0),
        (PROMISCUOUS_TARGET_THRESHOLD + 1, 0.25),
        (VERY_PROMISCUOUS_TARGET_THRESHOLD, 0.25),
        (VERY_PROMISCUOUS_TARGET_THRESHOLD + 1, 0.75),
    ],
)
def test_promiscuity_penalty_tiers(targets, expected):
    assert drug_promiscuity_penalty(targets) == pytest.approx(expected)


def test_target_counts_use_only_distinct_target_proteins_of_drugs():
    graph = primekg_style_graph(
        {"DR": "drug", "P1": "gene/protein", "P2": "gene/protein", "P3": "gene/protein", "D": "disease"},
        [
            (*TARGET, "DR", "P1"),
            (*TARGET, "DR", "P2"),
            ("drug_protein", "enzyme", "DR", "P3"),  # enzyme role, not a target
            (*TARGET, "P1", "D"),                      # not from a drug
        ],
    )
    counts = DrugTargetCounts(graph)
    assert counts["DR"] == 2
    assert counts["P1"] == 0
    assert counts["D"] == 0


def test_promiscuous_drug_score_is_path_sum_minus_penalty():
    # Drug H has 30 targets, each linked to a disease gene, so its path sum is large.
    proteins = [f"G{i:02d}" for i in range(30)]
    types = {"H": "drug", "D": "disease", **{p: "gene/protein" for p in proteins}}
    edges = []
    for protein in proteins:
        edges.append((*TARGET, "H", protein))
        edges.append((*GENE_DISEASE, protein, "D"))
    graph = primekg_style_graph(types, edges)

    ranking = rank(graph)
    candidate = next(c for c in ranking.ranked if c.drug_id == "H")
    assert candidate.n_targets == 30
    assert candidate.promiscuity_penalty == pytest.approx(0.25)
    assert candidate.path_sum == pytest.approx(3 * 1.5)
    assert candidate.score == pytest.approx(candidate.path_sum - 0.25)


def test_promiscuity_penalty_can_reorder_drugs():
    # Two drugs with identical path sums. The one with 30 extra targets must fall behind.
    types = {"CLEAN": "drug", "NOISY": "drug", "D": "disease", "G": "gene/protein", "X": "gene/protein"}
    edges = [(*TARGET, "CLEAN", "G"), (*GENE_DISEASE, "G", "D"), (*TARGET, "NOISY", "G")]
    for index in range(30):
        protein = f"N{index:02d}"
        types[protein] = "gene/protein"
        edges.append((*TARGET, "NOISY", protein))
    graph = primekg_style_graph(types, edges)

    ranking = rank(graph)
    assert [c.drug_id for c in ranking.ranked][0] == "CLEAN"


# Backward-search edge-check budget (bounds a densely connected disease)

def test_backward_search_budget_bounds_a_densely_connected_disease():
    # 30 independent drug -> protein -> disease chains. The drugs sit two hops back,
    # so a budget that runs out while still examining the disease's direct
    # predecessors (the proteins) must find zero candidates.
    n = 30
    types = {"D": "disease", **{f"A{i:02d}": "drug" for i in range(n)}, **{f"P{i:02d}": "gene/protein" for i in range(n)}}
    edges = []
    for i in range(n):
        edges.append((*TARGET, f"A{i:02d}", f"P{i:02d}"))
        edges.append((*GENE_DISEASE, f"P{i:02d}", "D"))
    graph = primekg_style_graph(types, edges)

    starved = rank(graph, max_backward_edge_checks=10)
    assert starved.backward_search_truncated
    assert starved.n_candidates == 0

    full = rank(graph, max_backward_edge_checks=100_000)
    assert not full.backward_search_truncated
    assert full.n_candidates == n


def test_backward_search_budget_is_rejected_below_one():
    with pytest.raises(ValueError, match="max_backward_edge_checks"):
        rank(build_graph(), max_backward_edge_checks=0)
