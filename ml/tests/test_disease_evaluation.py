"""Disease-based evaluation: indications, split, leakage, training graph, metrics, random baseline."""

import pytest

from graph_builders import primekg_style_graph
from repurposemap.evaluation import (
    DiseaseSplit,
    EvaluationConfig,
    IndicationRecord,
    build_training_graph,
    evaluate_disease_split,
    extract_indications,
    filtered_rank,
    find_disease_set_overlap,
    find_remaining_test_indications,
    find_remaining_test_pairs,
    hits_at_k,
    mean_reciprocal_rank,
    random_ranking,
    reciprocal_rank,
    split_by_disease,
    validate_split,
)
from repurposemap.evaluation.metrics import average_precision

TARGET = ("drug_protein", "target")
GENE_DISEASE = ("disease_protein", "associated with")
INDICATION = ("indication", "indication")
CONTRAINDICATION = ("contraindication", "contraindication")
DRUG_DRUG = ("drug_drug", "synergistic interaction")
PHENOTYPE_LINK = ("disease_phenotype_positive", "phenotype present")


# A small graph with a normal case (D1), a disease with two true drugs (D3), and a disease
# whose only link to the graph is the indication being held out (D4).
EVAL_TYPES = {
    "A1": "drug",
    "A2": "drug",
    "A3a": "drug",
    "A3b": "drug",
    "A4": "drug",
    "P1": "gene/protein",
    "P2": "gene/protein",
    "P3a": "gene/protein",
    "P3b": "gene/protein",
    "PH1": "effect/phenotype",
    "D1": "disease",
    "D2": "disease",
    "D3": "disease",
    "D4": "disease",
}
EVAL_EDGES = [
    (*TARGET, "A1", "P1"), (*GENE_DISEASE, "P1", "D1"), (*INDICATION, "A1", "D1"),
    (*TARGET, "A2", "P2"), (*GENE_DISEASE, "P2", "D2"), (*INDICATION, "A2", "D2"),
    (*TARGET, "A3a", "P3a"), (*GENE_DISEASE, "P3a", "D3"), (*INDICATION, "A3a", "D3"),
    (*TARGET, "A3b", "P3b"), (*GENE_DISEASE, "P3b", "D3"), (*INDICATION, "A3b", "D3"),
    (*INDICATION, "A4", "D4"),
    (*PHENOTYPE_LINK, "D4", "PH1"),  # non-treatment biology for D4, must survive the holdout
]


def eval_graph():
    return primekg_style_graph(EVAL_TYPES, EVAL_EDGES)


# Indication extraction

def test_extract_indications_only_uses_the_indication_relation():
    graph = primekg_style_graph(
        {"A": "drug", "D": "disease"},
        [(*INDICATION, "A", "D"), (*CONTRAINDICATION, "A", "D"), (*DRUG_DRUG, "A", "D")],
    )
    records = extract_indications(graph)
    assert records == (IndicationRecord(disease_id="D", drug_id="A", disease_name="D", drug_name="A"),)


def test_extract_indications_requires_drug_to_disease_direction():
    # The same relation stored disease -> drug (as PrimeKG also stores it) is not an indication record here.
    graph = primekg_style_graph({"A": "drug", "D": "disease"}, [(*INDICATION, "D", "A")])
    assert extract_indications(graph) == ()


def test_extract_indications_is_sorted_and_deduplicated():
    graph = primekg_style_graph(
        {"A": "drug", "B": "drug", "D": "disease"},
        [(*INDICATION, "B", "D"), (*INDICATION, "A", "D"), (*INDICATION, "A", "D")],
    )
    records = extract_indications(graph)
    assert [r.drug_id for r in records] == ["A", "B"]


def test_extract_indications_on_eval_graph_finds_all_five():
    assert len(extract_indications(eval_graph())) == 5


# Disease-based split

def _records(n_diseases: int) -> tuple[IndicationRecord, ...]:
    return tuple(
        IndicationRecord(disease_id=f"D{i:02d}", drug_id=f"A{i:02d}", disease_name=f"D{i:02d}", drug_name=f"A{i:02d}")
        for i in range(n_diseases)
    )


def test_split_is_deterministic_for_a_seed():
    records = _records(10)
    first = split_by_disease(records, seed=1, test_fraction=0.3)
    second = split_by_disease(records, seed=1, test_fraction=0.3)
    assert first.test_disease_ids == second.test_disease_ids
    assert first.train_disease_ids == second.train_disease_ids


def test_split_uses_test_fraction_of_disease_count():
    split = split_by_disease(_records(10), seed=0, test_fraction=0.3)
    assert len(split.test_disease_ids) == 3
    assert len(split.train_disease_ids) == 7


def test_split_respects_max_test_diseases_cap():
    split = split_by_disease(_records(10), seed=0, test_fraction=0.9, max_test_diseases=2)
    assert len(split.test_disease_ids) == 2


def test_split_holds_out_every_indication_of_a_test_disease():
    # D3 has two true drugs. If D3 is a test disease, both indications must be in the test set.
    split = split_by_disease(extract_indications(eval_graph()), seed=0, test_fraction=1.0, max_test_diseases=4)
    assert "D3" in split.test_disease_ids
    test_drugs_for_d3 = {r.drug_id for r in split.test_indications if r.disease_id == "D3"}
    assert test_drugs_for_d3 == {"A3a", "A3b"}
    assert all(r.disease_id != "D3" for r in split.train_indications)


def test_train_and_test_disease_sets_are_disjoint():
    split = split_by_disease(_records(10), seed=2, test_fraction=0.5)
    assert set(split.train_disease_ids).isdisjoint(split.test_disease_ids)


def test_validate_split_rejects_an_overlapping_split():
    bad = DiseaseSplit(
        train_disease_ids=("D1", "D2"),
        test_disease_ids=("D2", "D3"),
        train_indications=(),
        test_indications=(),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_split(bad)


def test_validate_split_rejects_a_misplaced_indication():
    bad = DiseaseSplit(
        train_disease_ids=("D1",),
        test_disease_ids=("D2",),
        train_indications=(IndicationRecord(disease_id="D2", drug_id="A"),),  # belongs to test, not train
        test_indications=(),
    )
    with pytest.raises(ValueError, match="not a train disease"):
        validate_split(bad)


def test_split_rejects_empty_records():
    with pytest.raises(ValueError):
        split_by_disease((), seed=0, test_fraction=0.1)


def test_evaluation_config_rejects_bad_values():
    with pytest.raises(ValueError):
        EvaluationConfig(test_fraction=0)
    with pytest.raises(ValueError):
        EvaluationConfig(max_test_diseases=0)
    with pytest.raises(ValueError):
        EvaluationConfig(top_ks=())


def test_evaluation_config_to_dict_has_no_hidden_fields():
    config = EvaluationConfig()
    assert set(config.to_dict()) == {
        "seed", "test_fraction", "max_test_diseases", "top_ks", "max_path_length",
        "paths_per_drug", "max_candidates", "time_limit_per_drug_s", "max_edge_checks_per_drug",
        "max_backward_edge_checks",
    }


# Training graph and leakage

def test_training_graph_removes_test_disease_indications_both_directions():
    graph = eval_graph()
    training = build_training_graph(graph, ["D1"])
    assert not training.has_edge("A1", "D1")
    # PrimeKG style edges are stored in both directions; neither should survive.
    graph.add_edge("D1", "A1", key=INDICATION, relation="indication", display_relation="indication")
    training = build_training_graph(graph, ["D1"])
    assert not training.has_edge("D1", "A1")


def test_training_graph_keeps_non_treatment_biology_for_test_diseases():
    training = build_training_graph(eval_graph(), ["D4"])
    assert training.has_edge("D4", "PH1")


def test_training_graph_keeps_train_disease_indications():
    training = build_training_graph(eval_graph(), ["D1"])
    assert training.has_edge("A2", "D2")  # D2 is not held out


def test_leakage_checks_pass_on_a_correctly_built_training_graph():
    graph = eval_graph()
    split = split_by_disease(extract_indications(graph), seed=0, test_fraction=1.0, max_test_diseases=4)
    training = build_training_graph(graph, split.test_disease_ids)
    assert find_remaining_test_indications(training, split) == []
    assert find_remaining_test_pairs(training, split) == []
    assert find_disease_set_overlap(split) == []


def test_leakage_check_fails_loudly_if_indications_were_not_removed():
    from repurposemap.evaluation.leakage import assert_no_leakage

    graph = eval_graph()
    split = split_by_disease(extract_indications(graph), seed=0, test_fraction=1.0, max_test_diseases=4)
    with pytest.raises(ValueError, match="leakage detected"):
        assert_no_leakage(graph, split)  # the raw graph, not the training graph: indications still present


# Metrics

@pytest.mark.parametrize(
    ("rank", "k", "expected"),
    [(1, 1, True), (1, 3, True), (3, 3, True), (4, 3, False), (None, 10, False)],
)
def test_hits_at_k_single_rank(rank, k, expected):
    assert hits_at_k([rank], k) == (1.0 if expected else 0.0)


def test_hits_at_k_averages_over_several_ranks():
    assert hits_at_k([1, 5, None], 3) == pytest.approx(1 / 3)


def test_hits_at_k_of_empty_is_zero():
    assert hits_at_k([], 10) == 0.0


def test_reciprocal_rank_values():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1, 4, None]) == pytest.approx((1 + 0.25 + 0) / 3)
    assert mean_reciprocal_rank([]) == 0.0


def test_filtered_rank_removes_other_true_ids_before_ranking():
    ranked = ["decoy", "true_b", "true_a", "other"]
    # Without filtering, true_a would be rank 3. With true_b filtered out, it is rank 2.
    assert filtered_rank("true_a", ["true_b"], ranked) == 2
    assert filtered_rank("true_b", ["true_a"], ranked) == 2


def test_filtered_rank_none_when_absent():
    assert filtered_rank("missing", [], ["a", "b"]) is None


def test_average_precision_is_an_unimplemented_placeholder():
    with pytest.raises(NotImplementedError):
        average_precision(1, 1)


# Random baseline

def test_random_ranking_is_reproducible_for_the_same_seed_and_disease():
    candidates = ["A", "B", "C", "D"]
    first = random_ranking(candidates, seed=7, disease_id="D1")
    second = random_ranking(candidates, seed=7, disease_id="D1")
    assert first == second
    assert sorted(first) == sorted(candidates)


def test_random_ranking_differs_for_a_different_disease():
    candidates = [f"drug{i}" for i in range(20)]
    a = random_ranking(candidates, seed=7, disease_id="D1")
    b = random_ranking(candidates, seed=7, disease_id="D2")
    assert a != b


# End-to-end disease evaluation

def test_evaluate_disease_split_end_to_end():
    graph = eval_graph()
    config = EvaluationConfig(seed=0, test_fraction=1.0, max_test_diseases=4, max_candidates=50)
    result = evaluate_disease_split(graph, config)

    assert result.n_test_diseases == 4
    assert result.n_test_indications == 5  # A1-D1, A2-D2, A3a-D3, A3b-D3, A4-D4
    assert result.runtime_s >= 0

    by_drug = {o.drug_id: o for o in result.outcomes}
    assert by_drug["A1"].path_rank == 1
    assert by_drug["A2"].path_rank == 1
    # D3 has two true drugs; filtering means each still ranks first among the others.
    assert by_drug["A3a"].path_rank == 1
    assert by_drug["A3b"].path_rank == 1
    # A4's only link to D4 was the indication edge that got held out, so it cannot be ranked.
    assert by_drug["A4"].path_rank is None
    assert by_drug["A4"].random_rank is None
    assert by_drug["A4"].n_candidates == 0

    # This graph is tiny, so nothing should hit a search budget.
    for outcome in result.outcomes:
        assert outcome.n_truncated_searches == 0
        assert outcome.candidate_cap_hit is False
        assert outcome.backward_search_truncated is False
    assert result.n_diseases_with_candidate_cap_hit == 0
    assert result.n_diseases_with_truncated_searches == 0
    assert result.n_diseases_with_backward_search_truncated == 0


def test_evaluate_disease_split_exposes_backward_search_truncation():
    # One disease with 30 independent drug -> protein -> disease chains, same pattern as
    # test_drug_ranking.test_backward_search_budget_bounds_a_densely_connected_disease. A
    # tiny backward-search budget must show up on the outcome, not just internally.
    n = 30
    types = {
        "D": "disease",
        **{f"A{i:02d}": "drug" for i in range(n)},
        **{f"P{i:02d}": "gene/protein" for i in range(n)},
    }
    edges = [(*INDICATION, "A00", "D")]
    for i in range(n):
        edges.append((*TARGET, f"A{i:02d}", f"P{i:02d}"))
        edges.append((*GENE_DISEASE, f"P{i:02d}", "D"))
    graph = primekg_style_graph(types, edges)

    config = EvaluationConfig(
        seed=0, test_fraction=1.0, max_test_diseases=1, max_candidates=50, max_backward_edge_checks=10
    )
    result = evaluate_disease_split(graph, config)

    outcome = result.outcomes[0]
    assert outcome.backward_search_truncated is True
    assert outcome.n_candidates == 0  # the budget ran out before any drug was reached
    assert result.n_diseases_with_backward_search_truncated == 1


def test_evaluate_disease_split_metrics_reflect_the_outcomes():
    result = evaluate_disease_split(
        eval_graph(), EvaluationConfig(seed=0, test_fraction=1.0, max_test_diseases=4, max_candidates=50)
    )
    # 4 of 5 outcomes rank 1st; one is unranked.
    assert result.hits_at(1) == pytest.approx(4 / 5)
    assert result.hits_at(10) == pytest.approx(4 / 5)
    assert result.mrr == pytest.approx(4 / 5)


def test_evaluate_disease_split_is_deterministic():
    graph = eval_graph()
    config = EvaluationConfig(seed=3, test_fraction=1.0, max_test_diseases=4, max_candidates=50)
    first = evaluate_disease_split(graph, config)
    second = evaluate_disease_split(graph, config)
    assert [(o.drug_id, o.path_rank, o.random_rank) for o in first.outcomes] == [
        (o.drug_id, o.path_rank, o.random_rank) for o in second.outcomes
    ]


def test_evaluate_disease_split_rejects_a_graph_without_indications():
    graph = primekg_style_graph({"A": "drug", "D": "disease"}, [(*TARGET, "A", "D")])
    with pytest.raises(ValueError):
        evaluate_disease_split(graph, EvaluationConfig())
