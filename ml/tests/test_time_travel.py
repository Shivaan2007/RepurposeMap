"""The time-travel test's rules (ml/TIME_TRAVEL_PROTOCOL.md). Pure logic on invented numbers."""

from __future__ import annotations

import pytest

from repurposemap.evaluation import time_travel as tt


@pytest.mark.parametrize(
    "date, prior",
    [
        ("2022-04-25", True), ("2022-04-26", False), ("2022-04", True), ("2022-05", False), ("2022-03-31", True),
        ("2022", True), ("2021-12-31", True), ("2023-01-01", False), ("2023", False), ("1999-06", True),
        (None, True), ("", True), ("not a date", True), (" 2022-05-01 ", False),
    ],
)
def test_trial_start_dates_follow_the_protocol(date, prior):
    assert tt.trial_is_prior(date) is prior


def test_classification_follows_the_definitions():
    assert tt.classify(tt.PairEvidence(prior_papers=1)) == "prior"
    assert tt.classify(tt.PairEvidence(prior_trials=1, later_papers=9)) == "prior"      # prior evidence wins over later
    assert tt.classify(tt.PairEvidence(later_papers=1)) == "new_hit"
    assert tt.classify(tt.PairEvidence(later_trials=2)) == "new_hit"
    assert tt.classify(tt.PairEvidence()) == "new_miss"
    assert tt.classify(tt.PairEvidence(prior_papers=5, error="timeout")) == "error"


def test_counts_separate_papers_trials_and_errors():
    evidences = [
        tt.PairEvidence(prior_papers=3), tt.PairEvidence(later_papers=2), tt.PairEvidence(later_trials=1),
        tt.PairEvidence(later_papers=1, later_trials=1), tt.PairEvidence(), tt.PairEvidence(error="x"),
    ]
    assert tt.counts(evidences) == {"pairs": 6, "prior": 1, "new": 4, "hits": 3, "paper_hits": 2, "trial_hits": 2, "errors": 1}


def test_pooled_rate_is_hits_over_new_pairs_and_none_without_any():
    assert tt.pooled_rate([{"new": 4, "hits": 1}, {"new": 6, "hits": 2}]) == pytest.approx(0.3)
    assert tt.pooled_rate([{"new": 0, "hits": 0}]) is None


def test_the_disease_sample_is_deterministic_sorted_and_without_replacement():
    ids = [str(i) for i in range(1000, 1100)]
    first = tt.sample_diseases(ids)
    assert first == tt.sample_diseases(list(reversed(ids)))       # input order does not matter
    assert len(first) == tt.N_DISEASES == len(set(first)) and first == sorted(first) and set(first) <= set(ids)
    assert first != tt.sample_diseases(ids, seed=7)
    with pytest.raises(ValueError):
        tt.sample_diseases(ids[:10])


def test_random_drugs_are_reproducible_distinct_and_depend_on_the_disease():
    drugs = [f"D{i}" for i in range(500)]
    a = tt.random_drugs(drugs, "123")
    assert a == tt.random_drugs(drugs, "123") and len(a) == len(set(a)) == tt.TOP_K
    assert a != tt.random_drugs(drugs, "124")


def test_popularity_ranks_by_degree_with_ties_broken_by_id():
    assert tt.popularity_top({"b": 5, "a": 5, "c": 9, "d": 1}, k=3) == ["c", "a", "b"]


def test_the_bootstrap_is_deterministic_and_finds_a_real_difference():
    a = [{"new": 10, "hits": 8} for _ in range(30)]
    b = [{"new": 10, "hits": 1} for _ in range(30)]
    result = tt.paired_bootstrap_difference(a, b, n_resamples=500)
    assert result == tt.paired_bootstrap_difference(a, b, n_resamples=500)
    assert result["difference"] == pytest.approx(0.7) and result["excludes_zero"] is True and result["ci_low"] > 0


def test_identical_rankers_show_no_difference():
    a = [{"new": 10, "hits": h} for h in (0, 1, 2, 3, 1, 0, 2, 4, 1, 2)]
    result = tt.paired_bootstrap_difference(a, list(a), n_resamples=500)
    assert result["difference"] == 0 and result["excludes_zero"] is False
    assert result["ci_low"] == result["ci_high"] == 0


def test_resamples_with_no_new_pairs_are_dropped_and_counted():
    a = [{"new": 0, "hits": 0}, {"new": 10, "hits": 5}]
    b = [{"new": 10, "hits": 1}, {"new": 10, "hits": 1}]
    result = tt.paired_bootstrap_difference(a, b, n_resamples=400)
    assert result["resamples_dropped"] > 0 and result["resamples_used"] + result["resamples_dropped"] == 400
    empty = tt.paired_bootstrap_difference([{"new": 0, "hits": 0}], [{"new": 0, "hits": 0}], n_resamples=50)
    assert empty["excludes_zero"] is None and empty["resamples_used"] == 0


def test_the_bootstrap_needs_the_same_diseases_for_both_rankers():
    with pytest.raises(ValueError):
        tt.paired_bootstrap_difference([{"new": 1, "hits": 0}], [], n_resamples=10)
