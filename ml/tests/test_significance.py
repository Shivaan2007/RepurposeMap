"""repurposemap.evaluation.significance -- the disease-level paired
bootstrap method frozen in docs/FINAL_RESULTS.md section 3. Pure numpy
logic, no torch, no real model, no real locked-test numbers below."""

from __future__ import annotations

import pytest

from repurposemap.evaluation.significance import paired_bootstrap_ci


def _outcomes(ranks_by_disease: dict[str, list[int | None]]) -> list[dict]:
    return [
        {"disease_id": disease_id, "drug_id": f"drug-{i}", "rank": rank}
        for disease_id, ranks in ranks_by_disease.items()
        for i, rank in enumerate(ranks)
    ]


def test_identical_outcomes_give_zero_diff_and_include_zero():
    outcomes = _outcomes({"d1": [1, 5], "d2": [3], "d3": [2, 2]})
    result = paired_bootstrap_ci(outcomes, outcomes, metric="mrr", n_resamples=500, seed=0)

    assert result.point_estimate_diff == 0.0
    assert result.ci_low <= 0.0 <= result.ci_high
    assert not result.excludes_zero


def test_a_strictly_better_than_b_excludes_zero_with_enough_diseases_and_resamples():
    # 30 diseases, A always rank 1 (perfect), B always rank 1000 (near-last) --
    # a large, consistent gap should not straddle zero even at 95%.
    a = _outcomes({f"d{i}": [1] for i in range(30)})
    b = _outcomes({f"d{i}": [1000] for i in range(30)})

    result = paired_bootstrap_ci(a, b, metric="mrr", n_resamples=2000, seed=42)

    assert result.point_estimate_diff > 0
    assert result.excludes_zero
    assert result.ci_low > 0


def test_hits_at_10_metric_selection():
    a = _outcomes({"d1": [5], "d2": [5]})  # within top 10
    b = _outcomes({"d1": [50], "d2": [50]})  # outside top 10

    result = paired_bootstrap_ci(a, b, metric="hits_at_10", n_resamples=200, seed=1)

    assert result.point_estimate_a == 1.0
    assert result.point_estimate_b == 0.0
    assert result.metric == "hits_at_10"


def test_unsupported_metric_raises():
    outcomes = _outcomes({"d1": [1]})
    with pytest.raises(ValueError, match="unsupported metric"):
        paired_bootstrap_ci(outcomes, outcomes, metric="not_a_metric", n_resamples=10, seed=0)


def test_mismatched_disease_sets_raises():
    a = _outcomes({"d1": [1], "d2": [2]})
    b = _outcomes({"d1": [1], "d3": [2]})
    with pytest.raises(ValueError, match="different disease sets"):
        paired_bootstrap_ci(a, b, metric="mrr", n_resamples=10, seed=0)


def test_is_deterministic_given_the_same_seed():
    a = _outcomes({"d1": [2, 4], "d2": [7], "d3": [1, 1, 9]})
    b = _outcomes({"d1": [3, 5], "d2": [6], "d3": [2, 2, 8]})

    first = paired_bootstrap_ci(a, b, metric="mrr", n_resamples=500, seed=7)
    second = paired_bootstrap_ci(a, b, metric="mrr", n_resamples=500, seed=7)

    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_n_diseases_matches_the_unique_disease_count():
    outcomes = _outcomes({"d1": [1], "d2": [2], "d3": [3]})
    result = paired_bootstrap_ci(outcomes, outcomes, metric="mrr", n_resamples=50, seed=0)
    assert result.n_diseases == 3
