"""Disease-level Average Precision (AP), and macro/micro aggregation (Stage C, corrected).

Co-positives are never filtered out here, unlike Hits@K/MRR's filtered-rank
convention -- see the module docstring in
``repurposemap.evaluation.metrics``.

This is true (non-interpolated) Average Precision, NOT trapezoidal PR-AUC --
an earlier version of ``average_precision`` computed the latter under the
same name; these tests check the corrected AP formula exactly:

    AP = (1 / n_positives) * sum(precision_at_rank for every positive rank)
"""

from __future__ import annotations

import numpy as np
import pytest

from repurposemap.evaluation.metrics import average_precision, macro_auprc, micro_auprc


def test_average_precision_is_one_when_every_positive_is_ranked_first():
    scores = [0.9, 0.8, 0.1, 0.05]
    labels = [1, 1, 0, 0]
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_raises_with_no_positives():
    # AP is mathematically undefined with zero positives -- never silently
    # returned as 0.0.
    with pytest.raises(ValueError, match="no positives"):
        average_precision([0.5, 0.2], [0, 0])


def test_average_precision_matches_a_hand_worked_example():
    # 1 positive among 4 candidates, ranked 2nd by score.
    # Sorted by score desc: labels = [0, 1, 0, 0]
    # ranks:      1    2    3    4
    # precision:  0/1  1/2  1/3  1/4
    # AP = mean of precision AT POSITIVE RANKS ONLY = precision at rank 2 = 0.5
    # (divided by n_positives = 1), i.e. AP = 0.5 / 1 = 0.5.
    scores = [0.9, 0.8, 0.5, 0.1]
    labels = [0, 1, 0, 0]
    assert average_precision(scores, labels) == pytest.approx(0.5)


def test_average_precision_with_two_positives_at_ranks_2_and_4():
    # Sorted desc, labels = [0, 1, 0, 1].
    # ranks:      1    2    3    4
    # precision:  0/1  1/2  1/3  2/4
    # Positive ranks are 2 and 4: precision there is 0.5 and 0.5.
    # AP = (0.5 + 0.5) / 2 = 0.5.
    scores = [0.9, 0.7, 0.5, 0.3]
    labels = [0, 1, 0, 1]
    assert average_precision(scores, labels) == pytest.approx(0.5)


def test_average_precision_is_insensitive_to_input_order_without_ties():
    scores = [0.3, 0.9, 0.1, 0.7]
    labels = [0, 1, 0, 1]
    value_a = average_precision(scores, labels)
    # Shuffle both arrays together -- with no tied scores, the resulting
    # sorted order (and so the AP) must not change.
    permutation = [2, 0, 3, 1]
    value_b = average_precision([scores[i] for i in permutation], [labels[i] for i in permutation])
    assert value_a == pytest.approx(value_b)


def test_average_precision_tie_handling_is_deterministic_but_order_dependent():
    # Two candidates tied at the same score, one positive and one negative.
    # Ties are broken by a STABLE sort -- i.e. by original input order, not
    # by label. Swapping which of the two tied rows comes first in the
    # input therefore changes which rank the positive lands at, and so
    # changes the AP. This is the documented, deterministic-but-arbitrary
    # tie behaviour described in average_precision's docstring.
    scores_positive_first = [0.5, 0.5]
    labels_positive_first = [1, 0]  # positive row listed first
    scores_negative_first = [0.5, 0.5]
    labels_negative_first = [0, 1]  # positive row listed second

    ap_positive_first = average_precision(scores_positive_first, labels_positive_first)
    ap_negative_first = average_precision(scores_negative_first, labels_negative_first)

    # Positive-first: positive lands at rank 1 -> precision 1/1 -> AP 1.0.
    assert ap_positive_first == pytest.approx(1.0)
    # Positive-second: positive lands at rank 2 -> precision 1/2 -> AP 0.5.
    assert ap_negative_first == pytest.approx(0.5)
    # Same scores, same multiset of labels, different input order -> different AP.
    assert ap_positive_first != pytest.approx(ap_negative_first)

    # But calling it again on the exact same input is fully deterministic.
    assert average_precision(scores_positive_first, labels_positive_first) == pytest.approx(ap_positive_first)


def test_average_precision_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        average_precision([0.1, 0.2], [1])


def test_average_precision_does_not_filter_co_positives():
    # Two positives for one "disease"; both stay in the label vector -- the
    # AP is computed over the whole universe, not per-positive with the
    # other filtered out (that is Hits@K/MRR's convention, not AP's).
    scores = [0.9, 0.8, 0.5]
    labels = [1, 1, 0]
    # Both positives ranked ahead of the one negative -> perfect AP.
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_macro_auprc_is_the_mean_of_per_disease_values():
    assert macro_auprc([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_macro_auprc_is_zero_for_an_empty_collection():
    assert macro_auprc([]) == 0.0


def test_macro_and_micro_auprc_differ_when_disease_positive_counts_are_unequal():
    # Disease A: 1 candidate, 1 positive, perfect score -> AP 1.0.
    # Disease B: 100 candidates, 1 positive ranked last -> AP = 1/100 = 0.01.
    scores_a, labels_a = [1.0], [1]
    scores_b = list(np.linspace(1.0, 0.0, 100))
    labels_b = [0] * 99 + [1]

    macro = macro_auprc([average_precision(scores_a, labels_a), average_precision(scores_b, labels_b)])
    micro = micro_auprc([scores_a, scores_b], [labels_a, labels_b])

    # Macro weighs both diseases equally (mean of 1.0 and 0.01 = 0.505);
    # micro pools every candidate across both into one ranking, so disease
    # A's single candidate and disease B's 100 candidates are not weighted
    # equally -- the two must differ, proving they are not accidentally
    # computing the same thing.
    assert macro == pytest.approx(0.505)
    assert macro != pytest.approx(micro, abs=1e-3)


def test_micro_auprc_is_zero_for_an_empty_collection():
    assert micro_auprc([], []) == 0.0
