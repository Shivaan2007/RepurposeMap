"""TransEEvaluationResult: full-drug-universe ranking, filtered Hits@K/MRR, AUPRC (Stage C)."""

from __future__ import annotations

import numpy as np
import pytest

from repurposemap.evaluation.indications import IndicationRecord
from repurposemap.evaluation.transe_baseline import TransEOutcome, evaluate_transe_on_diseases, indications_for_diseases


def _indication(disease_id: str, drug_id: str) -> IndicationRecord:
    return IndicationRecord(disease_id=disease_id, drug_id=drug_id, disease_name=disease_id, drug_name=drug_id)


DRUG_UNIVERSE = ["Drug1", "Drug2", "Drug3", "Drug4", "Drug5"]


def test_indications_for_diseases_filters_to_the_given_set():
    records = [_indication("D1", "A"), _indication("D2", "B"), _indication("D3", "C")]
    filtered = indications_for_diseases(records, {"D1", "D3"})
    assert {r.disease_id for r in filtered} == {"D1", "D3"}


def test_full_universe_evaluation_scores_every_drug_regardless_of_discovery():
    # A hand-built score function: "Drug2" is always the best match for D1,
    # regardless of path-search-style candidate discovery (there is none
    # here -- the whole universe is always scored).
    def score_fn(disease_id, drug_ids):
        del disease_id
        return np.array([0.1, 0.9, 0.2, 0.05, 0.3])  # Drug2 wins

    indications = [_indication("D1", "Drug2")]
    result = evaluate_transe_on_diseases(score_fn, ["D1"], indications, DRUG_UNIVERSE)

    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.transe_rank == 1
    assert outcome.drug_id == "Drug2"


def test_filtered_rank_excludes_co_positive_true_drugs_for_hits_and_mrr():
    # D1 has two true drugs: Drug1 (best score) and Drug3 (second-best).
    # Drug3's filtered rank should be 1 once Drug1 is excluded -- not 2.
    def score_fn(disease_id, drug_ids):
        del disease_id
        return np.array([0.9, 0.1, 0.8, 0.05, 0.3])  # Drug1 > Drug3 > Drug5 > ...

    indications = [_indication("D1", "Drug1"), _indication("D1", "Drug3")]
    result = evaluate_transe_on_diseases(score_fn, ["D1"], indications, DRUG_UNIVERSE)

    by_drug = {o.drug_id: o for o in result.outcomes}
    assert by_drug["Drug1"].transe_rank == 1
    assert by_drug["Drug3"].transe_rank == 1  # filtered: Drug1 excluded, so Drug3 is 1st among the rest


def test_auprc_does_not_filter_co_positives_unlike_hits_and_mrr():
    # Both Drug1 and Drug3 are true; both score highly. AUPRC over the
    # unfiltered label vector (both labeled 1) should be close to perfect,
    # since the two true drugs occupy the top two ranks.
    def score_fn(disease_id, drug_ids):
        del disease_id
        return np.array([0.9, 0.1, 0.85, 0.05, 0.3])

    indications = [_indication("D1", "Drug1"), _indication("D1", "Drug3")]
    result = evaluate_transe_on_diseases(score_fn, ["D1"], indications, DRUG_UNIVERSE)
    assert result.macro_auprc == pytest.approx(1.0)
    assert result.micro_auprc == pytest.approx(1.0)


def test_hits_at_k_and_mrr_are_computed_from_outcomes():
    def score_fn(disease_id, drug_ids):
        del disease_id
        return np.array([0.9, 0.1, 0.2, 0.05, 0.3])  # Drug1 is rank 1

    indications = [_indication("D1", "Drug1")]
    result = evaluate_transe_on_diseases(score_fn, ["D1"], indications, DRUG_UNIVERSE)

    assert result.hits_at(1) == 1.0
    assert result.hits_at(3) == 1.0
    assert result.mrr == pytest.approx(1.0)


def test_a_disease_with_no_matching_indications_is_skipped_not_errored():
    def score_fn(disease_id, drug_ids):
        del disease_id
        return np.zeros(len(drug_ids))

    result = evaluate_transe_on_diseases(score_fn, ["D1", "D2"], [_indication("D1", "Drug1")], DRUG_UNIVERSE)
    # D2 has no indication record in the list given -- it contributes nothing.
    assert {o.disease_id for o in result.outcomes} == {"D1"}


def test_macro_auprc_over_several_diseases_matches_hand_computation():
    def score_fn(disease_id, drug_ids):
        del drug_ids
        if disease_id == "D1":
            return np.array([0.9, 0.1, 0.2, 0.05, 0.3])  # true drug (Drug1) ranked 1st -> AUPRC 1.0
        return np.array([0.1, 0.9, 0.2, 0.05, 0.3])  # for D2, true drug is Drug3, ranked 3rd of 5

    indications = [_indication("D1", "Drug1"), _indication("D2", "Drug3")]
    result = evaluate_transe_on_diseases(score_fn, ["D1", "D2"], indications, DRUG_UNIVERSE)

    from repurposemap.evaluation.metrics import average_precision

    # D1: true drug Drug1 at index 0 -> label vector [1,0,0,0,0]
    ap_d1 = average_precision([0.9, 0.1, 0.2, 0.05, 0.3], [1, 0, 0, 0, 0])
    # D2: true drug Drug3 at index 2 -> label vector [0,0,1,0,0]
    ap_d2 = average_precision([0.1, 0.9, 0.2, 0.05, 0.3], [0, 0, 1, 0, 0])
    expected_macro = (ap_d1 + ap_d2) / 2

    assert result.macro_auprc == pytest.approx(expected_macro)


def test_score_fn_shape_mismatch_raises():
    def bad_score_fn(disease_id, drug_ids):
        del disease_id, drug_ids
        return np.array([0.1, 0.2])  # wrong length

    with pytest.raises(ValueError, match="one score per drug"):
        evaluate_transe_on_diseases(bad_score_fn, ["D1"], [_indication("D1", "Drug1")], DRUG_UNIVERSE)


def test_transe_outcome_is_a_plain_frozen_dataclass():
    outcome = TransEOutcome(disease_id="D1", disease_name="d1", drug_id="Drug1", drug_name="drug1", transe_rank=3)
    assert outcome.transe_rank == 3
    with pytest.raises(Exception):
        outcome.transe_rank = 4
