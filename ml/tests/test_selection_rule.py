"""The hyperparameter-search selection rule (Stage D).

The locked-test evaluation gate now lives in
``repurposemap.embeddings.export`` -- see ``test_export.py``.
"""

from __future__ import annotations

import pytest

from repurposemap.evaluation.transe_baseline import SearchCandidateResult, select_best_candidate


def test_highest_mrr_wins_when_not_close():
    candidates = [
        SearchCandidateResult("a", embedding_dim=64, val_mrr=0.01, val_macro_auprc=0.5),
        SearchCandidateResult("b", embedding_dim=128, val_mrr=0.05, val_macro_auprc=0.1),
        SearchCandidateResult("c", embedding_dim=64, val_mrr=0.02, val_macro_auprc=0.9),
    ]
    assert select_best_candidate(candidates) == 1


def test_near_tie_in_mrr_breaks_by_macro_ap():
    candidates = [
        SearchCandidateResult("a", embedding_dim=64, val_mrr=0.0500, val_macro_auprc=0.2),
        SearchCandidateResult("b", embedding_dim=128, val_mrr=0.0503, val_macro_auprc=0.4),  # within 0.0005
    ]
    assert select_best_candidate(candidates) == 1


def test_mrr_difference_just_outside_tie_threshold_does_not_tie_break():
    candidates = [
        SearchCandidateResult("a", embedding_dim=64, val_mrr=0.0500, val_macro_auprc=0.9),
        SearchCandidateResult("b", embedding_dim=128, val_mrr=0.0510, val_macro_auprc=0.1),  # > 0.0005 apart
    ]
    assert select_best_candidate(candidates) == 1  # higher MRR wins outright, AP ignored


def test_full_tie_breaks_by_smaller_embedding_dim():
    candidates = [
        SearchCandidateResult("a", embedding_dim=128, val_mrr=0.05, val_macro_auprc=0.5),
        SearchCandidateResult("b", embedding_dim=64, val_mrr=0.05, val_macro_auprc=0.5),
    ]
    assert select_best_candidate(candidates) == 1


def test_three_config_search_resolves_to_one_winner():
    candidates = [
        SearchCandidateResult("search-dim64", embedding_dim=64, val_mrr=0.012, val_macro_auprc=0.03),
        SearchCandidateResult("search-dim128", embedding_dim=128, val_mrr=0.018, val_macro_auprc=0.04),
        SearchCandidateResult("search-l1", embedding_dim=64, val_mrr=0.009, val_macro_auprc=0.02),
    ]
    assert select_best_candidate(candidates) == 1  # search-dim128, clear MRR win


def test_select_best_candidate_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one candidate"):
        select_best_candidate([])
