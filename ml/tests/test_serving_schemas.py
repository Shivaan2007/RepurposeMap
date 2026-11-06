"""repurposemap.serving.schemas -- CandidateRankingBundle validation.

All disease/drug names and IDs below are invented (DEMO ...) and say
nothing real about any actual disease, drug, or ranking."""

from __future__ import annotations

import pytest

from repurposemap.serving.schemas import (
    BundleValidationError,
    CandidateRankingBundle,
    CandidateUniverseDescription,
    RankedCandidateRecord,
    RankingArtifactProvenance,
    validate_bundle,
)


def _bundle(**overrides) -> CandidateRankingBundle:
    defaults = dict(
        disease_id="DEMO:0001",
        disease_name="demo disease",
        ranking_method="path_quality",
        provenance=RankingArtifactProvenance(ranking_method="path_quality"),
        generated_at="2026-11-05T00:00:00+00:00",
        candidate_universe=CandidateUniverseDescription(description="demo universe", size=2),
        candidates=[
            RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one", rank=1, score=0.9),
            RankedCandidateRecord(drug_id="DEMO:drug-2", drug_name="demo drug two", rank=2, score=0.5),
        ],
    )
    defaults.update(overrides)
    return CandidateRankingBundle(**defaults)


def test_a_well_formed_bundle_passes_validation():
    validate_bundle(_bundle())


def test_duplicate_ranks_are_rejected():
    bundle = _bundle(
        candidates=[
            RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one", rank=1),
            RankedCandidateRecord(drug_id="DEMO:drug-2", drug_name="demo drug two", rank=1),
        ]
    )
    with pytest.raises(BundleValidationError, match="duplicate rank"):
        validate_bundle(bundle)


def test_out_of_order_ranks_are_rejected():
    bundle = _bundle(
        candidates=[
            RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one", rank=2),
            RankedCandidateRecord(drug_id="DEMO:drug-2", drug_name="demo drug two", rank=1),
        ]
    )
    with pytest.raises(BundleValidationError, match="not sorted by rank"):
        validate_bundle(bundle)


def test_empty_candidate_list_is_rejected():
    bundle = _bundle(candidates=[])
    with pytest.raises(BundleValidationError, match="empty candidate list"):
        validate_bundle(bundle)


def test_schema_version_mismatch_is_rejected():
    bundle = _bundle()
    bundle.schema_version = 999
    with pytest.raises(BundleValidationError, match="schema_version"):
        validate_bundle(bundle)


def test_rank_below_one_is_rejected_by_pydantic_itself():
    with pytest.raises(ValueError):
        RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one", rank=0)
