"""repurposemap.serving.schemas -- CandidateRankingBundle validation.

All disease/drug names and IDs below are invented (DEMO ...) and say
nothing real about any actual disease, drug, or ranking."""

from __future__ import annotations

import pytest

from repurposemap.serving.schemas import (
    BundleValidationError,
    CandidateRankingBundle,
    CandidateUniverseDescription,
    ExplanationPathRecord,
    ExplanationPathStepRecord,
    RankedCandidateRecord,
    RankingArtifactProvenance,
    slugify_disease_id,
    validate_bundle,
)


def test_explanation_path_round_trips_through_json():
    path = ExplanationPathRecord(
        steps=[
            ExplanationPathStepRecord(
                source_id="DEMO:drug-1",
                source_name="demo drug one",
                source_type="drug",
                relation="demo_relation",
                display_relation="treats",
                target_id="DEMO:0001",
                target_name="demo disease",
                target_type="disease",
            )
        ],
        path_score=0.7,
    )
    record = RankedCandidateRecord(
        drug_id="DEMO:drug-1", drug_name="demo drug one", rank=1, score=0.9, explanation_paths=[path]
    )
    bundle = CandidateRankingBundle(
        disease_id="DEMO:0001",
        disease_name="demo disease",
        ranking_method="path_quality",
        provenance=RankingArtifactProvenance(ranking_method="path_quality"),
        generated_at="2026-11-05T00:00:00+00:00",
        candidate_universe=CandidateUniverseDescription(description="demo universe", size=1),
        candidates=[record],
    )

    reloaded = CandidateRankingBundle.model_validate_json(bundle.model_dump_json())

    assert reloaded.candidates[0].explanation_paths[0].path_score == 0.7
    assert reloaded.candidates[0].explanation_paths[0].steps[0].display_relation == "treats"


def test_candidate_without_explanation_paths_defaults_to_empty_list():
    record = RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one", rank=1)
    assert record.explanation_paths == []


def test_slugify_disease_id_replaces_unsafe_characters():
    assert slugify_disease_id("MONDO:0005015") == "MONDO_0005015"


def test_slugify_disease_id_is_a_no_op_for_already_safe_ids():
    assert slugify_disease_id("DEMO_0001") == "DEMO_0001"


def test_slugify_disease_id_is_deterministic():
    assert slugify_disease_id("EFO:0000400") == slugify_disease_id("EFO:0000400")


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


def test_duplicate_drug_ids_are_rejected():
    bundle = _bundle(
        candidates=[
            RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one", rank=1),
            RankedCandidateRecord(drug_id="DEMO:drug-1", drug_name="demo drug one (dup)", rank=2),
        ]
    )
    with pytest.raises(BundleValidationError, match="duplicate drug_id"):
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
