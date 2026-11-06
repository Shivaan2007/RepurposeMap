"""``POST /api/v1/rank`` and ``GET /api/v1/rank/diseases``.

Backed by the precomputed serving-bundle mechanism
(``repurposemap.serving``), never by live model inference. A request
for a disease with no bundle under ``data/serving/`` gets an honest
``model_status="not_available"`` with an empty candidate list -- the
same response shape this endpoint always returned before a serving
mechanism existed. A request for a disease that *does* have a bundle
gets its real, already-approved candidates back, labeled
``serving_mode="precomputed"`` so a caller never mistakes this for live
inference.

``data/serving/`` is empty today (see ``repurposemap/serving/loader.py``'s
docstring) because no committed experiment artifact anywhere holds a
per-disease ranked candidate list yet -- so in practice every request
still gets ``not_available`` until ``export_rankings.py`` is run against
a real, approved ranking output. Wiring that in is a data-export step
from here on, not a code change to this router.
"""

from __future__ import annotations

from fastapi import APIRouter

from repurposemap.api.schemas import (
    CandidateDrug,
    CandidateScore,
    DiseaseQuery,
    ExplanationPath,
    ExplanationStep,
    RankingResponse,
    SupportedDiseasesResponse,
    SupportedDiseaseSummary,
)
from repurposemap.serving.loader import ServingArtifactError, list_supported_diseases, load_bundle_for_disease
from repurposemap.serving.schemas import CandidateRankingBundle

router = APIRouter(prefix="/api/v1")


def _resolve_bundle(query: DiseaseQuery) -> CandidateRankingBundle | None:
    """The serving bundle matching ``query``, or ``None``. Matches on
    ``disease_id`` by exact filename lookup, or on ``disease_name`` by
    exact case-insensitive match against a bundle's own stated disease
    name -- never a fuzzy match, since silently guessing which bundle a
    free-text disease name refers to is exactly the kind of fabrication
    this endpoint must not do. A file that exists but fails to validate
    is treated as not matching (surfacing that failure is
    ``ServingArtifactError``'s own test's job, not this request's)."""

    if query.disease_id:
        try:
            return load_bundle_for_disease(query.disease_id)
        except ServingArtifactError:
            return None
    if query.disease_name:
        needle = query.disease_name.strip().lower()
        for bundle in list_supported_diseases():
            if bundle.disease_name.lower() == needle:
                return bundle
    return None


def _to_candidate_drug(record) -> CandidateDrug:
    return CandidateDrug(
        drug_id=record.drug_id,
        drug_name=record.drug_name,
        scores=[CandidateScore(value=record.score, method="precomputed", rank=record.rank)] if record.score is not None else [],
        evidence=[],
        explanation_paths=[
            ExplanationPath(
                steps=[ExplanationStep(**step.model_dump()) for step in path.steps],
                path_score=path.path_score,
            )
            for path in record.explanation_paths
        ],
        safety_flags=[],
    )


@router.get("/rank/diseases", response_model=SupportedDiseasesResponse)
def list_rank_diseases() -> SupportedDiseasesResponse:
    bundles = list_supported_diseases()
    return SupportedDiseasesResponse(
        diseases=[
            SupportedDiseaseSummary(
                disease_id=bundle.disease_id,
                disease_name=bundle.disease_name,
                ranking_method=bundle.ranking_method,
                candidate_count=len(bundle.candidates),
            )
            for bundle in bundles
        ]
    )


@router.post("/rank", response_model=RankingResponse)
def rank_candidates(query: DiseaseQuery) -> RankingResponse:
    bundle = _resolve_bundle(query)
    if bundle is None:
        return RankingResponse(
            disease_query=query,
            candidates=[],
            model_status="not_available",
            serving_mode="not_available",
            provenance=None,
        )

    candidates = [_to_candidate_drug(record) for record in bundle.candidates[: query.top_k]]
    return RankingResponse(
        disease_query=query,
        candidates=candidates,
        model_status="ready",
        serving_mode="precomputed",
        ranking_method=bundle.ranking_method,
        candidate_universe=bundle.candidate_universe.description,
        generated_at=bundle.generated_at,
        provenance=None,
    )
