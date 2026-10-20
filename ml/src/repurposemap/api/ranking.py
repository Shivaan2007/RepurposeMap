"""``POST /api/v1/rank`` -- the future candidate-ranking contract, stubbed.

Exists so the full ranking schema (``DiseaseQuery`` through
``RankingResponse``, including ``ExplanationPath``/``SafetyFlag``/etc.)
is part of the real OpenAPI schema today, not added later as a breaking
change -- but it is NOT backed by a model. It always returns
``model_status="not_available"`` with an empty candidate list. It never
fabricates a drug/disease result, never falls back to a random or mock
ranking, and never pretends a production model exists.

Wiring a real model in later means replacing this function's body with an
actual scoring call (e.g. the disease-based evaluation path baseline, or a
trained TransE checkpoint) -- the request/response contract already
matches what either would need to return.
"""

from __future__ import annotations

from fastapi import APIRouter

from repurposemap.api.schemas import DiseaseQuery, RankingResponse

router = APIRouter(prefix="/api/v1")


@router.post("/rank", response_model=RankingResponse)
def rank_candidates(query: DiseaseQuery) -> RankingResponse:
    return RankingResponse(disease_query=query, candidates=[], model_status="not_available", provenance=None)
