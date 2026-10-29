"""``POST /api/v1/target-scorecard`` -- a real, synchronous lookup of
PLAN.md's "target prioritization" researcher-workflow step: a target's
tractability, Open Targets' own prioritisation factors, and its
highest-expressing tissues, with no disease context. Backed by the Open
Targets Platform.

Synchronous for the same reason ``/api/v1/evidence`` and
``/api/v1/disease-biology`` are -- see ``disease_biology.py``'s
docstring. Follows the same name-resolution and error-shape convention
as that router: exactly one of ``gene_symbol``/``ensembl_id`` must be
given, a name is resolved via Open Targets' own search, and a real
failure raises 404/502 rather than returning 200 with an empty
scorecard.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from repurposemap.api.schemas import TargetScorecardQuery, TargetScorecardResponse
from repurposemap.biology import OpenTargetsAdapter, OpenTargetsError, OpenTargetsNotFoundError

router = APIRouter(prefix="/api/v1")


def get_open_targets_adapter() -> OpenTargetsAdapter:
    return OpenTargetsAdapter()


@router.post("/target-scorecard", response_model=TargetScorecardResponse)
def lookup_target_scorecard(
    query: TargetScorecardQuery,
    open_targets: OpenTargetsAdapter = Depends(get_open_targets_adapter),
) -> TargetScorecardResponse:
    if (query.gene_symbol is None) == (query.ensembl_id is None):
        raise HTTPException(status_code=422, detail="exactly one of gene_symbol or ensembl_id must be given")

    resolved_from = None
    ensembl_id = query.ensembl_id
    if ensembl_id is None:
        try:
            resolved_from = open_targets.resolve_target(query.gene_symbol, max_results=5)
        except OpenTargetsError as error:
            raise HTTPException(status_code=502, detail=f"Open Targets target search failed: {error}") from error
        if not resolved_from:
            raise HTTPException(status_code=404, detail=f"Open Targets found no target matching {query.gene_symbol!r}")
        ensembl_id = resolved_from[0].id

    try:
        scorecard = open_targets.get_target_scorecard(ensembl_id)
    except OpenTargetsNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OpenTargetsError as error:
        raise HTTPException(status_code=502, detail=f"Open Targets target-scorecard lookup failed: {error}") from error

    return TargetScorecardResponse(scorecard=scorecard, resolved_from=resolved_from)
