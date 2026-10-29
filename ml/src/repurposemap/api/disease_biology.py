"""``POST /api/v1/disease-biology`` -- a real, synchronous lookup of
PLAN.md's "disease biology" researcher-workflow step: the genes most
strongly associated with a disease, ranked by genetic evidence, with
their Reactome pathways. Backed by the Open Targets Platform.

Synchronous for the same reason ``/api/v1/evidence`` is -- see that
router's docstring. This call is at least a network round trip, and two
when a disease name needs resolving first.

Exactly one of ``disease_name``/``disease_id`` must be given. A name is
resolved via Open Targets' own search; if it resolves to nothing, or the
resolved/given ID does not exist in Open Targets, this returns a 404 with
a clear reason -- never an empty-but-200 "module" that looks like a real,
empty result.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from repurposemap.api.schemas import DiseaseBiologyQuery, DiseaseBiologyResponse
from repurposemap.biology import OpenTargetsAdapter, OpenTargetsError, OpenTargetsNotFoundError

router = APIRouter(prefix="/api/v1")


def get_open_targets_adapter() -> OpenTargetsAdapter:
    return OpenTargetsAdapter()


@router.post("/disease-biology", response_model=DiseaseBiologyResponse)
def lookup_disease_biology(
    query: DiseaseBiologyQuery,
    open_targets: OpenTargetsAdapter = Depends(get_open_targets_adapter),
) -> DiseaseBiologyResponse:
    if (query.disease_name is None) == (query.disease_id is None):
        raise HTTPException(status_code=422, detail="exactly one of disease_name or disease_id must be given")

    resolved_from = None
    efo_id = query.disease_id
    if efo_id is None:
        try:
            resolved_from = open_targets.resolve_disease(query.disease_name, max_results=5)
        except OpenTargetsError as error:
            raise HTTPException(status_code=502, detail=f"Open Targets disease search failed: {error}") from error
        if not resolved_from:
            raise HTTPException(status_code=404, detail=f"Open Targets found no disease matching {query.disease_name!r}")
        efo_id = resolved_from[0].id

    try:
        module = open_targets.get_disease_biology(efo_id, max_targets=query.max_targets)
    except OpenTargetsNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OpenTargetsError as error:
        raise HTTPException(status_code=502, detail=f"Open Targets disease-biology lookup failed: {error}") from error

    return DiseaseBiologyResponse(module=module, resolved_from=resolved_from)
