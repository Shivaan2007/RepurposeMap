"""``GET /api/v1/system/info`` -- safe, static metadata only.

Deliberately never loads the PrimeKG graph or hashes the raw CSV (that
costs real seconds -- see ``adapters.primekg.compute_file_fingerprint``'s
own docstring on why it is never called from a hot path). The provenance
fields below mirror ``PrimeKGFingerprint``'s own hand-recorded constants
(``retrieved_date``, ``dataverse_doi``) -- update both together if either
changes.

No ranking model is configured yet, so ``model`` always reports
unavailable. This is the honest default, not a placeholder to silently
replace with a guess later.
"""

from __future__ import annotations

from fastapi import APIRouter

from repurposemap.api.schemas import ModelProvenanceInfo, ModelStatus, Provenance, SystemInfoResponse
from repurposemap.api.version import API_VERSION, APPLICATION_NAME
from repurposemap.provenance import get_model_eligibility, get_most_recent_completed_stage, load_experiments
from repurposemap.provenance.service import CONFIGURED_CHECKPOINT, is_model_available

# Mirrors PrimeKGFingerprint's defaults (repurposemap.adapters.primekg) --
# recorded by hand because PrimeKG's distribution does not embed them.
PRIMEKG_RETRIEVED_DATE = "2026-10-05"
PRIMEKG_SOURCE = "PrimeKG (Harvard Dataverse, doi:10.7910/DVN/IXA7BM)"

router = APIRouter(prefix="/api/v1")


@router.get("/system/info", response_model=SystemInfoResponse)
def get_system_info() -> SystemInfoResponse:
    # load_experiments() only reads ml/experiments/*.json -- never
    # PrimeKG, never a checkpoint, never torch.
    experiment_records = load_experiments()
    eligibility = get_model_eligibility(experiment_records)
    available = is_model_available(experiment_records)

    model = ModelStatus(available=available, model_type=None, ready_for_inference=available)
    provenance = Provenance(source=PRIMEKG_SOURCE, retrieved_date=PRIMEKG_RETRIEVED_DATE, version=None)
    # `inference_eligible` is `available`, which needs an eligible record
    # AND a configured checkpoint -- the reason text must explain that
    # exact boolean, not the more permissive `eligibility.eligible`
    # (which ignores the checkpoint). Reporting eligibility.reason here
    # unconditionally previously produced a self-contradicting pair, e.g.
    # inference_eligible=False next to "a final-training experiment
    # record was found" -- true, but not why inference is still False.
    if not eligibility.eligible:
        eligibility_reason = eligibility.reason
    elif CONFIGURED_CHECKPOINT is None:
        eligibility_reason = f"{eligibility.reason}, but no checkpoint is explicitly configured for inference yet"
    else:
        eligibility_reason = eligibility.reason
    model_provenance = ModelProvenanceInfo(
        most_recent_completed_stage=get_most_recent_completed_stage(experiment_records),
        inference_eligible=available,
        eligibility_reason=eligibility_reason,
    )
    return SystemInfoResponse(
        application=APPLICATION_NAME,
        api_version=API_VERSION,
        model=model,
        data_provenance=provenance,
        model_provenance=model_provenance,
    )
