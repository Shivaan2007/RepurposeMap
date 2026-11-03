"""``POST /api/v1/evidence`` -- a real, synchronous lookup across PubMed,
ClinicalTrials.gov, openFDA and DailyMed for one drug/disease pair.

Synchronous for now: PLAN.md's tech stack calls for these slow,
rate-limited external calls to move off the request path and into a
background job (RQ + Redis) once that infrastructure exists. It does not
exist in this repo yet, so this first version calls all four adapters
in-process and simply accepts the request latency -- a stepping stone
wired into the real OpenAPI schema/TypeScript client today, not a
redesign later.

Each source is looked up independently through the adapters already built
in ``repurposemap.evidence``. One source failing (network error, rate
limit, parse error) never drops the other three's results, and is
reported in ``source_status`` with its real error message -- never
silently turned into "zero results", which would be indistinguishable
from a genuine empty answer. This endpoint never ranks, scores, or labels
a candidate; it only returns what each source reported, as reported.

Each source's lookup is wrapped twice: once for its own typed error
(``PubMedError``/``ClinicalTrialsError``/``OpenFDAError``/``DailyMedError``),
for a real, named failure mode with a clear message, and once more for any
other exception, reported with ``type(error).__name__`` rather than a
typed error's message. The second catch exists so a bug or an unanticipated
failure mode in one adapter still can't 500 the whole request and drop the
other three sources' real results -- it is a defense-in-depth backstop, not
a substitute for adding a proper typed error when a new real failure mode
is found.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from repurposemap.api.schemas import EvidenceLookupResponse, EvidenceQuery, SourceLookupStatus
from repurposemap.evidence import (
    ClinicalTrialsAdapter,
    ClinicalTrialsError,
    DailyMedAdapter,
    DailyMedError,
    EvidenceBundle,
    EvidenceSource,
    OpenFDAError,
    OpenFDALabelAdapter,
    PubMedError,
    PubMedEUtilsAdapter,
    SafetyEvidence,
)

router = APIRouter(prefix="/api/v1")


def get_pubmed_adapter() -> PubMedEUtilsAdapter:
    return PubMedEUtilsAdapter()


def get_clinicaltrials_adapter() -> ClinicalTrialsAdapter:
    return ClinicalTrialsAdapter()


def get_openfda_adapter() -> OpenFDALabelAdapter:
    return OpenFDALabelAdapter()


def get_dailymed_adapter() -> DailyMedAdapter:
    return DailyMedAdapter()


@router.post("/evidence", response_model=EvidenceLookupResponse)
def lookup_evidence(
    query: EvidenceQuery,
    pubmed: PubMedEUtilsAdapter = Depends(get_pubmed_adapter),
    clinicaltrials: ClinicalTrialsAdapter = Depends(get_clinicaltrials_adapter),
    openfda: OpenFDALabelAdapter = Depends(get_openfda_adapter),
    dailymed: DailyMedAdapter = Depends(get_dailymed_adapter),
) -> EvidenceLookupResponse:
    bundle = EvidenceBundle()
    statuses: list[SourceLookupStatus] = []
    safety: list[SafetyEvidence] = []

    try:
        bundle.pubmed = pubmed.search(query.drug_name, query.disease_name, max_results=query.max_results)
        statuses.append(SourceLookupStatus(source=EvidenceSource.PUBMED, succeeded=True))
    except PubMedError as error:
        statuses.append(SourceLookupStatus(source=EvidenceSource.PUBMED, succeeded=False, error=str(error)))
    except Exception as error:  # noqa: BLE001 -- backstop, see module docstring
        statuses.append(
            SourceLookupStatus(source=EvidenceSource.PUBMED, succeeded=False, error=f"{type(error).__name__}: {error}")
        )

    try:
        bundle.clinical_trials = clinicaltrials.search(query.drug_name, query.disease_name, max_results=query.max_results)
        statuses.append(SourceLookupStatus(source=EvidenceSource.CLINICAL_TRIALS, succeeded=True))
    except ClinicalTrialsError as error:
        statuses.append(SourceLookupStatus(source=EvidenceSource.CLINICAL_TRIALS, succeeded=False, error=str(error)))
    except Exception as error:  # noqa: BLE001 -- backstop, see module docstring
        statuses.append(
            SourceLookupStatus(
                source=EvidenceSource.CLINICAL_TRIALS, succeeded=False, error=f"{type(error).__name__}: {error}"
            )
        )

    try:
        safety.extend(openfda.search(query.drug_name, max_results=query.max_results))
        statuses.append(SourceLookupStatus(source=EvidenceSource.OPENFDA, succeeded=True))
    except OpenFDAError as error:
        statuses.append(SourceLookupStatus(source=EvidenceSource.OPENFDA, succeeded=False, error=str(error)))
    except Exception as error:  # noqa: BLE001 -- backstop, see module docstring
        statuses.append(
            SourceLookupStatus(source=EvidenceSource.OPENFDA, succeeded=False, error=f"{type(error).__name__}: {error}")
        )

    try:
        safety.extend(dailymed.search(query.drug_name, max_results=query.max_results))
        statuses.append(SourceLookupStatus(source=EvidenceSource.DAILYMED, succeeded=True))
    except DailyMedError as error:
        statuses.append(SourceLookupStatus(source=EvidenceSource.DAILYMED, succeeded=False, error=str(error)))
    except Exception as error:  # noqa: BLE001 -- backstop, see module docstring
        statuses.append(
            SourceLookupStatus(source=EvidenceSource.DAILYMED, succeeded=False, error=f"{type(error).__name__}: {error}")
        )

    bundle.safety = safety
    bundle.generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return EvidenceLookupResponse(query=query, bundle=bundle, source_status=statuses)
