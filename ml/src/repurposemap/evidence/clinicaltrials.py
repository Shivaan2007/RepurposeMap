"""ClinicalTrials.gov adapter: a drug/disease search through the public API v2.

One request to ``GET /api/v2/studies`` with ``query.intr`` (intervention) and
``query.cond`` (condition) returns full study records directly -- unlike
PubMed, there is no separate search-then-fetch step.

What a record means: this trial's intervention and condition fields matched
the search terms. It is not a claim that the drug works for the disease, and
``status``/``phase`` are ClinicalTrials.gov's own words, copied as reported.

Failures are never turned into an empty result. A network error, a non-200
reply or an unparseable body raises ``ClinicalTrialsError``, so "the lookup
failed" cannot be mistaken for "ClinicalTrials.gov has nothing".

    adapter = ClinicalTrialsAdapter()
    result = adapter.search_with_provenance(
        ClinicalTrialsQuery(drug_name="sildenafil", disease_name="pulmonary arterial hypertension")
    )
    result.total_count, [r.nct_id for r in result.records]
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from repurposemap.evidence.models import ClinicalTrialEvidence

CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2"
CLINICALTRIALS_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"
MAX_RESULTS_LIMIT = 100

Fetcher = Callable[[str, float], bytes]


class ClinicalTrialsError(Exception):
    """A ClinicalTrials.gov lookup failed. Never swallowed into an empty result."""


class ClinicalTrialsRequestError(ClinicalTrialsError):
    """The request did not complete: network error, timeout or HTTP error."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class ClinicalTrialsParseError(ClinicalTrialsError):
    """ClinicalTrials.gov replied, but the body was not the JSON that was expected."""


class ClinicalTrialsQuery(BaseModel):
    """An explicit drug/disease query, searched as intervention and condition terms."""

    drug_name: str
    disease_name: str
    max_results: int = Field(default=10, ge=1, le=MAX_RESULTS_LIMIT)

    @field_validator("drug_name", "disease_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be empty")
        return value


class ClinicalTrialsSearchResult(BaseModel):
    """Records plus the provenance needed to read them correctly."""

    intervention_term: str
    condition_term: str
    total_count: int = Field(description="all ClinicalTrials.gov matches; records holds at most max_results of them")
    records: list[ClinicalTrialEvidence]
    retrieved_at: str


def _urllib_fetch(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "repurposemap/0.1 (research tool)"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _first_phase(phases: object) -> str | None:
    if isinstance(phases, list) and phases:
        first = phases[0]
        return str(first) if first is not None else None
    return None


def parse_studies_json(body: bytes, *, retrieved_at: str) -> tuple[list[ClinicalTrialEvidence], int]:
    """Parse a ``/studies`` response. Deterministic: document order, no network."""

    try:
        payload = json.loads(body)
    except ValueError as error:
        raise ClinicalTrialsParseError(f"/studies returned invalid JSON: {error}") from error
    if not isinstance(payload, dict) or "studies" not in payload:
        raise ClinicalTrialsParseError("expected an object with a 'studies' list")

    studies = payload["studies"]
    if not isinstance(studies, list):
        raise ClinicalTrialsParseError("'studies' was not a list")

    records: list[ClinicalTrialEvidence] = []
    for study in studies:
        protocol = (study or {}).get("protocolSection") or {}
        identification = protocol.get("identificationModule") or {}
        status_module = protocol.get("statusModule") or {}
        design_module = protocol.get("designModule") or {}

        nct_id = identification.get("nctId")
        title = identification.get("briefTitle")
        if not nct_id or not title:
            continue  # without an NCT ID and a title there is nothing to cite

        records.append(
            ClinicalTrialEvidence(
                source_id=nct_id,
                nct_id=nct_id,
                title=title,
                url=CLINICALTRIALS_STUDY_URL.format(nct_id=nct_id),
                retrieved_at=retrieved_at,
                status=status_module.get("overallStatus"),
                phase=_first_phase(design_module.get("phases")),
            )
        )

    total_count = payload.get("totalCount")
    if not isinstance(total_count, int):
        raise ClinicalTrialsParseError("'totalCount' was missing or not an integer")
    return records, total_count


class ClinicalTrialsAdapter:
    """Implements ``repurposemap.evidence.adapters.ClinicalTrialAdapter`` against ClinicalTrials.gov API v2."""

    def __init__(
        self,
        *,
        base_url: str = CLINICALTRIALS_BASE_URL,
        timeout_s: float = 10.0,
        fetcher: Fetcher = _urllib_fetch,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._fetcher, self._now = fetcher, now

    def _get(self, params: dict[str, str]) -> bytes:
        url = f"{self._base_url}/studies?{urllib.parse.urlencode(params)}"
        try:
            return self._fetcher(url, self._timeout_s)
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            hint = f" (Retry-After: {retry_after})" if retry_after else ""
            raise ClinicalTrialsRequestError(f"/studies returned HTTP {error.code}{hint}", status=error.code, retry_after=retry_after) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ClinicalTrialsRequestError(f"/studies request failed: {error}") from error

    def search_with_provenance(self, query: ClinicalTrialsQuery) -> ClinicalTrialsSearchResult:
        body = self._get(
            {
                "query.intr": query.drug_name,
                "query.cond": query.disease_name,
                "pageSize": str(query.max_results),
                "format": "json",
            }
        )
        retrieved_at = self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        records, total_count = parse_studies_json(body, retrieved_at=retrieved_at)
        return ClinicalTrialsSearchResult(
            intervention_term=query.drug_name,
            condition_term=query.disease_name,
            total_count=total_count,
            records=records,
            retrieved_at=retrieved_at,
        )

    def search(self, drug_name: str, disease_name: str, max_results: int = 10) -> list[ClinicalTrialEvidence]:
        return self.search_with_provenance(
            ClinicalTrialsQuery(drug_name=drug_name, disease_name=disease_name, max_results=max_results)
        ).records
