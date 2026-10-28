"""openFDA adapter: a drug label safety-signal search through the public
Drug Label API.

One request to ``GET /drug/label.json`` (matched on ``openfda.generic_name``
or ``openfda.brand_name``) returns full label records directly -- like
ClinicalTrials.gov, there is no separate search-then-fetch step.

What a record means: this label matched the drug name, and its
``description`` is the label's own boxed-warning/contraindication/warnings
text, copied as reported (labelled with which section it came from). It is
never a dosing recommendation or a clinical judgment -- a pointer to a real
label only. ``url`` links to the matching DailyMed page via the label's own
``set_id``, when openFDA reports one.

openFDA returns HTTP 404 with an ``{"error": ...}`` body for a search with
no matches -- its documented way of saying "nothing found", not a failure.
That specific case is reported as zero records, same as PubMed/
ClinicalTrials.gov report a zero count. Every other failure (network error,
timeout, any other HTTP status, or an unparseable body) raises
``OpenFDAError``, so "the lookup failed" cannot be mistaken for "openFDA has
nothing".

    adapter = OpenFDALabelAdapter()
    result = adapter.search_with_provenance(DrugLabelQuery(drug_name="sildenafil"))
    result.total_count, [r.source_id for r in result.records]
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from repurposemap.evidence.models import EvidenceSource, SafetyEvidence

OPENFDA_BASE_URL = "https://api.fda.gov"
DAILYMED_LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
MAX_RESULTS_LIMIT = 100

# Checked in this order; the first section present in a label is what
# ``description`` reports. Most safety-critical first.
_DESCRIPTION_SECTIONS = ("boxed_warning", "contraindications", "warnings", "warnings_and_cautions", "adverse_reactions")

Fetcher = Callable[[str, float], bytes]


class OpenFDAError(Exception):
    """An openFDA lookup failed. Never swallowed into an empty result (a
    genuine no-match 404 is handled separately; see the module docstring)."""


class OpenFDARequestError(OpenFDAError):
    """The request did not complete: network error, timeout or an HTTP error other than a no-match 404."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class OpenFDAParseError(OpenFDAError):
    """openFDA replied, but the body was not the JSON that was expected."""


class DrugLabelQuery(BaseModel):
    """An explicit drug-name query, searched as generic or brand name."""

    drug_name: str
    max_results: int = Field(default=10, ge=1, le=MAX_RESULTS_LIMIT)

    @field_validator("drug_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = " ".join(value.replace('"', " ").split())
        if not value:
            raise ValueError("must not be empty")
        return value

    @property
    def search_term(self) -> str:
        return f'openfda.generic_name:"{self.drug_name}" OR openfda.brand_name:"{self.drug_name}"'


class DrugLabelSearchResult(BaseModel):
    """Records plus the provenance needed to read them correctly."""

    search_term: str
    total_count: int = Field(description="all openFDA matches; records holds at most max_results of them")
    records: list[SafetyEvidence]
    retrieved_at: str


def _urllib_fetch(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "repurposemap/0.1 (research tool)"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _first(value: object) -> str | None:
    if isinstance(value, list) and value:
        first = value[0]
        return str(first) if first is not None else None
    return None


def _description(label: dict) -> str:
    for section in _DESCRIPTION_SECTIONS:
        text = _first(label.get(section))
        if text:
            return f"{section}: {text}"
    return "No boxed warning, contraindication, warnings or adverse-reactions text in this label."


def parse_labels_json(body: bytes, *, retrieved_at: str) -> tuple[list[SafetyEvidence], int]:
    """Parse a ``/drug/label.json`` response. Deterministic: document order, no network."""

    try:
        payload = json.loads(body)
    except ValueError as error:
        raise OpenFDAParseError(f"/drug/label.json returned invalid JSON: {error}") from error
    if not isinstance(payload, dict) or "results" not in payload:
        raise OpenFDAParseError("expected an object with a 'results' list")

    results = payload["results"]
    if not isinstance(results, list):
        raise OpenFDAParseError("'results' was not a list")

    records: list[SafetyEvidence] = []
    for label in results:
        label = label or {}
        source_id = label.get("id")
        if not source_id:
            continue  # without an id there is nothing to cite

        openfda = label.get("openfda") or {}
        set_id = _first(openfda.get("spl_set_id")) or _first(openfda.get("set_id"))
        records.append(
            SafetyEvidence(
                source=EvidenceSource.OPENFDA,
                source_id=source_id,
                description=_description(label),
                url=DAILYMED_LABEL_URL.format(set_id=set_id) if set_id else None,
                retrieved_at=retrieved_at,
            )
        )

    meta = payload.get("meta") or {}
    meta_results = meta.get("results") or {}
    total_count = meta_results.get("total")
    if not isinstance(total_count, int):
        raise OpenFDAParseError("'meta.results.total' was missing or not an integer")
    return records, total_count


class OpenFDALabelAdapter:
    """Implements ``repurposemap.evidence.adapters.SafetyAdapter`` against the openFDA Drug Label API."""

    def __init__(
        self,
        *,
        base_url: str = OPENFDA_BASE_URL,
        timeout_s: float = 10.0,
        api_key: str | None = None,
        fetcher: Fetcher = _urllib_fetch,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._api_key = api_key
        self._fetcher, self._now = fetcher, now

    def _get(self, params: dict[str, str]) -> bytes | None:
        """Returns ``None`` for openFDA's documented no-match 404; raises for every other failure."""

        if self._api_key:
            params = {**params, "api_key": self._api_key}
        url = f"{self._base_url}/drug/label.json?{urllib.parse.urlencode(params)}"
        try:
            return self._fetcher(url, self._timeout_s)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            retry_after = error.headers.get("Retry-After") if error.headers else None
            hint = f" (Retry-After: {retry_after})" if retry_after else ""
            raise OpenFDARequestError(
                f"/drug/label.json returned HTTP {error.code}{hint}", status=error.code, retry_after=retry_after
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OpenFDARequestError(f"/drug/label.json request failed: {error}") from error

    def search_with_provenance(self, query: DrugLabelQuery) -> DrugLabelSearchResult:
        body = self._get({"search": query.search_term, "limit": str(query.max_results)})
        retrieved_at = self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if body is None:
            return DrugLabelSearchResult(search_term=query.search_term, total_count=0, records=[], retrieved_at=retrieved_at)
        records, total_count = parse_labels_json(body, retrieved_at=retrieved_at)
        return DrugLabelSearchResult(search_term=query.search_term, total_count=total_count, records=records, retrieved_at=retrieved_at)

    def search(self, drug_name: str, max_results: int = 10) -> list[SafetyEvidence]:
        return self.search_with_provenance(DrugLabelQuery(drug_name=drug_name, max_results=max_results)).records
