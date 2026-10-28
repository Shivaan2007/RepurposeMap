"""DailyMed adapter: a drug-name search through the public SPL API v2.

One request to ``GET /dailymed/services/v2/spls.json`` (filtered by
``drug_name``) returns a page of SPL (Structured Product Labeling) records
directly -- like ClinicalTrials.gov and openFDA, there is no separate
search-then-fetch step. This search endpoint reports each label's own
title and publication date; it does not return section text (boxed
warning, contraindications, ...) -- that is what the openFDA adapter
(``openfda.py``) gives instead, from the same underlying SPL documents.

What a record means: this label's drug name matched the search. ``title``
and ``published_date`` are DailyMed's own fields, copied as reported
(``published_date`` stays whatever text DailyMed sends, e.g. "Nov 01,
2013" -- not reformatted, so nothing is inferred about its precision).
Never a dosing recommendation or a clinical judgment -- a pointer to a real
label only.

Failures are never turned into an empty result. A network error, a non-200
reply or an unparseable body raises ``DailyMedError``, so "the lookup
failed" cannot be mistaken for "DailyMed has nothing".

    adapter = DailyMedAdapter()
    result = adapter.search_with_provenance(DailyMedQuery(drug_name="sildenafil"))
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

DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
DAILYMED_LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
MAX_RESULTS_LIMIT = 100


class DailyMedError(Exception):
    """A DailyMed lookup failed. Never swallowed into an empty result."""


class DailyMedRequestError(DailyMedError):
    """The request did not complete: network error, timeout or HTTP error."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class DailyMedParseError(DailyMedError):
    """DailyMed replied, but the body was not the JSON that was expected."""


Fetcher = Callable[[str, float], bytes]


class DailyMedQuery(BaseModel):
    """An explicit drug-name query."""

    drug_name: str
    max_results: int = Field(default=10, ge=1, le=MAX_RESULTS_LIMIT)

    @field_validator("drug_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be empty")
        return value


class DailyMedSearchResult(BaseModel):
    """Records plus the provenance needed to read them correctly."""

    drug_name: str
    total_count: int = Field(description="all DailyMed matches; records holds at most max_results of them")
    records: list[SafetyEvidence]
    retrieved_at: str


def _urllib_fetch(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "repurposemap/0.1 (research tool)"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _description(title: str, published_date: str | None) -> str:
    return f"{title} (published {published_date})" if published_date else title


def parse_spls_json(body: bytes, *, retrieved_at: str) -> tuple[list[SafetyEvidence], int]:
    """Parse a ``/spls.json`` response. Deterministic: document order, no network."""

    try:
        payload = json.loads(body)
    except ValueError as error:
        raise DailyMedParseError(f"/spls.json returned invalid JSON: {error}") from error
    if not isinstance(payload, dict) or "data" not in payload:
        raise DailyMedParseError("expected an object with a 'data' list")

    data = payload["data"]
    if not isinstance(data, list):
        raise DailyMedParseError("'data' was not a list")

    records: list[SafetyEvidence] = []
    for entry in data:
        entry = entry or {}
        set_id = entry.get("setid")
        title = entry.get("title")
        if not set_id or not title:
            continue  # without a setid and a title there is nothing to cite
        records.append(
            SafetyEvidence(
                source=EvidenceSource.DAILYMED,
                source_id=set_id,
                description=_description(title, entry.get("published_date")),
                url=DAILYMED_LABEL_URL.format(set_id=set_id),
                retrieved_at=retrieved_at,
            )
        )

    metadata = payload.get("metadata") or {}
    total_count = metadata.get("total_elements")
    if not isinstance(total_count, int):
        raise DailyMedParseError("'metadata.total_elements' was missing or not an integer")
    return records, total_count


class DailyMedAdapter:
    """Implements ``repurposemap.evidence.adapters.SafetyAdapter`` against DailyMed's SPL API v2."""

    def __init__(
        self,
        *,
        base_url: str = DAILYMED_BASE_URL,
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
        url = f"{self._base_url}/spls.json?{urllib.parse.urlencode(params)}"
        try:
            return self._fetcher(url, self._timeout_s)
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            hint = f" (Retry-After: {retry_after})" if retry_after else ""
            raise DailyMedRequestError(f"/spls.json returned HTTP {error.code}{hint}", status=error.code, retry_after=retry_after) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DailyMedRequestError(f"/spls.json request failed: {error}") from error

    def search_with_provenance(self, query: DailyMedQuery) -> DailyMedSearchResult:
        body = self._get({"drug_name": query.drug_name, "pagesize": str(query.max_results)})
        retrieved_at = self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        records, total_count = parse_spls_json(body, retrieved_at=retrieved_at)
        return DailyMedSearchResult(drug_name=query.drug_name, total_count=total_count, records=records, retrieved_at=retrieved_at)

    def search(self, drug_name: str, max_results: int = 10) -> list[SafetyEvidence]:
        return self.search_with_provenance(DailyMedQuery(drug_name=drug_name, max_results=max_results)).records
