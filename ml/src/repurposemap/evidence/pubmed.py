"""PubMed adapter: a drug/disease search through NCBI E-utilities.

``esearch`` finds PMIDs, ``efetch`` returns their records, and each record
becomes a typed ``PubMedEvidence`` that keeps its PMID, URL and the time it
was retrieved.

What a record means: this paper matched the search terms. It is not a claim
that the drug works for the disease. ``publication_types`` are PubMed's own
words, copied as reported; ``strength`` is a coarse ``EvidenceStrength``
derived from them by ``classify_strength`` below -- a deterministic mapping
over PubMed's own controlled vocabulary, never an inference from the title
or abstract text. A paper with no recognized publication type, or none at
all, gets ``UNKNOWN``, never a guessed label.

Failures are never turned into an empty result. A network error, a non-200
reply, an NCBI error message or an unparseable body raises ``PubMedError``,
so "the lookup failed" cannot be mistaken for "PubMed has nothing".
Requests are spaced by ``min_interval_s`` (NCBI allows 3 per second without
an API key); a 429 is surfaced with its Retry-After rather than retried.

    adapter = PubMedEUtilsAdapter(email="you@example.org")
    result = adapter.search_with_provenance(PubMedQuery(drug_name="sildenafil", disease_name="pulmonary arterial hypertension"))
    result.total_count, [r.pmid for r in result.records]
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from repurposemap.evidence.models import EvidenceStrength, PubMedEvidence

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_RECORD_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
MAX_RESULTS_LIMIT = 100

_MONTHS = {name: f"{number:02d}" for number, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

Fetcher = Callable[[str, float], bytes]


class PubMedError(Exception):
    """A PubMed lookup failed. Never swallowed into an empty result."""


class PubMedRequestError(PubMedError):
    """The request did not complete: network error, timeout, HTTP error or an NCBI error message."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class PubMedParseError(PubMedError):
    """NCBI replied, but the body was not the JSON/XML that was expected."""


class PubMedQuery(BaseModel):
    """An explicit drug/disease query. ``term`` is the exact PubMed search string sent."""

    drug_name: str
    disease_name: str
    max_results: int = Field(default=10, ge=1, le=MAX_RESULTS_LIMIT)

    @field_validator("drug_name", "disease_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = " ".join(value.replace('"', " ").split())  # quotes would break the quoted phrase
        if not value:
            raise ValueError("must not be empty")
        return value

    @property
    def term(self) -> str:
        def quoted(text: str) -> str:
            return f'"{text}"[Title/Abstract]'

        return f"{quoted(self.drug_name)} AND {quoted(self.disease_name)}"


class PubMedSearchResult(BaseModel):
    """Records plus the provenance needed to read them correctly."""

    query_term: str
    query_translation: str | None = Field(default=None, description="how PubMed itself interpreted the term, as reported")
    total_count: int = Field(description="all PubMed matches; records holds at most max_results of them")
    records: list[PubMedEvidence]
    retrieved_at: str


def _urllib_fetch(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "repurposemap/0.1 (research tool)"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    text = " ".join("".join(element.itertext()).split())
    return text or None


def _publication_date(article: ET.Element) -> str | None:
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = _text(pub_date.find("Year"))
        if year:
            month = _text(pub_date.find("Month"))
            if month and not month.isdigit():
                month = _MONTHS.get(month[:3].lower())  # unrecognised text such as "Spring": keep the year only
            day = _text(pub_date.find("Day"))
            parts = [year] + ([month.zfill(2)] if month else []) + ([day.zfill(2)] if month and day else [])
            return "-".join(parts)
        medline_date = _text(pub_date.find("MedlineDate"))
        if medline_date:
            return medline_date
    article_date = article.find("ArticleDate")
    if article_date is not None:
        year, month, day = (_text(article_date.find(tag)) for tag in ("Year", "Month", "Day"))
        if year:
            return "-".join(p for p in (year, month and month.zfill(2), month and day and day.zfill(2)) if p)
    return None


def _abstract(article: ET.Element) -> str | None:
    sections = []
    for node in article.findall("Abstract/AbstractText"):
        body = _text(node)
        if body:
            label = node.get("Label")
            sections.append(f"{label}: {body}" if label else body)
    return "\n".join(sections) or None


# PubMed's own MeSH publication-type strings, matched case-insensitively.
# Checked strongest first: a paper tagged with more than one type (common --
# e.g. "Randomized Controlled Trial" and "Multicenter Study" together) is
# classified by the strongest one present, not the first one PubMed listed.
_SYSTEMATIC_REVIEW_TYPES = {"systematic review", "meta-analysis"}
_CONTROLLED_TRIAL_TYPES = {
    "randomized controlled trial", "controlled clinical trial", "clinical trial",
    "clinical trial, phase i", "clinical trial, phase ii", "clinical trial, phase iii", "clinical trial, phase iv",
    "pragmatic clinical trial", "adaptive clinical trial", "equivalence trial",
}
_OBSERVATIONAL_TYPES = {"observational study", "comparative study", "multicenter study", "clinical study"}
_CASE_REPORT_TYPES = {"case reports"}


def classify_strength(publication_types: list[str]) -> EvidenceStrength:
    """Map PubMed's own publication-type labels to a coarse
    ``EvidenceStrength``, strongest type present wins. Returns ``UNKNOWN``
    for an empty list or a type this mapping does not recognize -- never a
    guess from the title, abstract, or anything PubMed did not itself say."""

    lowered = {t.lower() for t in publication_types}
    if lowered & _SYSTEMATIC_REVIEW_TYPES:
        return EvidenceStrength.SYSTEMATIC_REVIEW
    if lowered & _CONTROLLED_TRIAL_TYPES:
        return EvidenceStrength.CONTROLLED_TRIAL
    if lowered & _OBSERVATIONAL_TYPES:
        return EvidenceStrength.OBSERVATIONAL
    if lowered & _CASE_REPORT_TYPES:
        return EvidenceStrength.CASE_REPORT
    return EvidenceStrength.UNKNOWN


def parse_efetch_xml(xml_bytes: bytes, *, retrieved_at: str) -> list[PubMedEvidence]:
    """Parse an efetch ``PubmedArticleSet``. Deterministic: document order, no network."""

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as error:
        raise PubMedParseError(f"efetch returned invalid XML: {error}") from error
    if root.tag != "PubmedArticleSet":
        raise PubMedParseError(f"expected a PubmedArticleSet, got <{root.tag}>")

    records = []
    for node in root.findall("PubmedArticle"):
        pmid = _text(node.find("MedlineCitation/PMID"))
        article = node.find("MedlineCitation/Article")
        title = _text(article.find("ArticleTitle")) if article is not None else None
        if not pmid or article is None or not title:
            continue  # without a PMID and a title there is nothing to cite
        publication_types = [t for t in (_text(n) for n in article.findall("PublicationTypeList/PublicationType")) if t]
        records.append(
            PubMedEvidence(
                source_id=pmid,
                pmid=pmid,
                title=title,
                summary=_abstract(article),
                url=PUBMED_RECORD_URL.format(pmid=pmid),
                strength=classify_strength(publication_types),
                retrieved_at=retrieved_at,
                journal=_text(article.find("Journal/Title")),
                publication_date=_publication_date(article),
                publication_types=publication_types,
            )
        )
    return records


def parse_esearch_json(body: bytes) -> tuple[list[str], int, str | None]:
    """(pmids in PubMed's order, total match count, query translation)."""

    try:
        payload = json.loads(body)
        result = payload["esearchresult"]
    except (ValueError, KeyError, TypeError) as error:
        raise PubMedParseError(f"esearch returned unexpected JSON: {error!r}") from error
    if "ERROR" in result:
        raise PubMedRequestError(f"NCBI reported an error: {result['ERROR']}")
    try:
        ids = [str(i) for i in result["idlist"]]
        count = int(result["count"])
    except (KeyError, ValueError, TypeError) as error:
        raise PubMedParseError(f"esearch result lacks idlist/count: {error!r}") from error
    return ids, count, result.get("querytranslation")


class PubMedEUtilsAdapter:
    """Implements ``repurposemap.evidence.adapters.PubMedAdapter`` against E-utilities."""

    def __init__(
        self,
        *,
        base_url: str = EUTILS_BASE_URL,
        timeout_s: float = 10.0,
        min_interval_s: float = 0.34,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "repurposemap",
        fetcher: Fetcher = _urllib_fetch,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._min_interval_s = max(min_interval_s, 0.0)
        self._identity = {k: v for k, v in (("tool", tool), ("email", email), ("api_key", api_key)) if v}
        self._fetcher, self._sleep, self._monotonic, self._now = fetcher, sleep, monotonic, now
        self._last_request: float | None = None

    def _get(self, endpoint: str, params: dict[str, str]) -> bytes:
        if self._last_request is not None:
            wait = self._min_interval_s - (self._monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        url = f"{self._base_url}/{endpoint}?{urllib.parse.urlencode({**params, **self._identity})}"
        self._last_request = self._monotonic()
        try:
            return self._fetcher(url, self._timeout_s)
        except urllib.error.HTTPError as error:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            hint = f" (Retry-After: {retry_after})" if retry_after else ""
            raise PubMedRequestError(f"{endpoint} returned HTTP {error.code}{hint}", status=error.code, retry_after=retry_after) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise PubMedRequestError(f"{endpoint} request failed: {error}") from error

    def search_with_provenance(self, query: PubMedQuery) -> PubMedSearchResult:
        ids, total, translation = parse_esearch_json(
            self._get("esearch.fcgi", {"db": "pubmed", "term": query.term, "retmax": str(query.max_results), "retmode": "json"})
        )
        retrieved_at = self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        records: list[PubMedEvidence] = []
        if ids:
            fetched = parse_efetch_xml(
                self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}), retrieved_at=retrieved_at
            )
            by_pmid = {record.pmid: record for record in fetched}
            records = [by_pmid[i] for i in ids if i in by_pmid]  # keep PubMed's ranking
        return PubMedSearchResult(
            query_term=query.term, query_translation=translation, total_count=total, records=records, retrieved_at=retrieved_at
        )

    def search(self, drug_name: str, disease_name: str, max_results: int = 10) -> list[PubMedEvidence]:
        return self.search_with_provenance(
            PubMedQuery(drug_name=drug_name, disease_name=disease_name, max_results=max_results)
        ).records
