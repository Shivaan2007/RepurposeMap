"""PubMed adapter -- mocked HTTP only, no network.

The XML/JSON below are SYNTHETIC fixtures in E-utilities' real format. The
PMIDs, titles and journals are invented (DEMO ...) and say nothing about any
real drug or disease."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repurposemap.evidence import EvidenceSource, EvidenceStrength
from repurposemap.evidence.pubmed import (
    PubMedEUtilsAdapter,
    PubMedParseError,
    PubMedQuery,
    PubMedRequestError,
    parse_efetch_xml,
    parse_esearch_json,
)

RETRIEVED = "2026-10-26T12:00:00Z"

EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">90000002</PMID>
      <Article>
        <Journal><Title>DEMO Journal of Tests</Title>
          <JournalIssue><PubDate><Year>2021</Year><Month>Mar</Month><Day>5</Day></PubDate></JournalIssue></Journal>
        <ArticleTitle>DEMO study of <i>Drug A</i> in Disease B</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Demo background.</AbstractText>
          <AbstractText Label="RESULTS">Demo results.</AbstractText>
        </Abstract>
        <PublicationTypeList><PublicationType>Journal Article</PublicationType><PublicationType>Case Reports</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">90000001</PMID>
      <Article>
        <Journal><Title>DEMO Letters</Title>
          <JournalIssue><PubDate><MedlineDate>2019 Dec-2020 Jan</MedlineDate></PubDate></JournalIssue></Journal>
        <ArticleTitle>DEMO letter with no abstract</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation><Article><ArticleTitle>DEMO record with no PMID</ArticleTitle></Article></MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def esearch_json(ids, count=None, translation='"drug a"[Title/Abstract]'):
    return json.dumps({"esearchresult": {"count": str(len(ids) if count is None else count), "idlist": ids, "querytranslation": translation}}).encode()


class FakeFetcher:
    def __init__(self, *replies):
        self.replies, self.urls = list(replies), []

    def __call__(self, url, timeout_s):
        self.urls.append((url, timeout_s))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_adapter(fetcher, **kwargs):
    return PubMedEUtilsAdapter(
        fetcher=fetcher, sleep=lambda _s: None, now=lambda: datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc), **kwargs
    )


def test_query_builds_a_deterministic_quoted_term_and_strips_quotes():
    query = PubMedQuery(drug_name='  Drug  "A" ', disease_name="Disease B")
    assert query.term == '"Drug A"[Title/Abstract] AND "Disease B"[Title/Abstract]'
    assert query.term == PubMedQuery(drug_name='Drug "A"', disease_name="Disease B").term


@pytest.mark.parametrize("kwargs", [{"drug_name": " ", "disease_name": "x"}, {"drug_name": "x", "disease_name": ""}, {"drug_name": "x", "disease_name": "y", "max_results": 0}, {"drug_name": "x", "disease_name": "y", "max_results": 101}])
def test_query_rejects_empty_names_and_out_of_range_limits(kwargs):
    with pytest.raises(ValidationError):
        PubMedQuery(**kwargs)


def test_parse_efetch_extracts_metadata_abstract_and_provenance():
    records = parse_efetch_xml(EFETCH_XML, retrieved_at=RETRIEVED)
    assert [r.pmid for r in records] == ["90000002", "90000001"]  # document order; the PMID-less record is skipped
    first, second = records
    assert first.title == "DEMO study of Drug A in Disease B"  # inline markup flattened
    assert first.summary == "BACKGROUND: Demo background.\nRESULTS: Demo results."
    assert first.journal == "DEMO Journal of Tests"
    assert first.publication_date == "2021-03-05"
    assert first.publication_types == ["Journal Article", "Case Reports"]
    assert first.url == "https://pubmed.ncbi.nlm.nih.gov/90000002/"
    assert first.source == EvidenceSource.PUBMED and first.source_id == "90000002"
    assert first.retrieved_at == RETRIEVED
    assert second.summary is None  # no abstract is reported as None, never invented
    assert second.publication_date == "2019 Dec-2020 Jan"


def test_records_carry_no_evidence_label():
    assert all(r.strength == EvidenceStrength.UNKNOWN for r in parse_efetch_xml(EFETCH_XML, retrieved_at=RETRIEVED))


def test_parse_efetch_is_deterministic():
    assert parse_efetch_xml(EFETCH_XML, retrieved_at=RETRIEVED) == parse_efetch_xml(EFETCH_XML, retrieved_at=RETRIEVED)


@pytest.mark.parametrize("body", [b"not xml", b"<Other/>"])
def test_parse_efetch_rejects_unexpected_bodies(body):
    with pytest.raises(PubMedParseError):
        parse_efetch_xml(body, retrieved_at=RETRIEVED)


def test_parse_esearch_returns_ids_count_and_translation():
    assert parse_esearch_json(esearch_json(["2", "1"], count=57)) == (["2", "1"], 57, '"drug a"[Title/Abstract]')


def test_parse_esearch_surfaces_ncbi_errors_and_bad_json():
    with pytest.raises(PubMedRequestError, match="bad term"):
        parse_esearch_json(json.dumps({"esearchresult": {"ERROR": "bad term"}}).encode())
    for body in (b"<html>", json.dumps({"nope": 1}).encode(), json.dumps({"esearchresult": {"idlist": []}}).encode()):
        with pytest.raises(PubMedParseError):
            parse_esearch_json(body)


def test_search_keeps_pubmeds_ranking_and_reports_total_count():
    fetcher = FakeFetcher(esearch_json(["90000001", "90000002", "90000099"], count=412), EFETCH_XML)
    result = make_adapter(fetcher, timeout_s=7.0, email="a@example.org", api_key="K").search_with_provenance(
        PubMedQuery(drug_name="Drug A", disease_name="Disease B", max_results=3)
    )
    assert [r.pmid for r in result.records] == ["90000001", "90000002"]  # efetch order was reversed; 90000099 was not returned
    assert result.total_count == 412
    assert result.query_term == '"Drug A"[Title/Abstract] AND "Disease B"[Title/Abstract]'
    assert result.query_translation == '"drug a"[Title/Abstract]'
    assert result.retrieved_at == RETRIEVED and all(r.retrieved_at == RETRIEVED for r in result.records)

    (search_url, search_timeout), (fetch_url, _) = fetcher.urls
    assert search_timeout == 7.0
    search_query = urllib.parse.parse_qs(urllib.parse.urlparse(search_url).query)
    assert urllib.parse.urlparse(search_url).path.endswith("/esearch.fcgi")
    assert search_query["db"] == ["pubmed"] and search_query["retmax"] == ["3"]
    assert search_query["term"] == [result.query_term]
    assert search_query["email"] == ["a@example.org"] and search_query["api_key"] == ["K"] and search_query["tool"] == ["repurposemap"]
    assert urllib.parse.parse_qs(urllib.parse.urlparse(fetch_url).query)["id"] == ["90000001,90000002,90000099"]


def test_search_with_no_matches_skips_efetch_and_says_zero():
    fetcher = FakeFetcher(esearch_json([], count=0))
    result = make_adapter(fetcher).search_with_provenance(PubMedQuery(drug_name="x", disease_name="y"))
    assert result.records == [] and result.total_count == 0
    assert len(fetcher.urls) == 1


def test_protocol_search_returns_a_plain_list_of_records():
    fetcher = FakeFetcher(esearch_json(["90000002"]), EFETCH_XML)
    records = make_adapter(fetcher).search("Drug A", "Disease B", max_results=1)
    assert [r.pmid for r in records] == ["90000002"]


def test_network_failure_raises_instead_of_returning_an_empty_list():
    adapter = make_adapter(FakeFetcher(urllib.error.URLError("no route")))
    with pytest.raises(PubMedRequestError, match="no route"):
        adapter.search("x", "y")
    with pytest.raises(PubMedRequestError):
        make_adapter(FakeFetcher(TimeoutError("timed out"))).search("x", "y")


def test_rate_limit_is_surfaced_with_retry_after():
    error = urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests", {"Retry-After": "2"}, None)
    with pytest.raises(PubMedRequestError) as raised:
        make_adapter(FakeFetcher(error)).search("x", "y")
    assert raised.value.status == 429 and raised.value.retry_after == "2"
    assert "Retry-After: 2" in str(raised.value)


def test_efetch_failure_after_a_good_esearch_still_raises():
    fetcher = FakeFetcher(esearch_json(["90000001"]), urllib.error.URLError("dropped"))
    with pytest.raises(PubMedRequestError):
        make_adapter(fetcher).search("x", "y")


def test_requests_are_spaced_by_the_minimum_interval():
    clock = {"t": 100.0}
    sleeps = []
    fetcher = FakeFetcher(esearch_json(["90000001"]), EFETCH_XML)
    adapter = PubMedEUtilsAdapter(
        fetcher=fetcher, min_interval_s=0.5, sleep=lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)),
        monotonic=lambda: clock["t"], now=lambda: datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc),
    )
    adapter.search("x", "y")
    assert sleeps == [pytest.approx(0.5)]  # none before the first request, one wait before the second


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        PubMedEUtilsAdapter(timeout_s=0)
