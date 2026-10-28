"""DailyMed adapter -- mocked HTTP only, no network.

The JSON below is a SYNTHETIC fixture in the SPL API v2 real format. The
set ids and titles are invented (DEMO ...) and say nothing about any real
drug."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repurposemap.evidence import EvidenceSource
from repurposemap.evidence.dailymed import (
    DailyMedAdapter,
    DailyMedParseError,
    DailyMedQuery,
    DailyMedRequestError,
    parse_spls_json,
)

RETRIEVED = "2026-10-27T12:00:00Z"


def spls_json(data, total=None):
    return json.dumps({"data": data, "metadata": {"total_elements": len(data) if total is None else total}}).encode()


def demo_entry(set_id, *, title="DEMO LABEL (DEMO DRUG) TABLET", published_date="Nov 01, 2013"):
    return {"setid": set_id, "title": title, "published_date": published_date}


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
    return DailyMedAdapter(fetcher=fetcher, now=lambda: datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc), **kwargs)


@pytest.mark.parametrize("kwargs", [{"drug_name": " "}, {"drug_name": "x", "max_results": 0}, {"drug_name": "x", "max_results": 101}])
def test_query_rejects_empty_names_and_out_of_range_limits(kwargs):
    with pytest.raises(ValidationError):
        DailyMedQuery(**kwargs)


def test_parse_spls_extracts_title_date_and_url_as_reported():
    body = spls_json([demo_entry("DEMO-SET-1")])
    records, total = parse_spls_json(body, retrieved_at=RETRIEVED)
    assert len(records) == 1 and total == 1
    record = records[0]
    assert record.source == EvidenceSource.DAILYMED and record.source_id == "DEMO-SET-1"
    assert record.description == "DEMO LABEL (DEMO DRUG) TABLET (published Nov 01, 2013)"  # verbatim date text, not reformatted
    assert record.url == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=DEMO-SET-1"
    assert record.retrieved_at == RETRIEVED


def test_parse_spls_omits_the_published_date_clause_when_absent():
    body = spls_json([demo_entry("DEMO-SET-2", published_date=None)])
    records, _ = parse_spls_json(body, retrieved_at=RETRIEVED)
    assert records[0].description == "DEMO LABEL (DEMO DRUG) TABLET"


def test_parse_spls_skips_records_without_a_setid_or_title():
    missing_setid = {"title": "x", "published_date": "y"}
    missing_title = {"setid": "DEMO-SET-3", "published_date": "y"}
    body = spls_json([missing_setid, missing_title, demo_entry("DEMO-SET-4")])
    records, _ = parse_spls_json(body, retrieved_at=RETRIEVED)
    assert [r.source_id for r in records] == ["DEMO-SET-4"]


def test_parse_spls_is_deterministic():
    body = spls_json([demo_entry("DEMO-SET-5")])
    assert parse_spls_json(body, retrieved_at=RETRIEVED) == parse_spls_json(body, retrieved_at=RETRIEVED)


@pytest.mark.parametrize("body", [b"not json", json.dumps({"nope": 1}).encode(), json.dumps({"data": "x", "metadata": {"total_elements": 1}}).encode(), json.dumps({"data": []}).encode()])
def test_parse_spls_rejects_unexpected_bodies(body):
    with pytest.raises(DailyMedParseError):
        parse_spls_json(body, retrieved_at=RETRIEVED)


def test_search_builds_the_right_query_and_reports_total_count():
    fetcher = FakeFetcher(spls_json([demo_entry("DEMO-SET-6")], total=42))
    result = make_adapter(fetcher, timeout_s=7.0).search_with_provenance(
        DailyMedQuery(drug_name="Drug A", max_results=5)
    )
    assert [r.source_id for r in result.records] == ["DEMO-SET-6"]
    assert result.total_count == 42
    assert result.drug_name == "Drug A"
    assert result.retrieved_at == RETRIEVED and all(r.retrieved_at == RETRIEVED for r in result.records)

    (url, timeout), = fetcher.urls
    assert timeout == 7.0
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert urllib.parse.urlparse(url).path.endswith("/spls.json")
    assert parsed["drug_name"] == ["Drug A"] and parsed["pagesize"] == ["5"]


def test_search_with_no_matches_says_zero():
    fetcher = FakeFetcher(spls_json([], total=0))
    result = make_adapter(fetcher).search_with_provenance(DailyMedQuery(drug_name="x"))
    assert result.records == [] and result.total_count == 0


def test_protocol_search_returns_a_plain_list_of_records():
    fetcher = FakeFetcher(spls_json([demo_entry("DEMO-SET-7")]))
    records = make_adapter(fetcher).search("Drug A", max_results=1)
    assert [r.source_id for r in records] == ["DEMO-SET-7"]


def test_network_failure_raises_instead_of_returning_an_empty_list():
    adapter = make_adapter(FakeFetcher(urllib.error.URLError("no route")))
    with pytest.raises(DailyMedRequestError, match="no route"):
        adapter.search("x")
    with pytest.raises(DailyMedRequestError):
        make_adapter(FakeFetcher(TimeoutError("timed out"))).search("x")


def test_rate_limit_is_surfaced_with_retry_after():
    error = urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests", {"Retry-After": "2"}, None)
    with pytest.raises(DailyMedRequestError) as raised:
        make_adapter(FakeFetcher(error)).search("x")
    assert raised.value.status == 429 and raised.value.retry_after == "2"
    assert "Retry-After: 2" in str(raised.value)


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        DailyMedAdapter(timeout_s=0)
