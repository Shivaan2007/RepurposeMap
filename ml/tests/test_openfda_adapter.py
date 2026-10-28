"""openFDA adapter -- mocked HTTP only, no network.

The JSON below is a SYNTHETIC fixture in the Drug Label API's real format.
The ids, set ids and label text are invented (DEMO ...) and say nothing
about any real drug."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repurposemap.evidence import EvidenceSource
from repurposemap.evidence.openfda import (
    DrugLabelQuery,
    OpenFDALabelAdapter,
    OpenFDAParseError,
    OpenFDARequestError,
    parse_labels_json,
)

RETRIEVED = "2026-10-27T12:00:00Z"


def labels_json(results, total=None):
    return json.dumps({"results": results, "meta": {"results": {"total": len(results) if total is None else total}}}).encode()


def demo_label(label_id, *, set_id="DEMO-SET-1", boxed_warning=("DEMO boxed warning text.",), **sections):
    return {
        "id": label_id,
        "openfda": {"spl_set_id": [set_id]} if set_id else {},
        "boxed_warning": list(boxed_warning) if boxed_warning is not None else None,
        **sections,
    }


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
    return OpenFDALabelAdapter(fetcher=fetcher, now=lambda: datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc), **kwargs)


@pytest.mark.parametrize("kwargs", [{"drug_name": " "}, {"drug_name": "x", "max_results": 0}, {"drug_name": "x", "max_results": 101}])
def test_query_rejects_empty_names_and_out_of_range_limits(kwargs):
    with pytest.raises(ValidationError):
        DrugLabelQuery(**kwargs)


def test_query_builds_a_generic_or_brand_name_term():
    query = DrugLabelQuery(drug_name='Drug "A"')
    assert query.search_term == 'openfda.generic_name:"Drug A" OR openfda.brand_name:"Drug A"'


def test_parse_labels_extracts_boxed_warning_and_dailymed_url():
    body = labels_json([demo_label("DEMO-ID-1")])
    records, total = parse_labels_json(body, retrieved_at=RETRIEVED)
    assert len(records) == 1 and total == 1
    record = records[0]
    assert record.source == EvidenceSource.OPENFDA and record.source_id == "DEMO-ID-1"
    assert record.description == "boxed_warning: DEMO boxed warning text."
    assert record.url == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=DEMO-SET-1"
    assert record.retrieved_at == RETRIEVED


def test_parse_labels_falls_back_through_sections_in_priority_order():
    no_boxed = demo_label("DEMO-ID-2", boxed_warning=None, contraindications=["DEMO contraindication."])
    records, _ = parse_labels_json(labels_json([no_boxed]), retrieved_at=RETRIEVED)
    assert records[0].description == "contraindications: DEMO contraindication."

    only_adverse = demo_label("DEMO-ID-3", boxed_warning=None, adverse_reactions=["DEMO adverse reaction."])
    records, _ = parse_labels_json(labels_json([only_adverse]), retrieved_at=RETRIEVED)
    assert records[0].description == "adverse_reactions: DEMO adverse reaction."


def test_parse_labels_reports_no_sections_without_inventing_text():
    empty = demo_label("DEMO-ID-4", boxed_warning=None)
    records, _ = parse_labels_json(labels_json([empty]), retrieved_at=RETRIEVED)
    assert records[0].description == "No boxed warning, contraindication, warnings or adverse-reactions text in this label."


def test_parse_labels_reports_no_url_without_a_set_id():
    no_set_id = demo_label("DEMO-ID-5", set_id=None)
    records, _ = parse_labels_json(labels_json([no_set_id]), retrieved_at=RETRIEVED)
    assert records[0].url is None


def test_parse_labels_skips_records_without_an_id():
    missing_id = {"openfda": {}, "boxed_warning": ["x"]}
    records, _ = parse_labels_json(labels_json([missing_id, demo_label("DEMO-ID-6")]), retrieved_at=RETRIEVED)
    assert [r.source_id for r in records] == ["DEMO-ID-6"]


def test_parse_labels_is_deterministic():
    body = labels_json([demo_label("DEMO-ID-7")])
    assert parse_labels_json(body, retrieved_at=RETRIEVED) == parse_labels_json(body, retrieved_at=RETRIEVED)


@pytest.mark.parametrize("body", [b"not json", json.dumps({"nope": 1}).encode(), json.dumps({"results": "x", "meta": {"results": {"total": 1}}}).encode(), json.dumps({"results": []}).encode()])
def test_parse_labels_rejects_unexpected_bodies(body):
    with pytest.raises(OpenFDAParseError):
        parse_labels_json(body, retrieved_at=RETRIEVED)


def test_search_builds_the_right_query_and_reports_total_count():
    fetcher = FakeFetcher(labels_json([demo_label("DEMO-ID-8")], total=57))
    result = make_adapter(fetcher, timeout_s=7.0, api_key="K").search_with_provenance(
        DrugLabelQuery(drug_name="Drug A", max_results=5)
    )
    assert [r.source_id for r in result.records] == ["DEMO-ID-8"]
    assert result.total_count == 57
    assert result.retrieved_at == RETRIEVED and all(r.retrieved_at == RETRIEVED for r in result.records)

    (url, timeout), = fetcher.urls
    assert timeout == 7.0
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert urllib.parse.urlparse(url).path.endswith("/drug/label.json")
    assert parsed["search"] == [DrugLabelQuery(drug_name="Drug A").search_term]
    assert parsed["limit"] == ["5"] and parsed["api_key"] == ["K"]


def test_404_no_match_is_reported_as_zero_not_an_error():
    error = urllib.error.HTTPError("https://example.invalid", 404, "Not Found", None, None)
    result = make_adapter(FakeFetcher(error)).search_with_provenance(DrugLabelQuery(drug_name="x"))
    assert result.records == [] and result.total_count == 0


def test_protocol_search_returns_a_plain_list_of_records():
    fetcher = FakeFetcher(labels_json([demo_label("DEMO-ID-9")]))
    records = make_adapter(fetcher).search("Drug A", max_results=1)
    assert [r.source_id for r in records] == ["DEMO-ID-9"]


def test_network_failure_raises_instead_of_returning_an_empty_list():
    adapter = make_adapter(FakeFetcher(urllib.error.URLError("no route")))
    with pytest.raises(OpenFDARequestError, match="no route"):
        adapter.search("x")
    with pytest.raises(OpenFDARequestError):
        make_adapter(FakeFetcher(TimeoutError("timed out"))).search("x")


def test_non_404_http_errors_still_raise():
    error = urllib.error.HTTPError("https://example.invalid", 500, "Server Error", None, None)
    with pytest.raises(OpenFDARequestError) as raised:
        make_adapter(FakeFetcher(error)).search("x")
    assert raised.value.status == 500


def test_rate_limit_is_surfaced_with_retry_after():
    error = urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests", {"Retry-After": "2"}, None)
    with pytest.raises(OpenFDARequestError) as raised:
        make_adapter(FakeFetcher(error)).search("x")
    assert raised.value.status == 429 and raised.value.retry_after == "2"
    assert "Retry-After: 2" in str(raised.value)


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        OpenFDALabelAdapter(timeout_s=0)
