"""ClinicalTrials.gov adapter -- mocked HTTP only, no network.

The JSON below is a SYNTHETIC fixture in the API v2 real format. The NCT
IDs and titles are invented (DEMO ...) and say nothing about any real drug
or disease."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repurposemap.evidence import EvidenceSource
from repurposemap.evidence.clinicaltrials import (
    ClinicalTrialsAdapter,
    ClinicalTrialsParseError,
    ClinicalTrialsQuery,
    ClinicalTrialsRequestError,
    parse_studies_json,
)

RETRIEVED = "2026-10-27T12:00:00Z"


def studies_json(studies, total_count=None):
    return json.dumps({"studies": studies, "totalCount": len(studies) if total_count is None else total_count}).encode()


def demo_study(nct_id, *, title="DEMO study of Drug A in Disease B", status="RECRUITING", phases=("PHASE2",)):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": title},
            "statusModule": {"overallStatus": status},
            "designModule": {"phases": list(phases) if phases is not None else None},
        }
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
    return ClinicalTrialsAdapter(fetcher=fetcher, now=lambda: datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc), **kwargs)


@pytest.mark.parametrize("kwargs", [{"drug_name": " ", "disease_name": "x"}, {"drug_name": "x", "disease_name": ""}, {"drug_name": "x", "disease_name": "y", "max_results": 0}, {"drug_name": "x", "disease_name": "y", "max_results": 101}])
def test_query_rejects_empty_names_and_out_of_range_limits(kwargs):
    with pytest.raises(ValidationError):
        ClinicalTrialsQuery(**kwargs)


def test_parse_studies_extracts_metadata_and_provenance():
    body = studies_json([demo_study("NCT90000002"), demo_study("NCT90000001", status="COMPLETED", phases=("PHASE1", "PHASE2"))])
    records, total_count = parse_studies_json(body, retrieved_at=RETRIEVED)
    assert [r.nct_id for r in records] == ["NCT90000002", "NCT90000001"]  # document order preserved
    first, second = records
    assert first.title == "DEMO study of Drug A in Disease B"
    assert first.status == "RECRUITING"
    assert first.phase == "PHASE2"
    assert first.url == "https://clinicaltrials.gov/study/NCT90000002"
    assert first.source == EvidenceSource.CLINICAL_TRIALS and first.source_id == "NCT90000002"
    assert first.retrieved_at == RETRIEVED
    assert second.status == "COMPLETED" and second.phase == "PHASE1"  # first of several phases, as reported
    assert total_count == 2


def test_parse_studies_skips_records_without_an_nct_id_or_title():
    missing_id = {"protocolSection": {"identificationModule": {"briefTitle": "no id"}, "statusModule": {}, "designModule": {}}}
    missing_title = {"protocolSection": {"identificationModule": {"nctId": "NCT1"}, "statusModule": {}, "designModule": {}}}
    body = studies_json([missing_id, missing_title, demo_study("NCT90000003")])
    records, _ = parse_studies_json(body, retrieved_at=RETRIEVED)
    assert [r.nct_id for r in records] == ["NCT90000003"]


def test_parse_studies_reports_no_phase_when_absent():
    body = studies_json([demo_study("NCT90000004", phases=None)])
    records, _ = parse_studies_json(body, retrieved_at=RETRIEVED)
    assert records[0].phase is None


def test_parse_studies_is_deterministic():
    body = studies_json([demo_study("NCT90000005")])
    assert parse_studies_json(body, retrieved_at=RETRIEVED) == parse_studies_json(body, retrieved_at=RETRIEVED)


@pytest.mark.parametrize("body", [b"not json", json.dumps({"nope": 1}).encode(), json.dumps({"studies": "x", "totalCount": 1}).encode(), json.dumps({"studies": []}).encode()])
def test_parse_studies_rejects_unexpected_bodies(body):
    with pytest.raises(ClinicalTrialsParseError):
        parse_studies_json(body, retrieved_at=RETRIEVED)


def test_search_reports_total_count_and_builds_the_right_query():
    fetcher = FakeFetcher(studies_json([demo_study("NCT90000001")], total_count=412))
    result = make_adapter(fetcher, timeout_s=7.0).search_with_provenance(
        ClinicalTrialsQuery(drug_name="Drug A", disease_name="Disease B", max_results=5)
    )
    assert [r.nct_id for r in result.records] == ["NCT90000001"]
    assert result.total_count == 412
    assert result.intervention_term == "Drug A" and result.condition_term == "Disease B"
    assert result.retrieved_at == RETRIEVED and all(r.retrieved_at == RETRIEVED for r in result.records)

    (url, timeout), = fetcher.urls
    assert timeout == 7.0
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert urllib.parse.urlparse(url).path.endswith("/studies")
    assert parsed["query.intr"] == ["Drug A"] and parsed["query.cond"] == ["Disease B"]
    assert parsed["pageSize"] == ["5"] and parsed["format"] == ["json"]


def test_search_with_no_matches_says_zero():
    fetcher = FakeFetcher(studies_json([], total_count=0))
    result = make_adapter(fetcher).search_with_provenance(ClinicalTrialsQuery(drug_name="x", disease_name="y"))
    assert result.records == [] and result.total_count == 0


def test_protocol_search_returns_a_plain_list_of_records():
    fetcher = FakeFetcher(studies_json([demo_study("NCT90000002")]))
    records = make_adapter(fetcher).search("Drug A", "Disease B", max_results=1)
    assert [r.nct_id for r in records] == ["NCT90000002"]


def test_network_failure_raises_instead_of_returning_an_empty_list():
    adapter = make_adapter(FakeFetcher(urllib.error.URLError("no route")))
    with pytest.raises(ClinicalTrialsRequestError, match="no route"):
        adapter.search("x", "y")
    with pytest.raises(ClinicalTrialsRequestError):
        make_adapter(FakeFetcher(TimeoutError("timed out"))).search("x", "y")


def test_rate_limit_is_surfaced_with_retry_after():
    error = urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests", {"Retry-After": "2"}, None)
    with pytest.raises(ClinicalTrialsRequestError) as raised:
        make_adapter(FakeFetcher(error)).search("x", "y")
    assert raised.value.status == 429 and raised.value.retry_after == "2"
    assert "Retry-After: 2" in str(raised.value)


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        ClinicalTrialsAdapter(timeout_s=0)
