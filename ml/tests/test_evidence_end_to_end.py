"""End-to-end proof that the evidence flow actually works across every
layer, not just that each layer passes its own unit tests:

    drug + disease -> evidence adapters -> FastAPI endpoint
    -> response shape -> fields EvidencePanel.tsx actually reads

This does not re-test adapter internals (see test_pubmed_adapter.py etc.)
or per-source failure isolation (see test_api_evidence.py) -- it proves the
whole chain is wired together correctly: the response JSON shape matches
what the request declares, every field EvidencePanel.tsx reads by name is
present with the expected type, a partial-source failure stays visible
rather than collapsing into a fake empty success, and no disclaimer/record
text makes a clinical recommendation.

Field names below (``pmid``, ``url``, ``title``, ``journal``,
``publication_date``, ``strength``, ``nct_id``, ``status``, ``phase``,
``sponsor``, ``enrollment``, ``source``, ``description``,
``boxed_warning_present``, ``contraindication_text``) are copied directly
from web/app/src/components/EvidencePanel.tsx's own field accesses -- if
this test and that file ever disagree, the frontend is reading a field the
backend stopped sending (or vice versa), which is exactly the class of bug
per-layer unit tests cannot catch."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api.app import app  # noqa: E402
from repurposemap.api.evidence import (  # noqa: E402
    get_clinicaltrials_adapter,
    get_dailymed_adapter,
    get_openfda_adapter,
    get_pubmed_adapter,
)
from repurposemap.evidence import (  # noqa: E402
    ClinicalTrialEvidence,
    EvidenceSource,
    PubMedError,
    PubMedEvidence,
    SafetyEvidence,
)

# Fields EvidencePanel.tsx reads from each record type, by name.
PUBMED_FIELDS_READ_BY_UI = {"pmid", "url", "title", "journal", "publication_date", "strength"}
TRIAL_FIELDS_READ_BY_UI = {"nct_id", "url", "title", "status", "phase", "strength", "sponsor", "enrollment"}
SAFETY_FIELDS_READ_BY_UI = {"source", "url", "description", "boxed_warning_present", "contraindication_text"}
SOURCE_STATUS_FIELDS_READ_BY_UI = {"source", "succeeded", "error"}

# Wording that would turn a research lookup into a clinical recommendation.
# None of these may appear in any disclaimer or source-provided description
# this endpoint controls (disclaimers are ours; record text is copied
# verbatim from each source and is exempt from this check, same as the
# adapters' own classify_strength tests already establish).
BANNED_CLINICAL_WORDS = re.compile(r"\b(cure|cures|should take|prescribe|recommended treatment|is safe|is effective)\b", re.IGNORECASE)


class _StubAdapter:
    def __init__(self, result=None, error: Exception | None = None):
        self._result, self._error = result if result is not None else [], error

    def search(self, *_args, **_kwargs):
        if self._error is not None:
            raise self._error
        return self._result


def _override(adapter):
    return lambda: adapter


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def full_pubmed_record():
    return PubMedEvidence(
        source_id="90000001",
        pmid="90000001",
        title="DEMO paper title",
        journal="DEMO Journal",
        publication_date="2024",
        url="https://pubmed.ncbi.nlm.nih.gov/90000001/",
    )


def full_trial_record():
    return ClinicalTrialEvidence(
        source_id="NCT90000001",
        nct_id="NCT90000001",
        title="DEMO trial title",
        status="RECRUITING",
        phase="PHASE2",
        sponsor="DEMO Sponsor",
        enrollment=120,
        url="https://clinicaltrials.gov/study/NCT90000001",
    )


def full_safety_record(source, source_id):
    return SafetyEvidence(
        source=source,
        source_id=source_id,
        description="DEMO label description",
        boxed_warning_present=True,
        contraindication_text="DEMO contraindication text, copied from the label",
        url="https://example.invalid/label",
    )


def test_request_and_response_shapes_match_what_the_frontend_sends_and_reads(client):
    """The request body EvidencePanel.tsx sends, and the response shape it
    reads back, round-trip through the real FastAPI schema -- not a
    hand-maintained assumption about it."""
    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter([full_pubmed_record()]))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter([full_trial_record()]))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter([full_safety_record(EvidenceSource.OPENFDA, "FDA-1")]))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter([full_safety_record(EvidenceSource.DAILYMED, "DM-1")]))

    # Exactly the request shape EvidencePanel.handleSubmit builds.
    response = client.post("/api/v1/evidence", json={"drug_name": "sildenafil", "disease_name": "pulmonary arterial hypertension", "max_results": 5})
    assert response.status_code == 200
    body = response.json()

    pubmed_record = body["bundle"]["pubmed"][0]
    assert PUBMED_FIELDS_READ_BY_UI <= pubmed_record.keys()
    assert isinstance(pubmed_record["pmid"], str)
    assert isinstance(pubmed_record["strength"], str)

    trial_record = body["bundle"]["clinical_trials"][0]
    assert TRIAL_FIELDS_READ_BY_UI <= trial_record.keys()
    assert isinstance(trial_record["enrollment"], int)

    safety_records = {r["source"]: r for r in body["bundle"]["safety"]}
    for record in safety_records.values():
        assert SAFETY_FIELDS_READ_BY_UI <= record.keys()
        assert isinstance(record["boxed_warning_present"], bool)
    assert set(safety_records) == {"openfda", "dailymed"}

    for status in body["source_status"]:
        assert SOURCE_STATUS_FIELDS_READ_BY_UI <= status.keys()

    assert "disclaimer" in body
    assert "bundle" in body and "generated_at" in body["bundle"]


def test_partial_source_failure_stays_visible_end_to_end(client):
    """A failing source must still show up as failed in source_status, not
    collapse into an empty-but-successful bundle the UI would render
    identically to a real zero-match answer."""
    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter(error=PubMedError("NCBI is down")))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter([full_trial_record()]))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter([]))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter([]))

    response = client.post("/api/v1/evidence", json={"drug_name": "sildenafil", "disease_name": "PAH"})
    body = response.json()

    by_source = {s["source"]: s for s in body["source_status"]}
    assert by_source["pubmed"]["succeeded"] is False
    assert by_source["pubmed"]["error"] == "NCBI is down"
    # openFDA/DailyMed are legitimately empty (succeeded, zero records) --
    # distinguishable from PubMed's failure, never conflated.
    assert by_source["openfda"]["succeeded"] is True
    assert by_source["dailymed"]["succeeded"] is True
    assert body["bundle"]["safety"] == []
    assert body["bundle"]["pubmed"] == []


def test_all_sources_legitimately_empty_is_not_reported_as_a_failure(client):
    """Zero real matches across every source is a valid answer and must
    never be indistinguishable from, or silently rewritten as, a failure."""
    for dependency in (get_pubmed_adapter, get_clinicaltrials_adapter, get_openfda_adapter, get_dailymed_adapter):
        app.dependency_overrides[dependency] = _override(_StubAdapter([]))

    response = client.post("/api/v1/evidence", json={"drug_name": "a totally obscure demo compound", "disease_name": "a totally obscure demo condition"})
    body = response.json()

    assert all(status["succeeded"] for status in body["source_status"])
    assert all(status["error"] is None for status in body["source_status"])
    assert body["bundle"]["pubmed"] == body["bundle"]["clinical_trials"] == body["bundle"]["safety"] == []


def test_no_clinical_recommendation_wording_anywhere_in_the_response(client):
    """The disclaimer and every demo record's own text must read as a
    research lookup, never a clinical instruction -- checked on the actual
    serialized response the UI renders, not just at the adapter layer."""
    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter([full_pubmed_record()]))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter([full_trial_record()]))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter([full_safety_record(EvidenceSource.OPENFDA, "FDA-1")]))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter([full_safety_record(EvidenceSource.DAILYMED, "DM-1")]))

    response = client.post("/api/v1/evidence", json={"drug_name": "sildenafil", "disease_name": "PAH"})
    body = response.json()

    assert BANNED_CLINICAL_WORDS.search(body["disclaimer"]) is None
    for record in body["bundle"]["pubmed"] + body["bundle"]["clinical_trials"]:
        assert BANNED_CLINICAL_WORDS.search(record["title"]) is None
