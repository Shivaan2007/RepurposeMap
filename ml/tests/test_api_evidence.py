"""POST /api/v1/evidence -- adapters are injected via FastAPI dependency
overrides, so these tests never touch the network. Drug/disease names and
record contents below are invented (DEMO ...) and say nothing real."""

from __future__ import annotations

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
    ClinicalTrialsError,
    DailyMedError,
    OpenFDAError,
    PubMedError,
    PubMedEvidence,
    SafetyEvidence,
)


class _StubAdapter:
    """A minimal stand-in matching each real adapter's ``search`` signature."""

    def __init__(self, result=None, error: Exception | None = None):
        self._result, self._error = result if result is not None else [], error

    def search(self, *_args, **_kwargs):
        if self._error is not None:
            raise self._error
        return self._result


def _override(adapter) -> callable:
    return lambda: adapter


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def demo_pubmed(pmid="90000001"):
    return PubMedEvidence(source_id=pmid, pmid=pmid, title="DEMO paper", url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")


def demo_trial(nct_id="NCT90000001"):
    return ClinicalTrialEvidence(source_id=nct_id, nct_id=nct_id, title="DEMO trial")


def demo_safety(source, source_id="DEMO-1"):
    return SafetyEvidence(source=source, source_id=source_id, description="DEMO safety record")


def test_all_sources_succeed_and_are_merged_into_one_bundle(client):
    from repurposemap.evidence import EvidenceSource

    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter([demo_pubmed()]))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter([demo_trial()]))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter([demo_safety(EvidenceSource.OPENFDA, "FDA-1")]))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter([demo_safety(EvidenceSource.DAILYMED, "DM-1")]))

    response = client.post("/api/v1/evidence", json={"drug_name": "Drug A", "disease_name": "Disease B"})
    assert response.status_code == 200
    body = response.json()

    assert [r["pmid"] for r in body["bundle"]["pubmed"]] == ["90000001"]
    assert [r["nct_id"] for r in body["bundle"]["clinical_trials"]] == ["NCT90000001"]
    assert sorted(r["source_id"] for r in body["bundle"]["safety"]) == ["DM-1", "FDA-1"]
    assert body["bundle"]["generated_at"] is not None
    assert {s["source"] for s in body["source_status"]} == {"pubmed", "clinical_trials", "openfda", "dailymed"}
    assert all(s["succeeded"] for s in body["source_status"])
    assert all(s["error"] is None for s in body["source_status"])


def test_one_source_failing_does_not_drop_the_others(client):
    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter(error=PubMedError("NCBI is down")))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter([demo_trial()]))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter([]))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter([]))

    response = client.post("/api/v1/evidence", json={"drug_name": "Drug A", "disease_name": "Disease B"})
    assert response.status_code == 200
    body = response.json()

    assert body["bundle"]["pubmed"] == []  # failed, not just empty
    assert [r["nct_id"] for r in body["bundle"]["clinical_trials"]] == ["NCT90000001"]  # still returned

    by_source = {s["source"]: s for s in body["source_status"]}
    assert by_source["pubmed"] == {"source": "pubmed", "succeeded": False, "error": "NCBI is down"}
    assert by_source["clinical_trials"]["succeeded"] is True


def test_every_source_failing_still_returns_200_with_all_failures_reported(client):
    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter(error=PubMedError("down")))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter(error=ClinicalTrialsError("down")))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter(error=OpenFDAError("down")))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter(error=DailyMedError("down")))

    response = client.post("/api/v1/evidence", json={"drug_name": "Drug A", "disease_name": "Disease B"})
    assert response.status_code == 200
    body = response.json()

    assert body["bundle"]["pubmed"] == [] and body["bundle"]["clinical_trials"] == [] and body["bundle"]["safety"] == []
    assert all(not s["succeeded"] and s["error"] == "down" for s in body["source_status"])


def test_max_results_is_forwarded_to_every_adapter(client):
    seen = {}

    class RecordingAdapter:
        def __init__(self, name):
            self._name = name

        def search(self, *args, **kwargs):
            seen[self._name] = kwargs.get("max_results")
            return []

    app.dependency_overrides[get_pubmed_adapter] = _override(RecordingAdapter("pubmed"))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(RecordingAdapter("clinicaltrials"))
    app.dependency_overrides[get_openfda_adapter] = _override(RecordingAdapter("openfda"))
    app.dependency_overrides[get_dailymed_adapter] = _override(RecordingAdapter("dailymed"))

    response = client.post("/api/v1/evidence", json={"drug_name": "Drug A", "disease_name": "Disease B", "max_results": 3})
    assert response.status_code == 200
    assert seen == {"pubmed": 3, "clinicaltrials": 3, "openfda": 3, "dailymed": 3}


@pytest.mark.parametrize(
    "payload",
    [
        {"drug_name": "Drug A"},  # missing disease_name
        {"disease_name": "Disease B"},  # missing drug_name
        {"drug_name": "Drug A", "disease_name": "Disease B", "max_results": 0},
        {"drug_name": "Drug A", "disease_name": "Disease B", "max_results": 21},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/api/v1/evidence", json=payload).status_code == 422


def test_evidence_endpoint_never_touches_primekg(client, monkeypatch):
    import repurposemap.adapters.primekg as primekg_module

    def _poisoned(*_args, **_kwargs):
        raise AssertionError("load_primekg/compute_file_fingerprint must never be called for the evidence route")

    monkeypatch.setattr(primekg_module, "load_primekg", _poisoned)
    monkeypatch.setattr(primekg_module, "compute_file_fingerprint", _poisoned)

    app.dependency_overrides[get_pubmed_adapter] = _override(_StubAdapter([]))
    app.dependency_overrides[get_clinicaltrials_adapter] = _override(_StubAdapter([]))
    app.dependency_overrides[get_openfda_adapter] = _override(_StubAdapter([]))
    app.dependency_overrides[get_dailymed_adapter] = _override(_StubAdapter([]))

    response = client.post("/api/v1/evidence", json={"drug_name": "Drug A", "disease_name": "Disease B"})
    assert response.status_code == 200
