"""POST /api/v1/disease-biology -- the adapter is injected via a FastAPI
dependency override, so these tests never touch the network. Gene
symbols and disease names below are invented (DEMO ...) and say nothing
real."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api.app import app  # noqa: E402
from repurposemap.api.disease_biology import get_open_targets_adapter  # noqa: E402
from repurposemap.biology import DiseaseBiologyModule, DiseaseMatch, OpenTargetsNotFoundError, OpenTargetsRequestError  # noqa: E402


class _StubAdapter:
    def __init__(self, *, matches=None, module=None, resolve_error=None, biology_error=None):
        self._matches, self._module = matches, module
        self._resolve_error, self._biology_error = resolve_error, biology_error
        self.get_disease_biology_calls = []

    def resolve_disease(self, query, max_results=5):
        if self._resolve_error is not None:
            raise self._resolve_error
        return self._matches

    def get_disease_biology(self, efo_id, max_targets=10):
        self.get_disease_biology_calls.append((efo_id, max_targets))
        if self._biology_error is not None:
            raise self._biology_error
        return self._module


def _override(adapter):
    return lambda: adapter


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def demo_module(disease_id="MONDO_0000001"):
    return DiseaseBiologyModule(
        disease_id=disease_id, disease_name="demo disease", disease_description="A demo disease.",
        total_associated_targets=1, targets=[], retrieved_at="2026-10-29T12:00:00Z",
    )


def test_lookup_by_disease_id_skips_resolution(client):
    stub = _StubAdapter(module=demo_module())
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)

    response = client.post("/api/v1/disease-biology", json={"disease_id": "MONDO_0000001"})
    assert response.status_code == 200
    body = response.json()
    assert body["module"]["disease_id"] == "MONDO_0000001"
    assert body["resolved_from"] is None
    assert stub.get_disease_biology_calls == [("MONDO_0000001", 10)]


def test_lookup_by_disease_name_resolves_first_and_reports_matches(client):
    matches = [DiseaseMatch(id="MONDO_0000001", name="demo disease", search_score=100.0)]
    stub = _StubAdapter(matches=matches, module=demo_module())
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)

    response = client.post("/api/v1/disease-biology", json={"disease_name": "demo disease"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolved_from"] == [{"id": "MONDO_0000001", "name": "demo disease", "search_score": 100.0}]
    assert stub.get_disease_biology_calls == [("MONDO_0000001", 10)]


def test_both_disease_name_and_disease_id_is_rejected(client):
    response = client.post("/api/v1/disease-biology", json={"disease_name": "x", "disease_id": "MONDO_1"})
    assert response.status_code == 422


def test_neither_disease_name_nor_disease_id_is_rejected(client):
    response = client.post("/api/v1/disease-biology", json={})
    assert response.status_code == 422


def test_name_resolving_to_nothing_is_a_404(client):
    stub = _StubAdapter(matches=[])
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/disease-biology", json={"disease_name": "not a real disease"})
    assert response.status_code == 404


def test_unknown_disease_id_is_a_404(client):
    stub = _StubAdapter(biology_error=OpenTargetsNotFoundError("no such disease"))
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/disease-biology", json={"disease_id": "MONDO_9999999"})
    assert response.status_code == 404


def test_a_real_lookup_failure_is_a_502_not_a_200_with_empty_results(client):
    stub = _StubAdapter(biology_error=OpenTargetsRequestError("Open Targets is down"))
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/disease-biology", json={"disease_id": "MONDO_0000001"})
    assert response.status_code == 502


def test_resolve_failure_is_also_a_502(client):
    stub = _StubAdapter(resolve_error=OpenTargetsRequestError("Open Targets is down"))
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/disease-biology", json={"disease_name": "x"})
    assert response.status_code == 502


def test_max_targets_is_forwarded(client):
    stub = _StubAdapter(module=demo_module())
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/disease-biology", json={"disease_id": "MONDO_0000001", "max_targets": 25})
    assert response.status_code == 200
    assert stub.get_disease_biology_calls == [("MONDO_0000001", 25)]


@pytest.mark.parametrize("max_targets", [0, 51])
def test_invalid_max_targets_is_rejected(client, max_targets):
    response = client.post("/api/v1/disease-biology", json={"disease_id": "MONDO_0000001", "max_targets": max_targets})
    assert response.status_code == 422


def test_disease_biology_endpoint_never_touches_primekg(client, monkeypatch):
    import repurposemap.adapters.primekg as primekg_module

    def _poisoned(*_args, **_kwargs):
        raise AssertionError("load_primekg/compute_file_fingerprint must never be called for the disease-biology route")

    monkeypatch.setattr(primekg_module, "load_primekg", _poisoned)
    monkeypatch.setattr(primekg_module, "compute_file_fingerprint", _poisoned)

    app.dependency_overrides[get_open_targets_adapter] = _override(_StubAdapter(module=demo_module()))
    response = client.post("/api/v1/disease-biology", json={"disease_id": "MONDO_0000001"})
    assert response.status_code == 200
