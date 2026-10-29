"""POST /api/v1/target-scorecard -- the adapter is injected via a FastAPI
dependency override, so these tests never touch the network. Gene
symbols below are invented (DEMO ...) and say nothing real."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api.app import app  # noqa: E402
from repurposemap.api.target_scorecard import get_open_targets_adapter  # noqa: E402
from repurposemap.biology import OpenTargetsNotFoundError, OpenTargetsRequestError, TargetMatch, TargetScorecard  # noqa: E402


class _StubAdapter:
    def __init__(self, *, matches=None, scorecard=None, resolve_error=None, scorecard_error=None):
        self._matches, self._scorecard = matches, scorecard
        self._resolve_error, self._scorecard_error = resolve_error, scorecard_error
        self.get_target_scorecard_calls = []

    def resolve_target(self, query, max_results=5):
        if self._resolve_error is not None:
            raise self._resolve_error
        return self._matches

    def get_target_scorecard(self, ensembl_id):
        self.get_target_scorecard_calls.append(ensembl_id)
        if self._scorecard_error is not None:
            raise self._scorecard_error
        return self._scorecard


def _override(adapter):
    return lambda: adapter


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def demo_scorecard(ensembl_id="ENSG_DEMO_1"):
    return TargetScorecard(
        ensembl_id=ensembl_id, approved_symbol="DEMOG1", approved_name="demo gene 1",
        tractability=[], prioritisation_factors={}, top_tissue_expression=[], retrieved_at="2026-10-29T12:00:00Z",
    )


def test_lookup_by_ensembl_id_skips_resolution(client):
    stub = _StubAdapter(scorecard=demo_scorecard())
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)

    response = client.post("/api/v1/target-scorecard", json={"ensembl_id": "ENSG_DEMO_1"})
    assert response.status_code == 200
    body = response.json()
    assert body["scorecard"]["ensembl_id"] == "ENSG_DEMO_1"
    assert body["resolved_from"] is None
    assert stub.get_target_scorecard_calls == ["ENSG_DEMO_1"]


def test_lookup_by_gene_symbol_resolves_first_and_reports_matches(client):
    matches = [TargetMatch(id="ENSG_DEMO_1", name="DEMOG1", search_score=100.0)]
    stub = _StubAdapter(matches=matches, scorecard=demo_scorecard())
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)

    response = client.post("/api/v1/target-scorecard", json={"gene_symbol": "DEMOG1"})
    assert response.status_code == 200
    body = response.json()
    assert body["resolved_from"] == [{"id": "ENSG_DEMO_1", "name": "DEMOG1", "search_score": 100.0}]
    assert stub.get_target_scorecard_calls == ["ENSG_DEMO_1"]


def test_both_gene_symbol_and_ensembl_id_is_rejected(client):
    response = client.post("/api/v1/target-scorecard", json={"gene_symbol": "x", "ensembl_id": "ENSG_1"})
    assert response.status_code == 422


def test_neither_gene_symbol_nor_ensembl_id_is_rejected(client):
    response = client.post("/api/v1/target-scorecard", json={})
    assert response.status_code == 422


def test_symbol_resolving_to_nothing_is_a_404(client):
    stub = _StubAdapter(matches=[])
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/target-scorecard", json={"gene_symbol": "not a real gene"})
    assert response.status_code == 404


def test_unknown_ensembl_id_is_a_404(client):
    stub = _StubAdapter(scorecard_error=OpenTargetsNotFoundError("no such target"))
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/target-scorecard", json={"ensembl_id": "ENSG_FAKE"})
    assert response.status_code == 404


def test_a_real_lookup_failure_is_a_502_not_a_200_with_an_empty_scorecard(client):
    stub = _StubAdapter(scorecard_error=OpenTargetsRequestError("Open Targets is down"))
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/target-scorecard", json={"ensembl_id": "ENSG_DEMO_1"})
    assert response.status_code == 502


def test_resolve_failure_is_also_a_502(client):
    stub = _StubAdapter(resolve_error=OpenTargetsRequestError("Open Targets is down"))
    app.dependency_overrides[get_open_targets_adapter] = _override(stub)
    response = client.post("/api/v1/target-scorecard", json={"gene_symbol": "x"})
    assert response.status_code == 502


def test_target_scorecard_endpoint_never_touches_primekg(client, monkeypatch):
    import repurposemap.adapters.primekg as primekg_module

    def _poisoned(*_args, **_kwargs):
        raise AssertionError("load_primekg/compute_file_fingerprint must never be called for the target-scorecard route")

    monkeypatch.setattr(primekg_module, "load_primekg", _poisoned)
    monkeypatch.setattr(primekg_module, "compute_file_fingerprint", _poisoned)

    app.dependency_overrides[get_open_targets_adapter] = _override(_StubAdapter(scorecard=demo_scorecard()))
    response = client.post("/api/v1/target-scorecard", json={"ensembl_id": "ENSG_DEMO_1"})
    assert response.status_code == 200
