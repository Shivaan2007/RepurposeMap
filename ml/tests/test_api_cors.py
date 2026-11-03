"""CORS configuration (deployment-readiness task)."""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import repurposemap.api.app as app_module  # noqa: E402
from repurposemap.api import cors as cors_module  # noqa: E402


def _build_app_with_origins(monkeypatch, origins_env: str | None):
    if origins_env is None:
        monkeypatch.delenv(cors_module.CORS_ORIGINS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(cors_module.CORS_ORIGINS_ENV_VAR, origins_env)
    reloaded = importlib.reload(sys.modules["repurposemap.api.app"])
    return reloaded.app


def test_get_allowed_origins_defaults_to_dev_origins_when_unset(monkeypatch):
    monkeypatch.delenv(cors_module.CORS_ORIGINS_ENV_VAR, raising=False)
    assert cors_module.get_allowed_origins() == list(cors_module.DEFAULT_DEV_ORIGINS)


def test_get_allowed_origins_parses_comma_separated_list(monkeypatch):
    monkeypatch.setenv(
        cors_module.CORS_ORIGINS_ENV_VAR,
        "https://app.example.com, https://staging.example.com ,,",
    )
    assert cors_module.get_allowed_origins() == [
        "https://app.example.com",
        "https://staging.example.com",
    ]


def test_allowed_origin_receives_cors_headers(monkeypatch):
    app = _build_app_with_origins(monkeypatch, "https://app.example.com")
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "https://app.example.com"})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_disallowed_origin_does_not_receive_cors_headers(monkeypatch):
    app = _build_app_with_origins(monkeypatch, "https://app.example.com")
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_no_origin_header_still_works_without_cors_headers(monkeypatch):
    app = _build_app_with_origins(monkeypatch, "https://app.example.com")
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_credentials_are_never_enabled(monkeypatch):
    # No cookies/credentialed requests are used by this API -- allow_credentials
    # must stay False regardless of configured origins, since "*" + credentials
    # is never a safe combination.
    app = _build_app_with_origins(monkeypatch, "https://app.example.com")
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "https://app.example.com"})

    assert "access-control-allow-credentials" not in response.headers


def test_dev_default_allows_vite_dev_server_origin(monkeypatch):
    app = _build_app_with_origins(monkeypatch, None)
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_evidence_endpoint_reachable_cross_origin(monkeypatch):
    # Local cross-origin smoke test (Task 4): reproduces what a browser
    # sends when the frontend and API are on different origins, without
    # needing two real bound sockets (this sandbox blocks local port
    # binding). Adapters are stubbed via dependency overrides -- this
    # proves CORS + routing work together, not adapter behavior, which is
    # already covered by test_api_evidence.py and
    # test_evidence_end_to_end.py.
    app = _build_app_with_origins(monkeypatch, "http://localhost:5173")

    from repurposemap.api.evidence import (
        get_clinicaltrials_adapter,
        get_dailymed_adapter,
        get_openfda_adapter,
        get_pubmed_adapter,
    )

    class _EmptyAdapter:
        def search(self, *_args, **_kwargs):
            return []

    for dependency in (
        get_pubmed_adapter,
        get_clinicaltrials_adapter,
        get_openfda_adapter,
        get_dailymed_adapter,
    ):
        app.dependency_overrides[dependency] = lambda: _EmptyAdapter()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/evidence",
                json={"drug_name": "Drug A", "disease_name": "Disease B"},
                headers={"Origin": "http://localhost:5173"},
            )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    finally:
        app.dependency_overrides.clear()
