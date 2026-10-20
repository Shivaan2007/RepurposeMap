"""GET /health -- liveness only (Task D)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api.app import app  # noqa: E402
from repurposemap.api.health import SERVICE_NAME  # noqa: E402
from repurposemap.api.version import API_VERSION  # noqa: E402


def test_health_returns_ok_with_service_and_version():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"service": SERVICE_NAME, "status": "ok", "api_version": API_VERSION}
