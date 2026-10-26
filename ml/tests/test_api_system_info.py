"""GET /api/v1/system/info -- safe static metadata, no graph/model load (Task D)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api.app import app  # noqa: E402
from repurposemap.api.version import API_VERSION, APPLICATION_NAME  # noqa: E402


def test_system_info_reports_application_and_version():
    client = TestClient(app)
    response = client.get("/api/v1/system/info")

    assert response.status_code == 200
    body = response.json()
    assert body["application"] == APPLICATION_NAME
    assert body["api_version"] == API_VERSION


def test_system_info_reports_model_not_available_not_fabricated():
    # No model is configured yet -- this must say so honestly, never
    # guess or silently report a fake "ready" status.
    client = TestClient(app)
    body = client.get("/api/v1/system/info").json()

    assert body["model"]["available"] is False
    assert body["model"]["ready_for_inference"] is False
    assert body["model"]["model_type"] is None


def test_system_info_provenance_is_present_and_static():
    client = TestClient(app)
    body = client.get("/api/v1/system/info").json()

    provenance = body["data_provenance"]
    assert provenance is not None
    assert "PrimeKG" in provenance["source"]
    # No sha256/size_bytes field here -- those require reading the raw
    # file, which this endpoint must never do.
    assert "sha256" not in provenance
    assert "size_bytes" not in provenance


def test_system_info_keeps_inference_ineligible_without_a_configured_checkpoint():
    # A final-training experiment record exists (TransE), but no checkpoint is
    # explicitly configured, so system/info must keep reporting inference as
    # ineligible and the model as unavailable.
    client = TestClient(app)
    body = client.get("/api/v1/system/info").json()

    model_provenance = body["model_provenance"]
    assert model_provenance is not None
    assert model_provenance["inference_eligible"] is False
    assert model_provenance["most_recent_completed_stage"] is not None
    assert body["model"]["available"] is False
