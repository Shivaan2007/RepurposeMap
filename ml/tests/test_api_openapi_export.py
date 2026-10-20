"""OpenAPI generation and the export script (Task B/D).

Not required by the repo's test-collection convention to need a live
server: ``app.openapi()`` builds the schema purely from route/model
definitions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from repurposemap.api.app import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "ml" / "scripts" / "export_openapi.py"

EXPECTED_SCHEMAS = {
    "HealthResponse", "SystemInfoResponse", "ModelStatus", "Provenance",
    "DiseaseQuery", "CandidateDrug", "CandidateScore", "EvidenceSummary",
    "ExplanationPath", "ExplanationStep", "SafetyFlag", "RankingResponse",
}


def test_openapi_schema_is_valid_and_deterministic():
    schema_a = app.openapi()
    schema_b = app.openapi()
    assert schema_a == schema_b  # same route/model definitions -> identical schema

    assert schema_a["openapi"].startswith("3.")
    assert "/health" in schema_a["paths"]
    assert "/api/v1/system/info" in schema_a["paths"]
    assert "/api/v1/rank" in schema_a["paths"]


def test_openapi_schema_contains_every_contract_model():
    schema = app.openapi()
    defined = set(schema["components"]["schemas"])
    missing = EXPECTED_SCHEMAS - defined
    assert not missing, f"OpenAPI schema is missing contract model(s): {missing}"


def test_rank_endpoint_never_fabricates_a_result():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/api/v1/rank", json={"disease_name": "some disease", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_available"
    assert body["candidates"] == []  # never a fabricated/mock ranking


def test_export_openapi_script_writes_a_valid_schema(tmp_path):
    output_path = tmp_path / "openapi.json"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "ml" / "src")}
    result = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--output", str(output_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.is_file()
    schema = json.loads(output_path.read_text())
    assert schema["openapi"].startswith("3.")
    assert "/health" in schema["paths"]
