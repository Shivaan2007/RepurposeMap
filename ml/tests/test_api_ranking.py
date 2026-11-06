"""POST /api/v1/rank and GET /api/v1/rank/diseases -- backed by the
precomputed serving-bundle mechanism, monkeypatched to an isolated
fixture directory so these tests never touch the real (empty)
data/serving/. All disease/drug names and IDs below are invented
(DEMO ...) and say nothing real about any actual disease, drug, or
ranking."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api import ranking as ranking_module  # noqa: E402
from repurposemap.api.app import app  # noqa: E402

client = TestClient(app)

_BUNDLE = {
    "schema_version": 1,
    "disease_id": "DEMO:0001",
    "disease_name": "demo disease",
    "ranking_method": "path_quality",
    "provenance": {"ranking_method": "path_quality"},
    "generated_at": "2026-11-05T00:00:00+00:00",
    "candidate_universe": {"description": "demo universe", "size": 1},
    "candidates": [
        {
            "drug_id": "DEMO:drug-1",
            "drug_name": "demo drug one",
            "rank": 1,
            "score": 0.9,
            "explanation_paths": [
                {
                    "steps": [
                        {
                            "source_id": "DEMO:drug-1",
                            "source_name": "demo drug one",
                            "source_type": "drug",
                            "relation": "demo_relation",
                            "target_id": "DEMO:0001",
                            "target_name": "demo disease",
                            "target_type": "disease",
                        }
                    ],
                    "path_score": 0.7,
                }
            ],
        }
    ],
}


@pytest.fixture
def serving_dir(tmp_path, monkeypatch):
    (tmp_path / "DEMO_0001.json").write_text(json.dumps(_BUNDLE))
    monkeypatch.setattr(ranking_module, "list_supported_diseases", lambda: _real_list(tmp_path))
    monkeypatch.setattr(ranking_module, "load_bundle_for_disease", lambda disease_id, serving_dir=None: _real_load(disease_id, tmp_path))
    return tmp_path


def _real_list(directory):
    from repurposemap.serving.loader import list_supported_diseases as impl

    return impl(serving_dir=directory)


def _real_load(disease_id, directory):
    from repurposemap.serving.loader import load_bundle_for_disease as impl

    return impl(disease_id, serving_dir=directory)


def test_rank_reports_not_available_when_no_bundle_exists_for_the_disease():
    response = client.post("/api/v1/rank", json={"disease_name": "an unseeded disease", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_available"
    assert body["serving_mode"] == "not_available"
    assert body["candidates"] == []


def test_rank_by_disease_name_returns_the_precomputed_bundle(serving_dir):
    response = client.post("/api/v1/rank", json={"disease_name": "demo disease", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "ready"
    assert body["serving_mode"] == "precomputed"
    assert body["ranking_method"] == "path_quality"
    assert body["candidate_universe"] == "demo universe"
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["drug_name"] == "demo drug one"
    assert body["candidates"][0]["explanation_paths"][0]["path_score"] == 0.7


def test_rank_by_disease_name_is_case_insensitive_exact_match(serving_dir):
    response = client.post("/api/v1/rank", json={"disease_name": "DEMO DISEASE", "top_k": 5})
    assert response.json()["model_status"] == "ready"


def test_rank_by_disease_id_returns_the_precomputed_bundle(serving_dir):
    response = client.post("/api/v1/rank", json={"disease_id": "DEMO_0001", "top_k": 5})
    assert response.json()["model_status"] == "ready"


def test_rank_never_fabricates_candidates_for_an_unrelated_disease_name(serving_dir):
    response = client.post("/api/v1/rank", json={"disease_name": "a completely different disease", "top_k": 5})
    body = response.json()
    assert body["model_status"] == "not_available"
    assert body["candidates"] == []


def test_rank_diseases_lists_supported_diseases(serving_dir):
    response = client.get("/api/v1/rank/diseases")
    assert response.status_code == 200
    body = response.json()
    assert body["diseases"] == [
        {"disease_id": "DEMO:0001", "disease_name": "demo disease", "ranking_method": "path_quality", "candidate_count": 1}
    ]


def test_rank_diseases_is_empty_when_no_bundles_exist():
    response = client.get("/api/v1/rank/diseases")
    assert response.json()["diseases"] == []
