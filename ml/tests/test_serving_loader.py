"""repurposemap.serving.loader -- reading CandidateRankingBundle files
from an isolated fixture directory, never the real data/serving/.

All disease/drug names and IDs below are invented (DEMO ...) and say
nothing real about any actual disease, drug, or ranking."""

from __future__ import annotations

import json

import pytest

from repurposemap.serving.loader import ServingArtifactError, list_supported_diseases, load_bundle_for_disease

_VALID_BUNDLE = {
    "schema_version": 1,
    "disease_id": "DEMO:0001",
    "disease_name": "demo disease",
    "ranking_method": "path_quality",
    "provenance": {"ranking_method": "path_quality"},
    "generated_at": "2026-11-05T00:00:00+00:00",
    "candidate_universe": {"description": "demo universe", "size": 1},
    "candidates": [{"drug_id": "DEMO:drug-1", "drug_name": "demo drug one", "rank": 1, "score": 0.9}],
}


def test_load_bundle_for_disease_returns_none_when_no_file_exists(tmp_path):
    assert load_bundle_for_disease("DEMO:missing", serving_dir=tmp_path) is None


def test_load_bundle_for_disease_returns_none_when_directory_does_not_exist(tmp_path):
    assert load_bundle_for_disease("DEMO:0001", serving_dir=tmp_path / "does-not-exist") is None


def test_load_bundle_for_disease_loads_a_valid_file(tmp_path):
    (tmp_path / "DEMO_0001.json").write_text(json.dumps(_VALID_BUNDLE))
    bundle = load_bundle_for_disease("DEMO_0001", serving_dir=tmp_path)
    assert bundle is not None
    assert bundle.disease_name == "demo disease"
    assert len(bundle.candidates) == 1


def test_load_bundle_for_disease_raises_on_malformed_json(tmp_path):
    (tmp_path / "DEMO_0001.json").write_text("{not valid json")
    with pytest.raises(ServingArtifactError):
        load_bundle_for_disease("DEMO_0001", serving_dir=tmp_path)


def test_load_bundle_for_disease_raises_on_schema_violation(tmp_path):
    broken = dict(_VALID_BUNDLE)
    broken["candidates"] = []
    (tmp_path / "DEMO_0001.json").write_text(json.dumps(broken))
    with pytest.raises(ServingArtifactError, match="empty candidate list"):
        load_bundle_for_disease("DEMO_0001", serving_dir=tmp_path)


def test_list_supported_diseases_is_empty_for_an_empty_directory(tmp_path):
    assert list_supported_diseases(serving_dir=tmp_path) == []


def test_list_supported_diseases_skips_malformed_files_rather_than_raising(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_VALID_BUNDLE))
    (tmp_path / "bad.json").write_text("{not valid json")
    bundles = list_supported_diseases(serving_dir=tmp_path)
    assert len(bundles) == 1
    assert bundles[0].disease_name == "demo disease"


def test_list_supported_diseases_sorts_by_disease_name(tmp_path):
    b_bundle = dict(_VALID_BUNDLE, disease_id="DEMO:b", disease_name="zzz disease")
    a_bundle = dict(_VALID_BUNDLE, disease_id="DEMO:a", disease_name="aaa disease")
    (tmp_path / "b.json").write_text(json.dumps(b_bundle))
    (tmp_path / "a.json").write_text(json.dumps(a_bundle))
    names = [bundle.disease_name for bundle in list_supported_diseases(serving_dir=tmp_path)]
    assert names == ["aaa disease", "zzz disease"]
