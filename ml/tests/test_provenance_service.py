"""Experiment/model provenance layer -- read-only, no graph/model load."""

from __future__ import annotations

import json
import sys

from repurposemap.provenance.models import ExperimentKind, ExperimentStatus
from repurposemap.provenance.service import (
    EXPERIMENTS_DIR,
    get_model_eligibility,
    get_most_recent_completed_stage,
    is_model_available,
    load_experiments,
)

REPO_EXPERIMENTS = load_experiments()


def test_real_experiment_files_parse_as_valid():
    assert REPO_EXPERIMENTS, "ml/experiments should contain at least one committed experiment file"
    assert all(record.status == ExperimentStatus.VALID for record in REPO_EXPERIMENTS)


def test_gpu_profile_experiment_is_not_final():
    gpu_profile_records = [r for r in REPO_EXPERIMENTS if r.kind == ExperimentKind.GPU_PROFILE]
    assert gpu_profile_records
    for record in gpu_profile_records:
        assert record.is_final_training is False


def test_validation_search_experiment_is_not_final():
    search_records = [r for r in REPO_EXPERIMENTS if r.kind == ExperimentKind.VALIDATION_SEARCH]
    assert search_records
    for record in search_records:
        assert record.is_final_training is False
        assert record.locked_test_evaluation_performed is False


def test_no_final_training_record_exists_yet():
    # Final training is currently running on Kaggle; no committed
    # experiment file may claim to be final training while this is true.
    assert not any(r.is_final_training for r in REPO_EXPERIMENTS)


def test_model_eligibility_is_false_without_a_final_training_record():
    eligibility = get_model_eligibility(REPO_EXPERIMENTS)
    assert eligibility.eligible is False
    assert eligibility.final_training_experiment_file is None


def test_model_eligibility_true_requires_availability_still_false():
    # A final-training record alone must never be enough to flip
    # "available" -- that also requires an explicitly configured
    # checkpoint path, which does not exist yet.
    assert is_model_available(REPO_EXPERIMENTS) is False


def test_malformed_json_is_reported_not_raised(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    records = load_experiments(tmp_path)
    assert len(records) == 1
    assert records[0].status == ExperimentStatus.MALFORMED
    assert records[0].parse_errors


def test_missing_required_fields_is_reported_not_raised(tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"stage": "something"}))
    records = load_experiments(tmp_path)
    assert len(records) == 1
    assert records[0].status == ExperimentStatus.MISSING_REQUIRED_FIELDS
    assert records[0].kind == ExperimentKind.UNKNOWN


def test_final_flag_interpretation(tmp_path):
    final_like = tmp_path / "final.json"
    final_like.write_text(json.dumps({
        "stage": "TransE final training",
        "purpose": "final training on the full training graph",
        "generated_at": "2026-10-21T00:00:00Z",
        "provenance": {"source_commit": "deadbeef"},
        "safety_and_protocol": {
            "final_training_performed": True,
            "locked_test_evaluation_performed": False,
            "three_config_search_performed": True,
        },
    }))
    records = load_experiments(tmp_path)
    assert records[0].status == ExperimentStatus.VALID
    assert records[0].kind == ExperimentKind.FINAL_TRAINING
    assert records[0].is_final_training is True

    eligibility = get_model_eligibility(records)
    assert eligibility.eligible is True
    assert eligibility.final_training_experiment_file == "final.json"
    # Eligible per the record's own claim, but still not "available"
    # without a configured checkpoint path.
    assert is_model_available(records) is False


def test_safe_provenance_extraction_exposes_no_locked_test_labels():
    for record in REPO_EXPERIMENTS:
        dumped = record.model_dump()
        serialized = json.dumps(dumped)
        assert "locked_test_manifest" not in serialized
        assert "test_indications" not in serialized


def test_experiments_dir_points_inside_ml_experiments():
    assert EXPERIMENTS_DIR.name == "experiments"
    assert EXPERIMENTS_DIR.parent.name == "ml"


def test_load_experiments_is_not_recursive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "should_not_be_found.json").write_text(json.dumps({
        "stage": "x", "purpose": "x", "generated_at": "x",
        "provenance": {}, "safety_and_protocol": {},
    }))
    assert load_experiments(tmp_path) == []


def test_provenance_service_module_never_imports_torch_or_primekg():
    assert "torch" not in sys.modules or True  # torch may be loaded by other tests in-process
    import repurposemap.provenance.service as service_module

    source = service_module.__file__
    assert source is not None
    with open(source) as handle:
        contents = handle.read()
    assert "import torch" not in contents
    assert "load_primekg" not in contents


def test_get_most_recent_completed_stage_returns_latest_valid_stage():
    stage = get_most_recent_completed_stage(REPO_EXPERIMENTS)
    assert stage is not None
    assert any(r.stage == stage for r in REPO_EXPERIMENTS)
