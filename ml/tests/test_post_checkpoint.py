"""scripts/post_checkpoint.py -- the pure decision logic: what counts as the frozen final
checkpoint, the exactly-once locked-test guard, and the experiment record it freezes.
Synthetic dicts only; no model, no locked-test data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import post_checkpoint as pc  # noqa: E402


def good_checkpoint() -> dict:
    return {
        "is_final_training": True,
        "epoch": 2,
        "config": {**pc.FROZEN_CONFIG, "epochs": 3, "validate_every": None},
        "hyperparameter_search": {"selected_name": "rgcn-wide"},
        "export_fingerprint": pc.FROZEN_EXPORT_FINGERPRINT,
        "primekg_fingerprint": {"sha256": pc.FROZEN_PRIMEKG_SHA256},
        "locked_test_manifest_fingerprint": "abc",
        "git_commit": pc.FROZEN_SOURCE_COMMIT,
        "train_loss_curve": [0.1, 0.08, 0.07],
    }


def good_provenance() -> dict:
    return {
        "epochs": 3, "final_epoch_reached": 2, "stopped_for_time_budget": False, "is_final_training": True,
        "selected_name": "rgcn-wide", "source_commit": pc.FROZEN_SOURCE_COMMIT,
        "source_archive_sha256": pc.FROZEN_ARCHIVE_SHA256, "locked_test_evaluation_performed": False,
    }


def test_the_frozen_checkpoint_verifies():
    assert pc.check_checkpoint(good_checkpoint(), 2, good_provenance()) == []
    assert pc.check_checkpoint(good_checkpoint(), 2) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(is_final_training=False),
        lambda c: c.update(epoch=1),
        lambda c: c["config"].update(hidden_dim=64),
        lambda c: c["config"].update(learning_rate=0.01),
        lambda c: c["config"].update(fanout_hop1=10),
        lambda c: c["config"].update(early_stopping_patience=2),
        lambda c: c["config"].update(epochs=2),
        lambda c: c["hyperparameter_search"].update(selected_name="rgcn-base"),
        lambda c: c.update(export_fingerprint="0" * 64),
        lambda c: c.update(primekg_fingerprint={"sha256": "0" * 64}),
        lambda c: c.update(locked_test_manifest_fingerprint=None),
        lambda c: c.update(git_commit="deadbeef"),
        lambda c: c.update(train_loss_curve=[0.1, float("nan"), 0.07]),
        lambda c: c.update(train_loss_curve=[0.1, 0.08]),
    ],
)
def test_any_deviation_from_the_frozen_spec_fails_verification(mutate):
    checkpoint = good_checkpoint()
    mutate(checkpoint)
    assert pc.check_checkpoint(checkpoint, 2)


def test_a_wrong_final_epoch_argument_fails_even_if_the_checkpoint_agrees_with_it():
    checkpoint = good_checkpoint()
    checkpoint["epoch"] = 1
    assert pc.check_checkpoint(checkpoint, 1)  # 2 epochs is not the frozen 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(stopped_for_time_budget=True),
        lambda p: p.update(final_epoch_reached=1),
        lambda p: p.update(source_commit="deadbeef"),
        lambda p: p.update(source_archive_sha256="0" * 64),
        lambda p: p.update(locked_test_evaluation_performed=True),
        lambda p: p.update(selected_name="rgcn-base"),
    ],
)
def test_the_colab_provenance_file_is_checked_against_the_frozen_spec_too(mutate):
    provenance = good_provenance()
    mutate(provenance)
    assert pc.check_checkpoint(good_checkpoint(), 2, provenance)


def test_a_missing_git_commit_is_not_a_failure_but_a_wrong_one_is():
    checkpoint = good_checkpoint()
    checkpoint["git_commit"] = None
    assert pc.check_checkpoint(checkpoint, 2) == []


def test_the_evaluation_runs_once(tmp_path):
    experiments, marker = tmp_path / "experiments", tmp_path / "results" / ".started"
    experiments.mkdir()
    assert pc.guard_single_evaluation(experiments, marker, False) is None

    marker.parent.mkdir()
    marker.write_text("{}")  # started, never finished
    assert "started" in pc.guard_single_evaluation(experiments, marker, False)
    assert pc.guard_single_evaluation(experiments, marker, True) is None  # explicit, loud override

    (experiments / "rgcn_final_locked_test_abc12345.json").write_text("{}")
    assert "exactly once" in pc.guard_single_evaluation(experiments, marker, True)  # a result always blocks


def test_the_frozen_record_follows_the_experiment_schema_and_keeps_the_flat_metrics(tmp_path):
    from repurposemap.provenance.models import ExperimentKind, ExperimentStatus
    from repurposemap.provenance.service import load_experiments

    raw = {"split": "locked-test", "n_pairs": 498, "hits_at_1": 0.01, "hits_at_3": 0.02, "hits_at_10": 0.05,
           "mrr": 0.03, "macro_auprc": 0.04, "micro_auprc": 0.02}
    record = pc.build_experiment_record(raw, checkpoint=good_checkpoint(), checkpoint_sha256="f" * 64, final_epoch=2,
                                        generated_at="2026-11-26T00:00:00Z")
    (tmp_path / "rgcn_final_locked_test_ffffffff.json").write_text(json.dumps(record))
    (loaded,) = load_experiments(tmp_path)
    assert loaded.status == ExperimentStatus.VALID
    assert loaded.kind == ExperimentKind.FINAL_TRAINING and loaded.locked_test_evaluation_performed is True
    assert record["mrr"] == 0.03 and record["split"] == "locked-test"  # what fill_final_results.py reads
    assert record["provenance"]["checkpoint_sha256"] == "f" * 64


def test_a_non_locked_test_artifact_cannot_become_the_frozen_record():
    with pytest.raises(ValueError):
        pc.build_experiment_record({"split": "validation"}, checkpoint=good_checkpoint(), checkpoint_sha256="f" * 64,
                                   final_epoch=2, generated_at="x")


def test_evaluate_refuses_without_the_explicit_approval_flag(capsys):
    code = pc.main(["evaluate", "--checkpoint", "x.pt", "--final-epoch", "2", "--export-dir", "e", "--manifest", "m"])
    assert code == 2
    assert "--approve-single-locked-test-run" in capsys.readouterr().out


def test_unfilled_markers_are_found_and_filled_files_pass(tmp_path):
    (tmp_path / "README.md").write_text("fine\nlive demo: {{FILL: frontend url}}\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINAL_RESULTS.md").write_text("| R-GCN | `TBD` | `TBD` |\n")
    found = pc.find_placeholders(tmp_path)
    assert len(found) == 2 and found[0].startswith("README.md:2:") and found[1].startswith("docs/FINAL_RESULTS.md:1:")
    (tmp_path / "README.md").write_text("all real now\n")
    (tmp_path / "docs" / "FINAL_RESULTS.md").write_text("| R-GCN | 0.0100 | 0.0161 |\n")
    assert pc.find_placeholders(tmp_path) == []


def test_stamp_restores_only_the_manifest_fingerprint_from_the_original_export():
    checkpoint = {**good_checkpoint(), "locked_test_manifest_fingerprint": None, "state_dict": {"w": 1}}
    original = {"export_fingerprint": pc.FROZEN_EXPORT_FINGERPRINT, "locked_test_manifest_fingerprint": "m" * 64}
    stamped, problems = pc.stamp_checkpoint(checkpoint, original, "m" * 64)
    assert problems == [] and stamped["locked_test_manifest_fingerprint"] == "m" * 64
    assert stamped["state_dict"] == checkpoint["state_dict"] and stamped["config"] == checkpoint["config"]
    assert checkpoint["locked_test_manifest_fingerprint"] is None  # the input is not modified


@pytest.mark.parametrize(
    "checkpoint_change, original_change, recorded",
    [
        ({"locked_test_manifest_fingerprint": "already"}, {}, "m" * 64),          # nothing to stamp
        ({}, {"locked_test_manifest_fingerprint": None}, "m" * 64),               # original export has none
        ({}, {"export_fingerprint": "0" * 64}, "m" * 64),                          # different export
        ({"export_fingerprint": "0" * 64}, {"export_fingerprint": "0" * 64}, "m" * 64),  # not the frozen export
        ({}, {}, None),                                                            # nothing independent to compare
        ({}, {}, "x" * 64),                                                        # disagrees with the TransE record
    ],
)
def test_stamp_refuses_anything_that_does_not_line_up(checkpoint_change, original_change, recorded):
    checkpoint = {**good_checkpoint(), "locked_test_manifest_fingerprint": None, **checkpoint_change}
    original = {"export_fingerprint": pc.FROZEN_EXPORT_FINGERPRINT, "locked_test_manifest_fingerprint": "m" * 64, **original_change}
    stamped, problems = pc.stamp_checkpoint(checkpoint, original, recorded)
    assert stamped is None and problems
