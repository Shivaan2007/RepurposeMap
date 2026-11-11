"""repurposemap.serving.export_rankings -- assembling a bundle from
already-ranked candidate records, synthetic test input only.

All disease/drug names and IDs below are invented (DEMO ...) and say
nothing real about any actual disease, drug, or ranking."""

from __future__ import annotations

import json

import pytest

from repurposemap.serving.export_rankings import build_bundle, main


def _records():
    return [
        {"drug_id": "DEMO:drug-2", "drug_name": "demo drug two", "rank": 2, "score": 0.5},
        {"drug_id": "DEMO:drug-1", "drug_name": "demo drug one", "rank": 1, "score": 0.9},
        {"drug_id": "DEMO:drug-3", "drug_name": "demo drug three", "rank": 3, "score": 0.1},
    ]


def test_build_bundle_sorts_by_rank_regardless_of_input_order():
    bundle = build_bundle(
        disease_id="DEMO:0001",
        disease_name="demo disease",
        ranking_method="path_quality",
        candidate_universe_description="demo universe",
        candidate_records=_records(),
        top_k=10,
    )
    assert [c.rank for c in bundle.candidates] == [1, 2, 3]
    assert bundle.candidates[0].drug_id == "DEMO:drug-1"


def test_build_bundle_truncates_explicitly_to_top_k():
    bundle = build_bundle(
        disease_id="DEMO:0001",
        disease_name="demo disease",
        ranking_method="path_quality",
        candidate_universe_description="demo universe",
        candidate_records=_records(),
        top_k=2,
    )
    assert len(bundle.candidates) == 2
    assert [c.rank for c in bundle.candidates] == [1, 2]


def test_build_bundle_rejects_empty_candidate_records():
    with pytest.raises(ValueError, match="no candidate records"):
        build_bundle(
            disease_id="DEMO:0001",
            disease_name="demo disease",
            ranking_method="path_quality",
            candidate_universe_description="demo universe",
            candidate_records=[],
            top_k=10,
        )


def test_build_bundle_rejects_top_k_below_one():
    with pytest.raises(ValueError, match="top_k"):
        build_bundle(
            disease_id="DEMO:0001",
            disease_name="demo disease",
            ranking_method="path_quality",
            candidate_universe_description="demo universe",
            candidate_records=_records(),
            top_k=0,
        )


def test_build_bundle_carries_explanation_paths_through_from_raw_dicts():
    records = [
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
    ]
    bundle = build_bundle(
        disease_id="DEMO:0001",
        disease_name="demo disease",
        ranking_method="path_quality",
        candidate_universe_description="demo universe",
        candidate_records=records,
        top_k=10,
    )
    assert bundle.candidates[0].explanation_paths[0].path_score == 0.7
    assert bundle.candidates[0].explanation_paths[0].steps[0].relation == "demo_relation"


def test_build_bundle_carries_provenance_through():
    bundle = build_bundle(
        disease_id="DEMO:0001",
        disease_name="demo disease",
        ranking_method="transe",
        candidate_universe_description="demo universe",
        candidate_records=_records(),
        top_k=10,
        source_experiment_file="demo_experiment.json",
        source_commit="abc1234",
    )
    assert bundle.provenance.ranking_method == "transe"
    assert bundle.provenance.source_experiment_file == "demo_experiment.json"
    assert bundle.provenance.source_commit == "abc1234"


def test_cli_main_writes_a_valid_bundle_file(tmp_path):
    candidates_file = tmp_path / "candidates.json"
    candidates_file.write_text(json.dumps(_records()))
    output_file = tmp_path / "out.json"

    exit_code = main(
        [
            "--disease-id", "DEMO:0001",
            "--disease-name", "demo disease",
            "--ranking-method", "path_quality",
            "--candidate-universe-description", "demo universe",
            "--candidates", str(candidates_file),
            "--top-k", "2",
            "--output", str(output_file),
        ]
    )

    assert exit_code == 0
    written = json.loads(output_file.read_text())
    assert written["disease_id"] == "DEMO:0001"
    assert len(written["candidates"]) == 2


def test_cli_main_with_output_dir_derives_a_slugified_filename(tmp_path):
    """The recommended real-workflow path: --output-dir, not --output,
    so the written filename is guaranteed to match what
    repurposemap.serving.loader.load_bundle_for_disease will look up for
    this exact disease_id -- see test_serving_loader.py's matching
    round-trip test."""
    candidates_file = tmp_path / "candidates.json"
    candidates_file.write_text(json.dumps(_records()))
    output_dir = tmp_path / "serving"

    exit_code = main(
        [
            "--disease-id", "DEMO:0001",
            "--disease-name", "demo disease",
            "--ranking-method", "path_quality",
            "--candidate-universe-description", "demo universe",
            "--candidates", str(candidates_file),
            "--top-k", "2",
            "--output-dir", str(output_dir),
        ]
    )

    assert exit_code == 0
    written_file = output_dir / "DEMO_0001.json"
    assert written_file.is_file()
    assert json.loads(written_file.read_text())["disease_id"] == "DEMO:0001"


def test_cli_main_rejects_both_output_and_output_dir(tmp_path):
    candidates_file = tmp_path / "candidates.json"
    candidates_file.write_text(json.dumps(_records()))

    with pytest.raises(SystemExit):
        main(
            [
                "--disease-id", "DEMO:0001",
                "--disease-name", "demo disease",
                "--ranking-method", "path_quality",
                "--candidate-universe-description", "demo universe",
                "--candidates", str(candidates_file),
                "--top-k", "2",
                "--output", str(tmp_path / "a.json"),
                "--output-dir", str(tmp_path),
            ]
        )
