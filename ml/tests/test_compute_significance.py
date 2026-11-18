"""ml/scripts/compute_significance.py -- runs the frozen disease-level
paired bootstrap on two --per-disease-output JSON files. Pure JSON/numpy
logic, no torch, no real model, no real locked-test numbers below."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compute_significance import SplitMismatchError, _load_outcomes, main  # noqa: E402


def _write_outcomes(path: Path, *, split: str, ranks_by_disease: dict[str, list[int]]) -> None:
    outcomes = [
        {"disease_id": disease_id, "drug_id": f"drug-{i}", "rank": rank}
        for disease_id, ranks in ranks_by_disease.items()
        for i, rank in enumerate(ranks)
    ]
    path.write_text(json.dumps({"split": split, "outcomes": outcomes}))


def test_load_outcomes_rejects_a_non_locked_test_file(tmp_path):
    path = tmp_path / "validation.json"
    _write_outcomes(path, split="validation", ranks_by_disease={"d1": [1]})
    with pytest.raises(SplitMismatchError):
        _load_outcomes(path, label="--rgcn-outcomes")


def test_load_outcomes_accepts_a_locked_test_file(tmp_path):
    path = tmp_path / "locked.json"
    _write_outcomes(path, split="locked-test", ranks_by_disease={"d1": [1], "d2": [3]})
    outcomes = _load_outcomes(path, label="--rgcn-outcomes")
    assert len(outcomes) == 2


def test_main_writes_an_output_artifact_with_both_metrics(tmp_path, capsys):
    rgcn_path = tmp_path / "rgcn.json"
    transe_path = tmp_path / "transe.json"
    output_path = tmp_path / "significance.json"
    diseases = {f"d{i}": [1] for i in range(10)}  # identical -> zero diff, deterministic
    _write_outcomes(rgcn_path, split="locked-test", ranks_by_disease=diseases)
    _write_outcomes(transe_path, split="locked-test", ranks_by_disease=diseases)

    exit_code = main(
        [
            "--rgcn-outcomes", str(rgcn_path),
            "--transe-outcomes", str(transe_path),
            "--n-resamples", "100",
            "--seed", "0",
            "--output", str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert set(payload["results"]) == {"mrr", "hits_at_10"}
    assert payload["results"]["mrr"]["diff_rgcn_minus_transe"] == 0.0
    assert payload["seed"] == 0
    assert payload["n_resamples"] == 100

    captured = capsys.readouterr()
    assert "mrr:" in captured.out
    assert "hits_at_10:" in captured.out


def test_main_refuses_a_validation_split_transe_file(tmp_path):
    rgcn_path = tmp_path / "rgcn.json"
    transe_path = tmp_path / "transe.json"
    _write_outcomes(rgcn_path, split="locked-test", ranks_by_disease={"d1": [1]})
    _write_outcomes(transe_path, split="validation", ranks_by_disease={"d1": [1]})

    with pytest.raises(SplitMismatchError):
        main(
            [
                "--rgcn-outcomes", str(rgcn_path),
                "--transe-outcomes", str(transe_path),
            ]
        )
