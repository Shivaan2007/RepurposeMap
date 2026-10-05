from pathlib import Path

import pytest

from repurposemap.cli import main

ALPHA = "DEMO Drug Alpha"
ETA = "DEMO Disease Eta"


def test_stats_command_reports_totals(sample_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main(["stats", "--csv", str(sample_csv)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Total nodes: 15" in out
    assert "Total edges: 23" in out
    assert "drug_targets_protein: 5" in out


def test_search_command_finds_matching_entities(
    sample_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["search", "--query", "drug", "--csv", str(sample_csv)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "DEMO Drug Alpha (drug)" in out
    assert "DEMO Drug Epsilon (drug)" in out


def test_search_command_reports_no_matches(
    sample_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["search", "--query", "zzz-absent", "--csv", str(sample_csv)])
    assert exit_code == 0
    assert "No entities match" in capsys.readouterr().out


def test_search_max_results_is_respected(sample_csv: Path, capsys: pytest.CaptureFixture[str]):
    main(["search", "--query", "DEMO", "--max-results", "2", "--csv", str(sample_csv)])
    out = capsys.readouterr().out
    assert "2 match(es)" in out


def test_paths_command_prints_numbered_routes(
    sample_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["paths", "--source-name", ALPHA, "--target-name", "DEMO Disease Zeta", "--csv", str(sample_csv)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "4 path(s)" in out
    assert "[1] 1 hop(s)" in out
    assert "--drug_indicated_for_disease-->" in out


def test_paths_command_respects_max_path_length(
    sample_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(
        ["paths", "--source-name", ALPHA, "--target-name", ETA, "--max-path-length", "3", "--csv", str(sample_csv)]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "No directed path" in out


def test_paths_command_reports_missing_entity_as_error(
    sample_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["paths", "--source-name", "DEMO Drug Missing", "--target-name", ETA, "--csv", str(sample_csv)])
    assert exit_code == 1
    assert "source entity not found" in capsys.readouterr().err


def test_missing_csv_is_reported_as_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main(["stats", "--csv", str(tmp_path / "nope.csv")])
    assert exit_code == 1
    assert "not found" in capsys.readouterr().err
