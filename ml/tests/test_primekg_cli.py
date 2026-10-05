from pathlib import Path

import pytest

from repurposemap.cli import main

DRUG_MU_ID = "109"
DRUG_KAPPA_ID = "103"
PROT_XI_ID = "102"


def test_stats_with_primekg_source_reports_counts(primekg_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main(["stats", "--source", "primekg", "--csv", str(primekg_csv)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Total nodes: 12" in out
    assert "Total edges: 15" in out
    assert "indication: 6" in out


def test_search_with_primekg_source_shows_id_and_external_id(
    primekg_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["search", "--source", "primekg", "--query", "DEMO Drug Eta", "--csv", str(primekg_csv)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "DEMO Drug Eta (drug)  id=111  external_id=DEMO-DRUG-ETA" in out


def test_paths_by_name_with_primekg_source(primekg_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main([
        "paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-name", "DEMO Drug Kappa", "--target-name", "DEMO Protein Xi", "--max-path-length", "2",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "1 path(s)" in out
    assert "DEMO Drug Kappa" in out and "DEMO Protein Nu" in out and "DEMO Protein Xi" in out


def test_paths_by_id_with_primekg_source(primekg_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main([
        "paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-id", DRUG_KAPPA_ID, "--target-id", PROT_XI_ID, "--max-path-length", "2",
    ])
    assert exit_code == 0
    assert "1 path(s)" in capsys.readouterr().out


def test_ambiguous_name_lists_candidates_instead_of_guessing(
    primekg_csv: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main([
        "paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-name", "DEMO Compound Pi", "--target-name", "DEMO Protein Xi",
    ])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "matches 2 entities" in err
    assert "id=105" in err and "id=106" in err
    assert "--source-id" in err


def test_ambiguous_name_resolved_by_id(primekg_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main([
        "paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-id", "106", "--target-name", "DEMO Disease Sigma",
    ])
    assert exit_code == 0
    assert "1 path(s)" in capsys.readouterr().out


def test_unknown_id_is_reported_as_error(primekg_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main([
        "paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-id", "99999999", "--target-name", "DEMO Protein Xi",
    ])
    assert exit_code == 1
    assert "source entity not found: '99999999'" in capsys.readouterr().err


def test_no_path_with_primekg_source_exits_zero(primekg_csv: Path, capsys: pytest.CaptureFixture[str]):
    exit_code = main([
        "paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-id", DRUG_MU_ID, "--target-id", PROT_XI_ID,
    ])
    assert exit_code == 0
    assert "No directed path" in capsys.readouterr().out


def test_missing_primekg_file_is_reported_with_default_location(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["stats", "--source", "primekg", "--csv", str(tmp_path / "kg.csv")])
    assert exit_code == 1
    assert "PrimeKG CSV not found" in capsys.readouterr().err


def test_endpoint_needs_name_or_id(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        main(["paths", "--target-name", "DEMO Protein Xi"])
