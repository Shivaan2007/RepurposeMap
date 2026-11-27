"""scripts/pick_case_studies.py on the synthetic PrimeKG fixture: the selection rule only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import pick_case_studies as pcs  # noqa: E402
from repurposemap.cli import main as cli_main  # noqa: E402
from repurposemap.embeddings.export import load_export  # noqa: E402


@pytest.fixture
def export_dir(primekg_csv, tmp_path):
    out = tmp_path / "export"
    assert cli_main(["transe-export", "--source", "primekg", "--csv", str(primekg_csv), "--output-dir", str(out),
                     "--manifest-path", str(tmp_path / "m.json"), "--seed", "0", "--max-test-diseases", "1",
                     "--test-fraction", "1.0", "--val-fraction", "0.0"]) == 0
    return out


def test_locked_test_diseases_are_never_picked(export_dir):
    loaded = load_export(export_dir)
    picked = {r["disease_id"] for r in pcs.pick(loaded, 10, 10)}
    assert picked and not picked & set(loaded.test_disease_ids)


def test_each_pick_has_a_known_drug_and_the_case_drug_is_the_lowest_index(export_dir):
    for row in pcs.pick(load_export(export_dir), 10, 10):
        assert row["drug_ids"] and row["case_drug_id"] == row["drug_ids"][0]
        assert row["drug_ids"] == sorted(row["drug_ids"], key=pcs._key)


def test_only_the_first_k_get_a_case_study_and_the_order_is_deterministic(export_dir):
    loaded = load_export(export_dir)
    rows = pcs.pick(loaded, 10, 1)
    assert rows[0]["case_drug_id"] is not None
    assert all(r["case_drug_id"] is None for r in rows[1:])
    degrees = [r["degree"] for r in rows]
    assert degrees == sorted(degrees, reverse=True)
    assert rows == pcs.pick(loaded, 10, 1)


def test_names_are_read_from_the_csv_without_loading_it_whole(primekg_csv):
    assert pcs.names_for(primekg_csv, {"110", "109"}) == {"110": "DEMO Disease Tau", "109": "DEMO Drug Mu"}


def test_the_cli_prints_a_table_and_runnable_commands(export_dir, primekg_csv, capsys):
    assert pcs.main(["--export-dir", str(export_dir), "--primekg-csv", str(primekg_csv), "--n-diseases", "2", "--n-case-studies", "1"]) == 0
    out = capsys.readouterr().out
    assert "post_checkpoint.py export" in out and "post_checkpoint.py case-study" in out and "--known-drug-id" in out
