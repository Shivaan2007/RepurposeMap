import csv
from pathlib import Path

import repurposemap

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "synthetic_demo_graph.csv"

EXPECTED_COLUMNS = ["source_name", "source_type", "relation", "target_name", "target_type"]


def test_package_imports():
    assert repurposemap.__version__ == "0.1.0"


def test_sample_csv_has_expected_header():
    with SAMPLE_PATH.open(newline="") as f:
        header = next(csv.reader(f))
    assert header == EXPECTED_COLUMNS


def test_sample_csv_entities_are_labelled_synthetic():
    with SAMPLE_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "sample CSV has no edges"
    for row in rows:
        assert row["source_name"].startswith("DEMO "), row
        assert row["target_name"].startswith("DEMO "), row
