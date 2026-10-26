"""The Kaggle runner script, executed for real on CPU against a tiny synthetic export.

Regression: the script once had no ``if __name__ == "__main__"`` guard, so
running it did nothing and exited 0 -- which would have read as a passing
smoke test. These tests run it as a script and check what it prints."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from graph_builders import primekg_style_graph  # noqa: E402
from repurposemap.embeddings.export import build_export, write_export  # noqa: E402

ML_DIR = Path(__file__).resolve().parents[1]
RUNNER = ML_DIR / "kaggle" / "rgcn_sampled_kaggle_runner.py"


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("runner_export")
    types, edges = {}, []
    for i in range(60):
        types[f"D{i}"], types[f"DIS{i}"] = "drug", "disease"
        edges.append(("indication", "indication", f"D{i}", f"DIS{i}"))
    for i in range(5):
        types[f"P{i}"] = "protein"
        edges.append(("drug_protein", "target", "D0", f"P{i}"))
    csv_path = root / "fake.csv"
    csv_path.write_bytes(b"fake primekg csv content")
    bundle = build_export(primekg_style_graph(types, edges), csv_path=str(csv_path), seed=42, test_fraction=0.1,
                          max_test_diseases=6, val_fraction=0.2, repo_dir=".")
    write_export(bundle, root / "export")
    return root / "export"


def run_runner(*args):
    env = {**os.environ, "PYTHONPATH": str(ML_DIR / "src")}
    return subprocess.run([sys.executable, str(RUNNER), *args], capture_output=True, text=True, env=env, timeout=300)


def test_running_the_script_prints_usage_instead_of_silently_doing_nothing():
    result = run_runner("--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_smoke_mode_runs_trains_stops_resumes_and_reports(export_dir, tmp_path):
    result = run_runner("smoke", "--export-dir", str(export_dir), "--work-dir", str(tmp_path), "--device", "cpu")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "smoke_report.json").read_text())
    assert report["resume_ok"] is True
    assert report["stopped_for_time_budget_first"] is True
    assert report["resumed_final_epoch"] == 1


def test_final_before_any_search_fails_with_a_clear_message(export_dir, tmp_path):
    result = run_runner("final", "--export-dir", str(export_dir), "--work-dir", str(tmp_path), "--device", "cpu")
    assert result.returncode != 0
    assert "run `search` first" in (result.stdout + result.stderr)
    assert "Traceback" not in result.stderr


def _load_runner():
    spec = importlib.util.spec_from_file_location("rgcn_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_status_only_calls_it_complete_when_every_epoch_ran():
    status = _load_runner().final_status
    assert status({"final_epoch": 0, "stopped_for_time_budget": True}, 3).startswith("INCOMPLETE")
    assert status({"final_epoch": 2, "stopped_for_time_budget": False}, 3).startswith("COMPLETE")
    # Hitting the time budget on the very last epoch is still a finished run.
    assert status({"final_epoch": 2, "stopped_for_time_budget": True}, 3).startswith("COMPLETE")
