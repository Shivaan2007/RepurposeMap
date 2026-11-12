"""The Colab final-training launcher's fail-closed pre-flight checks.

Does not exercise actual training (that needs a real export + GPU, covered
by test_rgcn_kaggle_runner.py's own synthetic-CPU smoke path for the shared
training code). This only checks that the launcher's own guards -- archive
integrity, locked-test absence, search_log.json agreement with the frozen
rgcn-wide/3-epoch spec -- actually refuse to continue when violated."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "colab_rgcn_final_runner" / "colab_rgcn_final_launcher.py"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("colab_rgcn_final_launcher", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "fake.tar.gz"
    inner = tmp_path / "inner.txt"
    inner.write_text("not the real archive")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname="inner.txt")
    return archive


def test_archive_hash_mismatch_refuses(launcher, tmp_path):
    archive = _make_archive(tmp_path)
    with pytest.raises(SystemExit, match="SHA-256"):
        launcher.verify_and_extract_archive(archive, tmp_path / "extract")


def test_archive_missing_refuses(launcher, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        launcher.verify_and_extract_archive(tmp_path / "does_not_exist.tar.gz", tmp_path / "extract")


def _clean_export(tmp_path: Path) -> Path:
    export = tmp_path / "export"
    export.mkdir()
    (export / "triples.npz").write_bytes(b"x")
    (export / "metadata.json").write_text(json.dumps({"ok": True}))
    (export / "validation_indications.json").write_text("[]")
    return export


def test_safety_check_passes_on_clean_export(launcher, tmp_path):
    launcher.safety_check(_clean_export(tmp_path))  # must not raise


def test_safety_check_refuses_on_locked_test_manifest(launcher, tmp_path):
    export = _clean_export(tmp_path)
    (export / "locked_test_manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="locked_test_manifest.json"):
        launcher.safety_check(export)


def test_safety_check_refuses_on_test_indications_key(launcher, tmp_path):
    export = _clean_export(tmp_path)
    (export / "metadata.json").write_text(json.dumps({"test_indications": []}))
    with pytest.raises(SystemExit, match="test_indications"):
        launcher.safety_check(export)


def test_safety_check_refuses_on_locked_test_fingerprint(launcher, tmp_path):
    export = _clean_export(tmp_path)
    (export / "metadata.json").write_text(json.dumps({"locked_test_manifest_fingerprint": "abc"}))
    with pytest.raises(SystemExit, match="locked_test_manifest_fingerprint"):
        launcher.safety_check(export)


def test_safety_check_refuses_on_missing_required_file(launcher, tmp_path):
    export = tmp_path / "incomplete"
    export.mkdir()
    (export / "triples.npz").write_bytes(b"x")
    with pytest.raises(SystemExit, match="missing required file"):
        launcher.safety_check(export)


def _search_log(**overrides) -> dict:
    base = {
        "status": "complete",
        "selected_name": "rgcn-wide",
        "runs": [{"name": "rgcn-wide", "best_val_epoch": 2}],
    }
    base.update(overrides)
    return base


def test_search_log_happy_path(launcher, tmp_path):
    path = tmp_path / "search_log.json"
    path.write_text(json.dumps(_search_log()))
    result = launcher.load_and_verify_search_log(path)
    assert result["selected_name"] == "rgcn-wide"


def test_search_log_refuses_wrong_selection(launcher, tmp_path):
    path = tmp_path / "search_log.json"
    path.write_text(json.dumps(_search_log(selected_name="rgcn-base",
                                             runs=[{"name": "rgcn-base", "best_val_epoch": 1}])))
    with pytest.raises(SystemExit, match="frozen"):
        launcher.load_and_verify_search_log(path)


def test_search_log_refuses_incomplete_status(launcher, tmp_path):
    path = tmp_path / "search_log.json"
    path.write_text(json.dumps(_search_log(status="in_progress")))
    with pytest.raises(SystemExit, match="complete"):
        launcher.load_and_verify_search_log(path)


def test_search_log_refuses_wrong_epoch_count(launcher, tmp_path):
    path = tmp_path / "search_log.json"
    path.write_text(json.dumps(_search_log(runs=[{"name": "rgcn-wide", "best_val_epoch": 5}])))
    with pytest.raises(SystemExit, match="does not match the frozen"):
        launcher.load_and_verify_search_log(path)


def test_search_log_refuses_missing_file(launcher, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        launcher.load_and_verify_search_log(tmp_path / "missing.json")


def test_real_archive_and_search_log_pass_verification(launcher):
    """End-to-end sanity against the actual committed artifacts this
    launcher is meant to run against -- not synthetic data."""
    real_archive = REPO_ROOT / "kaggle_rgcn_final_v5_recovery" / "repurposemap-0f23fe3.tar.gz"
    real_search_log = REPO_ROOT / "kaggle_rgcn_v8_recovery" / "rgcn_search" / "search_log.json"
    if not real_archive.is_file() or not real_search_log.is_file():
        pytest.skip("real Kaggle recovery artifacts not present in this checkout")
    result = launcher.load_and_verify_search_log(real_search_log)
    assert result["selected_name"] == "rgcn-wide"
    import hashlib
    assert hashlib.sha256(real_archive.read_bytes()).hexdigest() == launcher.EXPECTED_ARCHIVE_SHA256
