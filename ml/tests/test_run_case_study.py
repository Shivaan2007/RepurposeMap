"""ml/scripts/run_case_study.py -- the post-model masked-case-study runner.
Runs real (tiny) model training/inference against the small PrimeKG
fixture graph, never the real data/raw/primekg or a real disease/drug.
Invokes the script as a subprocess, same convention as
test_api_openapi_export.py's export_openapi.py test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from repurposemap.cli import main as cli_main  # noqa: E402
from repurposemap.embeddings.export import load_export  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ml" / "scripts" / "run_case_study.py"
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "ml" / "src")}

# The fixture's three (drug, disease) indication pairs -- see
# ml/tests/fixtures/primekg_mini.csv. Exactly one disease ends up in the
# test split (--max-test-diseases 1 --test-fraction 1.0), so this picks
# whichever pair actually landed in the training-eligible (fit) pool --
# the one that has something in final_triples to mask -- instead of
# hardcoding a guess that might land in the excluded test split.
_CANDIDATE_PAIRS = {"110": "109", "112": "111", "104": "103"}


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT, capture_output=True, text=True, env=ENV
    )


@pytest.fixture
def export_dir(primekg_csv, tmp_path):
    output_dir = tmp_path / "export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    exit_code = cli_main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_csv),
            "--output-dir", str(output_dir), "--manifest-path", str(manifest_path),
            "--seed", "0", "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.0",
        ]
    )
    assert exit_code == 0
    return output_dir


@pytest.fixture
def fit_pair(export_dir):
    """A (disease_id, drug_id) pair from _CANDIDATE_PAIRS that this
    deterministic split actually put in the training-eligible pool."""
    loaded = load_export(export_dir)
    fit_diseases = set(loaded.fit_disease_ids)
    for disease_id, drug_id in _CANDIDATE_PAIRS.items():
        if disease_id in fit_diseases:
            return disease_id, drug_id
    pytest.fail(f"none of {_CANDIDATE_PAIRS} landed in fit_disease_ids={loaded.fit_disease_ids}")


@pytest.fixture
def final_rgcn_checkpoint(export_dir, tmp_path):
    """Same hand-flipped-flag approach as test_export_model_rankings.py --
    see that file's fixture docstring for why."""
    search_checkpoint = tmp_path / "search_ckpt.pt"
    exit_code = cli_main(
        [
            "rgcn-train", "--export-dir", str(export_dir), "--config-name", "rgcn-base",
            "--epochs", "1", "--output", str(search_checkpoint),
        ]
    )
    assert exit_code == 0

    checkpoint = torch.load(search_checkpoint, weights_only=False)
    checkpoint["is_final_training"] = True
    final_checkpoint = tmp_path / "final_ckpt.pt"
    torch.save(checkpoint, final_checkpoint)
    return final_checkpoint


def test_refuses_a_non_final_checkpoint(export_dir, fit_pair, tmp_path):
    disease_id, drug_id = fit_pair
    non_final_checkpoint = tmp_path / "search_ckpt.pt"
    cli_main(
        [
            "rgcn-train", "--export-dir", str(export_dir), "--config-name", "rgcn-base",
            "--epochs", "1", "--output", str(non_final_checkpoint),
        ]
    )

    result = _run(
        [
            "--export-dir", str(export_dir), "--checkpoint", str(non_final_checkpoint), "--model", "rgcn",
            "--disease-id", disease_id, "--known-drug-id", drug_id, "--ranking-method", "rgcn",
            "--output", str(tmp_path / "case_study.json"),
        ]
    )
    assert result.returncode != 0
    assert "is_final_training" in result.stderr


def test_runs_a_real_masked_case_study_and_writes_rank(primekg_csv, export_dir, fit_pair, final_rgcn_checkpoint, tmp_path):
    disease_id, drug_id = fit_pair
    output_path = tmp_path / "case_study.json"

    result = _run(
        [
            "--export-dir", str(export_dir), "--checkpoint", str(final_rgcn_checkpoint), "--model", "rgcn",
            "--primekg-csv", str(primekg_csv), "--disease-id", disease_id, "--known-drug-id", drug_id,
            "--ranking-method", "rgcn", "--output", str(output_path),
        ]
    )
    assert result.returncode == 0, result.stderr

    written = json.loads(output_path.read_text())
    assert written["spec"]["disease_id"] == disease_id
    assert written["spec"]["known_drug_id"] == drug_id
    assert written["rank"] is not None and written["rank"] >= 1
    assert written["provenance"]["triples_level_masking"].startswith("final_triples row(s)")
    assert len(written["limitations"]) == 2


def test_masking_actually_removes_a_row_from_final_triples(export_dir, fit_pair):
    """The scientific-integrity check this whole script exists for: the
    known indication must actually be present in final_triples before
    masking (otherwise the model never needed masking to avoid "seeing"
    it in the first place -- see run_case_study.py's module docstring)."""
    disease_id, drug_id = fit_pair
    loaded = load_export(export_dir)
    disease_idx = loaded.entity_index.id_to_index[disease_id]
    drug_idx = loaded.entity_index.id_to_index[drug_id]
    h, r, t = loaded.final_triples[:, 0], loaded.final_triples[:, 1], loaded.final_triples[:, 2]
    forward = (h == drug_idx) & (r == loaded.indication_relation_idx) & (t == disease_idx)
    backward = (h == disease_idx) & (r == loaded.indication_relation_idx) & (t == drug_idx)
    assert (forward | backward).sum() > 0
