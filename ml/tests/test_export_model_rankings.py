"""ml/scripts/export_model_rankings.py -- the final-checkpoint -> serving-
bundle glue script. Runs real (tiny) model training/inference against the
small PrimeKG fixture graph, never the real data/raw/primekg or
data/serving/. All IDs/names below are the synthetic fixture's own
(DEMO-...), not real PrimeKG content. Invokes the script as a subprocess,
same convention as test_api_openapi_export.py's export_openapi.py test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from repurposemap.cli import main as cli_main  # noqa: E402
from repurposemap.serving.loader import load_bundle_for_disease  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ml" / "scripts" / "export_model_rankings.py"
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "ml" / "src")}

# DEMO-DIS-TAU in ml/tests/fixtures/primekg_mini.csv
_DISEASE_ID = "110"
# DEMO-GENE-NU, a gene/protein node -- used for the wrong-node-type check.
_GENE_ID = "101"


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
def final_rgcn_checkpoint(export_dir, tmp_path):
    """A real (1-epoch) R-GCN checkpoint, hand-flipped to
    is_final_training=True -- this project's actual final-training path
    (--config-name final) needs a recorded 3-config search log, which is
    out of scope to fabricate just for this test; flipping the flag on an
    otherwise-real trained checkpoint exercises export_model_rankings.py's
    own logic (encode/score/rank/validate/write) just as thoroughly."""
    search_checkpoint = tmp_path / "search_ckpt.pt"
    exit_code = cli_main(
        [
            "rgcn-train", "--export-dir", str(export_dir), "--config-name", "rgcn-base",
            "--epochs", "1", "--output", str(search_checkpoint),
        ]
    )
    assert exit_code == 0

    checkpoint = torch.load(search_checkpoint, weights_only=False)
    assert checkpoint["is_final_training"] is False
    checkpoint["is_final_training"] = True
    final_checkpoint = tmp_path / "final_ckpt.pt"
    torch.save(checkpoint, final_checkpoint)
    return final_checkpoint


def test_refuses_a_non_final_checkpoint(export_dir, tmp_path):
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
            "--disease-id", _DISEASE_ID, "--top-k", "3", "--ranking-method", "rgcn",
            "--output-dir", str(tmp_path / "serving"),
        ]
    )
    assert result.returncode != 0
    assert "is_final_training" in result.stderr
    assert not (tmp_path / "serving").exists() or not list((tmp_path / "serving").glob("*.json"))


def test_writes_a_loadable_bundle_for_a_final_checkpoint(primekg_csv, export_dir, final_rgcn_checkpoint, tmp_path):
    serving_dir = tmp_path / "serving"

    result = _run(
        [
            "--export-dir", str(export_dir), "--checkpoint", str(final_rgcn_checkpoint), "--model", "rgcn",
            "--primekg-csv", str(primekg_csv), "--disease-id", _DISEASE_ID, "--top-k", "3",
            "--ranking-method", "rgcn", "--source-experiment-file", "demo_experiment.json",
            "--output-dir", str(serving_dir),
        ]
    )
    assert result.returncode == 0, result.stderr

    bundle = load_bundle_for_disease(_DISEASE_ID, serving_dir=serving_dir)
    assert bundle is not None
    assert bundle.disease_id == _DISEASE_ID
    assert bundle.disease_name == "DEMO Disease Tau"
    assert bundle.ranking_method == "rgcn"
    assert bundle.provenance.source_experiment_file == "demo_experiment.json"
    assert bundle.candidate_universe.size is not None and bundle.candidate_universe.size > 0
    assert len(bundle.candidates) <= 3
    assert [c.rank for c in bundle.candidates] == list(range(1, len(bundle.candidates) + 1))
    drug_ids = [c.drug_id for c in bundle.candidates]
    assert len(drug_ids) == len(set(drug_ids))


def test_writes_a_raw_ranking_audit_file_when_requested(primekg_csv, export_dir, final_rgcn_checkpoint, tmp_path):
    serving_dir = tmp_path / "serving"
    raw_dir = tmp_path / "raw"

    result = _run(
        [
            "--export-dir", str(export_dir), "--checkpoint", str(final_rgcn_checkpoint), "--model", "rgcn",
            "--primekg-csv", str(primekg_csv), "--disease-id", _DISEASE_ID, "--top-k", "3",
            "--ranking-method", "rgcn", "--output-dir", str(serving_dir),
            "--raw-ranking-output-dir", str(raw_dir),
        ]
    )
    assert result.returncode == 0, result.stderr

    raw_files = list(raw_dir.glob("*.json"))
    assert len(raw_files) == 1


def test_rejects_a_disease_id_not_in_the_entity_index(primekg_csv, export_dir, final_rgcn_checkpoint, tmp_path):
    result = _run(
        [
            "--export-dir", str(export_dir), "--checkpoint", str(final_rgcn_checkpoint), "--model", "rgcn",
            "--primekg-csv", str(primekg_csv), "--disease-id", "not-a-real-id", "--top-k", "3",
            "--ranking-method", "rgcn", "--output-dir", str(tmp_path / "serving"),
        ]
    )
    assert result.returncode != 0
    assert "entity index" in result.stderr


def test_rejects_a_disease_id_that_is_actually_a_gene(primekg_csv, export_dir, final_rgcn_checkpoint, tmp_path):
    result = _run(
        [
            "--export-dir", str(export_dir), "--checkpoint", str(final_rgcn_checkpoint), "--model", "rgcn",
            "--primekg-csv", str(primekg_csv), "--disease-id", _GENE_ID, "--top-k", "3",
            "--ranking-method", "rgcn", "--output-dir", str(tmp_path / "serving"),
        ]
    )
    assert result.returncode != 0
    assert "node_type" in result.stderr
