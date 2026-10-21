"""The transe-export/transe-train/transe-evaluate CLI commands (Stage D)."""

from __future__ import annotations

import json

import pytest

from repurposemap.cli import GRAPH_FREE_COMMANDS, build_parser, main


@pytest.fixture
def primekg_export_csv(primekg_csv, tmp_path):
    """Use the existing tiny PrimeKG-schema fixture to drive a real export."""
    return primekg_csv


def test_transe_export_rejects_a_manifest_path_inside_the_output_dir(primekg_export_csv, tmp_path):
    output_dir = tmp_path / "transe_export"
    exit_code = main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_export_csv),
            "--output-dir", str(output_dir),
            "--manifest-path", str(output_dir / "locked_test_manifest.json"),
            "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.5",
        ]
    )
    assert exit_code == 1


def test_transe_export_writes_the_expected_files_and_keeps_manifest_separate(primekg_export_csv, tmp_path, capsys):
    output_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    exit_code = main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_export_csv),
            "--output-dir", str(output_dir), "--manifest-path", str(manifest_path),
            "--seed", "0", "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.0",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert set(p.name for p in output_dir.iterdir()) == {"triples.npz", "metadata.json", "validation_indications.json"}
    assert manifest_path.exists()
    assert not (output_dir / "locked_test_manifest.json").exists()
    assert "export_fingerprint" in out
    assert "LOCAL ONLY" in out


def test_transe_train_and_evaluate_do_not_call_the_graph_loader(primekg_export_csv, tmp_path, monkeypatch):
    output_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_export_csv),
            "--output-dir", str(output_dir), "--manifest-path", str(manifest_path),
            "--seed", "0", "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.0",
        ]
    )

    import repurposemap.cli as cli_module

    def _poisoned_load_source(*_args, **_kwargs):
        raise AssertionError("_load_source must never be called for a graph-free command")

    monkeypatch.setattr(cli_module, "_load_source", _poisoned_load_source)

    checkpoint_path = tmp_path / "ckpt.pt"
    exit_code = main(
        [
            "transe-train", "--export-dir", str(output_dir), "--config-name", "search-dim64",
            "--epochs", "1", "--validate-every", "1", "--output", str(checkpoint_path),
        ]
    )
    assert exit_code == 0
    assert checkpoint_path.exists()

    exit_code = main(["transe-evaluate", "--export-dir", str(output_dir), "--checkpoint", str(checkpoint_path), "--split", "validation"])
    assert exit_code == 0


def test_transe_evaluate_refuses_locked_test_without_manifest(primekg_export_csv, tmp_path):
    output_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_export_csv),
            "--output-dir", str(output_dir), "--manifest-path", str(manifest_path),
            "--seed", "0", "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.0",
        ]
    )
    checkpoint_path = tmp_path / "ckpt.pt"
    main(
        [
            "transe-train", "--export-dir", str(output_dir), "--config-name", "search-dim64",
            "--epochs", "1", "--output", str(checkpoint_path),
        ]
    )

    exit_code = main(["transe-evaluate", "--export-dir", str(output_dir), "--checkpoint", str(checkpoint_path), "--split", "locked-test"])
    assert exit_code == 1


def test_transe_evaluate_refuses_locked_test_against_a_non_final_checkpoint(primekg_export_csv, tmp_path):
    output_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_export_csv),
            "--output-dir", str(output_dir), "--manifest-path", str(manifest_path),
            "--seed", "0", "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.0",
        ]
    )
    checkpoint_path = tmp_path / "ckpt.pt"
    main(
        [
            "transe-train", "--export-dir", str(output_dir), "--config-name", "search-dim64",
            "--epochs", "1", "--output", str(checkpoint_path),
        ]
    )  # a search-phase checkpoint -- is_final_training is False

    exit_code = main(
        [
            "transe-evaluate", "--export-dir", str(output_dir), "--checkpoint", str(checkpoint_path),
            "--split", "locked-test", "--manifest", str(manifest_path),
        ]
    )
    assert exit_code == 1


def test_transe_subcommands_are_registered_and_graph_free_set_is_exact():
    import argparse

    parser = build_parser()
    subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
    command_names = set(subparsers_action.choices)
    for name in ("transe-export", "transe-train", "transe-evaluate", "rgcn-train", "rgcn-evaluate"):
        assert name in command_names
    assert set(GRAPH_FREE_COMMANDS) == {"transe-train", "transe-evaluate", "rgcn-train", "rgcn-evaluate"}
    assert "transe-export" not in GRAPH_FREE_COMMANDS
    # R-GCN reuses TransE's export; there is deliberately no rgcn-export command.
    assert "rgcn-export" not in command_names
