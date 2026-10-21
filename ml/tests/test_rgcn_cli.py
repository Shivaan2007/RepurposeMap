"""The rgcn-train/rgcn-evaluate CLI commands -- reuse the exact same
export package TransE's own transe-export command writes (see
test_transe_cli.py; there is deliberately no rgcn-export command)."""

from __future__ import annotations

import pytest

from repurposemap.cli import main


@pytest.fixture
def rgcn_export_dir(primekg_csv, tmp_path):
    output_dir = tmp_path / "rgcn_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    exit_code = main(
        [
            "transe-export", "--source", "primekg", "--csv", str(primekg_csv),
            "--output-dir", str(output_dir), "--manifest-path", str(manifest_path),
            "--seed", "0", "--max-test-diseases", "1", "--test-fraction", "1.0", "--val-fraction", "0.0",
        ]
    )
    assert exit_code == 0
    return output_dir, manifest_path


def test_rgcn_train_and_evaluate_do_not_call_the_graph_loader(rgcn_export_dir, tmp_path, monkeypatch):
    output_dir, _manifest_path = rgcn_export_dir

    import repurposemap.cli as cli_module

    def _poisoned_load_source(*_args, **_kwargs):
        raise AssertionError("_load_source must never be called for a graph-free command")

    monkeypatch.setattr(cli_module, "_load_source", _poisoned_load_source)

    checkpoint_path = tmp_path / "ckpt.pt"
    exit_code = main(
        [
            "rgcn-train", "--export-dir", str(output_dir), "--config-name", "rgcn-base",
            "--epochs", "1", "--validate-every", "1", "--output", str(checkpoint_path),
        ]
    )
    assert exit_code == 0
    assert checkpoint_path.exists()

    exit_code = main(["rgcn-evaluate", "--export-dir", str(output_dir), "--checkpoint", str(checkpoint_path), "--split", "validation"])
    assert exit_code == 0


def test_rgcn_evaluate_refuses_locked_test_without_manifest(rgcn_export_dir, tmp_path):
    output_dir, _manifest_path = rgcn_export_dir
    checkpoint_path = tmp_path / "ckpt.pt"
    main(
        [
            "rgcn-train", "--export-dir", str(output_dir), "--config-name", "rgcn-base",
            "--epochs", "1", "--output", str(checkpoint_path),
        ]
    )

    exit_code = main(["rgcn-evaluate", "--export-dir", str(output_dir), "--checkpoint", str(checkpoint_path), "--split", "locked-test"])
    assert exit_code == 1


def test_rgcn_evaluate_refuses_locked_test_against_a_non_final_checkpoint(rgcn_export_dir, tmp_path):
    output_dir, manifest_path = rgcn_export_dir
    checkpoint_path = tmp_path / "ckpt.pt"
    main(
        [
            "rgcn-train", "--export-dir", str(output_dir), "--config-name", "rgcn-base",
            "--epochs", "1", "--output", str(checkpoint_path),
        ]
    )  # a search-phase checkpoint -- is_final_training is False

    exit_code = main(
        [
            "rgcn-evaluate", "--export-dir", str(output_dir), "--checkpoint", str(checkpoint_path),
            "--split", "locked-test", "--manifest", str(manifest_path),
        ]
    )
    assert exit_code == 1


def test_rgcn_config_name_choices_include_the_three_search_configs_and_final():
    from repurposemap.cli import build_parser
    import argparse

    parser = build_parser()
    subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
    rgcn_train_parser = subparsers_action.choices["rgcn-train"]
    config_name_action = next(a for a in rgcn_train_parser._actions if a.dest == "config_name")  # noqa: SLF001
    assert set(config_name_action.choices) == {"rgcn-base", "rgcn-wide", "rgcn-moreBases", "final"}
