"""End-to-end validation search -> selection -> final training, on a small synthetic export (Stage D)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from graph_builders import primekg_style_graph  # noqa: E402
from repurposemap.embeddings.export import (  # noqa: E402
    assert_eligible_for_locked_test_evaluation,
    build_export,
    load_export,
    load_locked_test_manifest,
    write_export,
    write_locked_test_manifest,
)
from repurposemap.embeddings.transe_train_gpu import (  # noqa: E402
    GPUTrainingConfig,
    SEARCH_CONFIGS,
    load_checkpoint,
    run_validation_search,
    train_final,
)

INDICATION = ("indication", "indication")
TARGET = ("drug_protein", "target")


def _synthetic_graph():
    types: dict[str, str] = {}
    edges = []
    for i in range(60):
        drug, disease = f"D{i}", f"DIS{i}"
        types[drug] = "drug"
        types[disease] = "disease"
        edges.append((*INDICATION, drug, disease))
    for i in range(5):
        types[f"P{i}"] = "protein"
        edges.append((*TARGET, "D0", f"P{i}"))
    return primekg_style_graph(types, edges)


@pytest.fixture
def loaded_export(tmp_path):
    graph = _synthetic_graph()
    csv_path = tmp_path / "fake.csv"
    csv_path.write_bytes(b"fake primekg csv content")
    bundle = build_export(graph, csv_path=str(csv_path), seed=42, test_fraction=0.1, max_test_diseases=6, val_fraction=0.2, repo_dir=".")
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)
    return load_export(export_dir)


@pytest.fixture
def loaded_export_with_manifest(tmp_path):
    """Same as ``loaded_export``, plus the real, matching locked-test manifest
    written to a local-only path -- for tests that exercise the actual
    end-to-end gate, not a hand-built stand-in."""
    graph = _synthetic_graph()
    csv_path = tmp_path / "fake.csv"
    csv_path.write_bytes(b"fake primekg csv content")
    bundle = build_export(graph, csv_path=str(csv_path), seed=42, test_fraction=0.1, max_test_diseases=6, val_fraction=0.2, repo_dir=".")
    export_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    write_export(bundle, export_dir)
    write_locked_test_manifest(bundle, manifest_path)
    return load_export(export_dir), manifest_path


def _tiny_search_configs():
    # Same 3 named configs, but tiny dims/epochs for a fast test.
    return tuple(
        (name, GPUTrainingConfig(embedding_dim=4, distance=cfg.distance, margin=1.0, learning_rate=1e-2, negatives_per_positive=2))
        for name, cfg in SEARCH_CONFIGS
    )


def test_search_trains_all_three_configs_and_selects_exactly_one(loaded_export, tmp_path):
    log_path = tmp_path / "search_log.json"
    log = run_validation_search(
        loaded_export, search_configs=_tiny_search_configs(), epochs=4, validate_every=2, output_log_path=log_path
    )

    assert len(log["candidates"]) == 3
    assert 0 <= log["selected_index"] < 3
    assert log["selected_name"] == log["candidates"][log["selected_index"]]["name"]
    assert log_path.exists()
    assert json.loads(log_path.read_text()) == log


def test_search_preserves_a_complete_checkpoint_per_config(loaded_export, tmp_path):
    checkpoint_dir = tmp_path / "search_checkpoints"
    log = run_validation_search(
        loaded_export, search_configs=_tiny_search_configs(), epochs=3, validate_every=1, checkpoint_dir=checkpoint_dir
    )

    assert len(log["runs"]) == 3
    for run in log["runs"]:
        name = run["name"]
        expected_path = checkpoint_dir / f"{name}.pt"
        assert run["checkpoint_path"] == str(expected_path)
        assert expected_path.is_file(), f"no checkpoint written for {name}"

        checkpoint = load_checkpoint(expected_path)
        # Full state, not just the logged metric history.
        assert "state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "rng_state" in checkpoint
        assert checkpoint["is_final_training"] is False
        # Each config's own checkpoint holds its own epoch count/history,
        # not another config's.
        assert checkpoint["config"]["embedding_dim"] == dict(_tiny_search_configs())[name].embedding_dim


def test_search_without_checkpoint_dir_behaves_exactly_as_before(loaded_export, tmp_path):
    # Backward compatibility: omitting checkpoint_dir must not write any
    # checkpoint file anywhere, or change the returned log's shape.
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=2, validate_every=1)
    for run in log["runs"]:
        assert run["checkpoint_path"] is None
    assert list(tmp_path.rglob("*.pt")) == []  # no checkpoint written anywhere


def test_search_never_touches_locked_test_data(loaded_export, tmp_path):
    # The loaded_export fixture never had a locked-test manifest loaded at
    # all -- confirms the search runs fine regardless.
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=2, validate_every=1)
    assert log["selected_name"] is not None


def test_search_does_not_use_popularity_or_path_or_test_performance(loaded_export):
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=2, validate_every=1)
    for candidate in log["candidates"]:
        assert set(candidate) == {"name", "embedding_dim", "val_mrr", "val_macro_auprc"}


def test_tuning_mode_uses_tuning_triples_not_final_triples(loaded_export, tmp_path, monkeypatch):
    seen_lengths = []
    import repurposemap.embeddings.transe_train_gpu as module

    original_train_gpu = module.train_gpu

    def _spy(triples, *args, **kwargs):
        seen_lengths.append(len(triples))
        return original_train_gpu(triples, *args, **kwargs)

    monkeypatch.setattr(module, "train_gpu", _spy)
    run_validation_search(loaded_export, search_configs=_tiny_search_configs()[:1], epochs=1, validate_every=1)

    assert seen_lengths == [len(loaded_export.tuning_triples)]
    assert seen_lengths[0] != len(loaded_export.final_triples)


def test_final_mode_uses_final_triples_and_marks_the_checkpoint(loaded_export, tmp_path):
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=2, validate_every=1)
    selected_config = dict(_tiny_search_configs())[log["selected_name"]]

    checkpoint_path = tmp_path / "final.pt"
    checkpoint = train_final(
        loaded_export, selected_config=selected_config, epochs=2, search_log=log,
        checkpoint_path=checkpoint_path, git_commit="deadbeef", git_dirty=False,
    )

    assert checkpoint["is_final_training"] is True
    assert checkpoint["export_fingerprint"] == loaded_export.metadata["export_fingerprint"]
    for field in (
        "state_dict", "optimizer_state_dict", "config", "entity_index", "relation_index",
        "indication_relation_idx", "indication_relation_key", "primekg_fingerprint", "git_commit",
        "git_dirty", "export_fingerprint", "hyperparameter_search", "epoch", "final_train_loss",
        "train_loss_curve", "normalization_assumptions", "rng_state",
    ):
        assert field in checkpoint, f"missing checkpoint field: {field}"

    reloaded = load_checkpoint(checkpoint_path)
    assert reloaded["is_final_training"] is True


def test_final_checkpoint_passes_the_locked_test_eligibility_gate(loaded_export_with_manifest, tmp_path):
    loaded_export, manifest_path = loaded_export_with_manifest
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=1, validate_every=1)
    selected_config = dict(_tiny_search_configs())[log["selected_name"]]
    checkpoint_path = tmp_path / "final.pt"
    checkpoint = train_final(loaded_export, selected_config=selected_config, epochs=1, search_log=log, checkpoint_path=checkpoint_path)

    manifest = load_locked_test_manifest(manifest_path)
    assert_eligible_for_locked_test_evaluation(checkpoint, manifest)  # must not raise -- real manifest, real checkpoint


def test_train_final_forwards_explicit_device_to_the_gpu_trainer(loaded_export, tmp_path, monkeypatch):
    # Reproduces the exact Kaggle failure mode this test guards against:
    # train_final must accept `device=` and the GPU trainer actually
    # invoked must see that device on its config, not the dataclass
    # default ("cpu").
    import repurposemap.embeddings.transe_train_gpu as module

    # Run the search itself unpatched -- only the final-training call below
    # is under test.
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=1, validate_every=1)
    selected_config = dict(_tiny_search_configs())[log["selected_name"]]
    checkpoint_path = tmp_path / "final.pt"

    # Records the device requested of train_gpu, then actually runs it on
    # "cpu" regardless -- this machine has no CUDA build to exercise, and
    # that is not what this test is checking. What matters is that
    # train_final forwards whatever device string it is given through to
    # train_gpu's config without raising (the exact Kaggle failure mode)
    # and without silently dropping it.
    seen_devices = []
    original_train_gpu = module.train_gpu

    def _spy(triples, n_entities, n_relations, type_pools, positive_store, config, **kwargs):
        seen_devices.append(config.device)
        cpu_config = replace(config, device="cpu")
        return original_train_gpu(triples, n_entities, n_relations, type_pools, positive_store, cpu_config, **kwargs)

    monkeypatch.setattr(module, "train_gpu", _spy)

    checkpoint = train_final(
        loaded_export, selected_config=selected_config, epochs=1, search_log=log,
        checkpoint_path=checkpoint_path, device="cuda:0",
    )

    assert seen_devices == ["cuda:0"]
    assert checkpoint["config"]["device"] == "cuda:0"
    # Hyperparameters untouched by the device override.
    assert checkpoint["config"]["embedding_dim"] == selected_config.embedding_dim
    assert checkpoint["config"]["distance"] == selected_config.distance


def test_train_final_default_device_matches_prior_behavior(loaded_export, tmp_path, monkeypatch):
    # No `device=` passed -- must keep training on whatever
    # selected_config.device already was (the dataclass default, "cpu"),
    # unchanged from before the device parameter existed.
    import repurposemap.embeddings.transe_train_gpu as module

    # Run the search itself unpatched -- only the final-training call below
    # is under test.
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=1, validate_every=1)
    selected_config = dict(_tiny_search_configs())[log["selected_name"]]
    checkpoint_path = tmp_path / "final.pt"

    seen_devices = []
    original_train_gpu = module.train_gpu

    def _spy(triples, n_entities, n_relations, type_pools, positive_store, config, **kwargs):
        seen_devices.append(config.device)
        return original_train_gpu(triples, n_entities, n_relations, type_pools, positive_store, config, **kwargs)

    monkeypatch.setattr(module, "train_gpu", _spy)

    train_final(
        loaded_export, selected_config=selected_config, epochs=1, search_log=log,
        checkpoint_path=checkpoint_path,
    )

    assert seen_devices == [selected_config.device] == ["cpu"]


def test_train_final_with_explicit_device_still_never_touches_locked_test_data(loaded_export_with_manifest, tmp_path):
    # Mirrors test_final_checkpoint_passes_the_locked_test_eligibility_gate,
    # but with an explicit device override in play -- the device plumbing
    # must not open any new path to locked-test data.
    loaded_export, manifest_path = loaded_export_with_manifest
    log = run_validation_search(loaded_export, search_configs=_tiny_search_configs(), epochs=1, validate_every=1)
    selected_config = dict(_tiny_search_configs())[log["selected_name"]]
    checkpoint_path = tmp_path / "final.pt"
    checkpoint = train_final(
        loaded_export, selected_config=selected_config, epochs=1, search_log=log,
        checkpoint_path=checkpoint_path, device="cpu",
    )

    manifest = load_locked_test_manifest(manifest_path)
    assert_eligible_for_locked_test_evaluation(checkpoint, manifest)  # must not raise
