"""The local export package: round-trip, leakage, and fingerprint correctness (Stage D)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from graph_builders import primekg_style_graph
from repurposemap.embeddings.export import (
    assert_eligible_for_locked_test_evaluation,
    build_export,
    compute_export_fingerprint,
    compute_locked_test_manifest_fingerprint,
    load_export,
    load_locked_test_manifest,
    recompute_manifest_fingerprint,
    write_export,
    write_locked_test_manifest,
)

INDICATION = ("indication", "indication")
TARGET = ("drug_protein", "target")


def _synthetic_graph(n_diseases: int = 60, n_proteins: int = 5):
    types: dict[str, str] = {}
    edges = []
    for i in range(n_diseases):
        drug, disease = f"D{i}", f"DIS{i}"
        types[drug] = "drug"
        types[disease] = "disease"
        edges.append((*INDICATION, drug, disease))
    for i in range(n_proteins):
        types[f"P{i}"] = "protein"
        edges.append((*TARGET, "D0", f"P{i}"))
    return primekg_style_graph(types, edges)


def _write_fake_csv(path):
    path.write_bytes(b"fake primekg csv content for fingerprinting")
    return path


def _build(tmp_path, *, seed: int = 42, max_test_diseases: int = 6, val_fraction: float = 0.2):
    graph = _synthetic_graph()
    csv_path = _write_fake_csv(tmp_path / "fake.csv")
    return graph, build_export(
        graph, csv_path=str(csv_path), seed=seed, test_fraction=0.1, max_test_diseases=max_test_diseases, val_fraction=val_fraction, repo_dir="."
    )


def test_export_round_trip_reconstructs_every_field(tmp_path):
    _graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)

    loaded = load_export(export_dir)

    assert len(loaded.entity_index) == len(bundle.metadata["entity_index"]["index_to_id"])
    assert len(loaded.relation_index) == len(bundle.metadata["relation_index"]["index_to_id"])
    assert loaded.indication_relation_idx == bundle.metadata["indication_relation_idx"]
    assert loaded.val_disease_ids == tuple(bundle.metadata["val_disease_ids"])
    assert loaded.test_disease_ids == tuple(bundle.metadata["test_disease_ids"])
    assert loaded.fit_disease_ids == tuple(bundle.metadata["fit_disease_ids"])
    assert np.array_equal(loaded.final_triples, bundle.arrays.final_triples)
    assert np.array_equal(loaded.val_indication_mask, bundle.arrays.val_indication_mask)
    assert len(loaded.val_indications) == len(bundle.val_indications)
    assert set(loaded.drug_ids) == {f"D{i}" for i in range(60)}


def test_tuning_triples_derived_graph_free_equals_masked_final_triples(tmp_path):
    _graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)
    loaded = load_export(export_dir)

    expected = loaded.final_triples[~loaded.val_indication_mask]
    assert np.array_equal(loaded.tuning_triples, expected)
    # tuning_triples must be strictly smaller (some validation indication rows removed)
    assert len(loaded.tuning_triples) < len(loaded.final_triples)


def test_final_triples_contain_validation_indications_but_not_locked_test(tmp_path):
    graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)
    write_locked_test_manifest(bundle, tmp_path / "locked_test_manifest.json")
    loaded = load_export(export_dir)
    manifest = load_locked_test_manifest(tmp_path / "locked_test_manifest.json")

    test_disease_indices = {loaded.entity_index.id_to_index[d] for d in manifest.test_disease_ids}
    indication_rows = loaded.final_triples[:, 1] == loaded.indication_relation_idx
    heads, tails = loaded.final_triples[:, 0], loaded.final_triples[:, 2]
    touches_test = np.isin(heads[indication_rows], list(test_disease_indices)) | np.isin(
        tails[indication_rows], list(test_disease_indices)
    )
    assert touches_test.sum() == 0  # no leakage, either direction

    # But validation diseases' indications ARE present (real training signal).
    val_disease_indices = {loaded.entity_index.id_to_index[d] for d in loaded.val_disease_ids}
    touches_val = np.isin(heads[indication_rows], list(val_disease_indices)) | np.isin(
        tails[indication_rows], list(val_disease_indices)
    )
    assert touches_val.sum() > 0


def test_upload_package_contains_zero_locked_test_drug_disease_pairs(tmp_path):
    """The structural guarantee, not a gate: parse every uploaded file and
    confirm no (drug_id, disease_id) pair from the locked-test manifest
    appears anywhere in it, as an actual indication record -- not just a
    substring coincidence."""
    graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)
    write_locked_test_manifest(bundle, tmp_path / "locked_test_manifest.json")

    manifest = load_locked_test_manifest(tmp_path / "locked_test_manifest.json")
    test_pairs = {(r.drug_id, r.disease_id) for r in manifest.test_indications}
    assert len(test_pairs) > 0  # the test is meaningless if there's nothing to check

    # 1. validation_indications.json: no row's (drug_id, disease_id) pair matches.
    val_records = json.loads((export_dir / "validation_indications.json").read_text())
    val_pairs = {(r["drug_id"], r["disease_id"]) for r in val_records}
    assert val_pairs.isdisjoint(test_pairs)

    # 2. metadata.json: test_disease_ids (IDs only) are allowed; but no
    # structure resembling a (drug, disease) pair record exists for them.
    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert set(metadata["test_disease_ids"]) == set(manifest.test_disease_ids)  # IDs present, as designed
    assert "test_indications" not in metadata
    assert "test_drug_ids" not in metadata

    # 3. triples.npz: final_triples has no indication triple touching a test disease.
    loaded = load_export(export_dir)
    test_disease_indices = {loaded.entity_index.id_to_index[d] for d in manifest.test_disease_ids}
    indication_rows = loaded.final_triples[:, 1] == loaded.indication_relation_idx
    heads, tails = loaded.final_triples[:, 0], loaded.final_triples[:, 2]
    touches_test = np.isin(heads[indication_rows], list(test_disease_indices)) | np.isin(
        tails[indication_rows], list(test_disease_indices)
    )
    assert touches_test.sum() == 0


def test_locked_test_manifest_is_not_written_inside_the_export_directory(tmp_path):
    _graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    write_export(bundle, export_dir)
    write_locked_test_manifest(bundle, manifest_path)

    assert manifest_path.exists()
    assert not (export_dir / "locked_test_manifest.json").exists()
    assert set(export_dir.iterdir()) == {
        export_dir / "triples.npz",
        export_dir / "metadata.json",
        export_dir / "validation_indications.json",
    }


def test_the_full_pipeline_loads_and_derives_tuning_triples_with_no_manifest_present_anywhere(tmp_path):
    """The Kaggle/GPU training package must be fully usable with no locked-test
    manifest on disk at all -- not merely absent from the upload, but never
    required to exist anywhere the loader looks."""
    _graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)
    # Deliberately never call write_locked_test_manifest at all.

    assert not (tmp_path / "locked_test_manifest.json").exists()
    loaded = load_export(export_dir)  # must not raise, must not look for the manifest
    assert len(loaded.tuning_triples) > 0
    assert len(loaded.val_indications) > 0


def test_deterministic_export_same_graph_and_seed_produces_identical_content(tmp_path):
    graph = _synthetic_graph()
    csv_path = _write_fake_csv(tmp_path / "fake.csv")
    bundle_a = build_export(graph, csv_path=str(csv_path), seed=42, test_fraction=0.1, max_test_diseases=6, val_fraction=0.2, repo_dir=".")
    bundle_b = build_export(graph, csv_path=str(csv_path), seed=42, test_fraction=0.1, max_test_diseases=6, val_fraction=0.2, repo_dir=".")

    assert np.array_equal(bundle_a.arrays.final_triples, bundle_b.arrays.final_triples)
    assert np.array_equal(bundle_a.arrays.val_indication_mask, bundle_b.arrays.val_indication_mask)
    assert bundle_a.metadata["export_fingerprint"] == bundle_b.metadata["export_fingerprint"]


# --- Canonical export fingerprint ---


def _fingerprint_inputs():
    triples = np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int32)
    mask = np.array([False, True])
    codes = np.array([0, 1, 0], dtype=np.int8)
    metadata = {"split_seed": 42, "val_disease_ids": ["a", "b"]}
    return triples, mask, codes, metadata


def test_fingerprint_is_identical_for_independently_built_identical_content():
    triples, mask, codes, metadata = _fingerprint_inputs()
    fp_a = compute_export_fingerprint(triples.copy(), mask.copy(), codes.copy(), dict(metadata))
    fp_b = compute_export_fingerprint(triples.copy(), mask.copy(), codes.copy(), dict(metadata))
    assert fp_a == fp_b


def test_fingerprint_changes_when_one_triple_changes():
    triples, mask, codes, metadata = _fingerprint_inputs()
    base = compute_export_fingerprint(triples, mask, codes, metadata)
    changed_triples = triples.copy()
    changed_triples[0, 2] = 99
    changed = compute_export_fingerprint(changed_triples, mask, codes, metadata)
    assert base != changed


def test_fingerprint_changes_when_a_split_id_changes():
    triples, mask, codes, metadata = _fingerprint_inputs()
    base = compute_export_fingerprint(triples, mask, codes, metadata)
    changed_metadata = {**metadata, "val_disease_ids": ["a", "c"]}
    changed = compute_export_fingerprint(triples, mask, codes, changed_metadata)
    assert base != changed


def test_fingerprint_changes_when_a_provenance_field_changes():
    triples, mask, codes, metadata = _fingerprint_inputs()
    base = compute_export_fingerprint(triples, mask, codes, metadata)
    changed_metadata = {**metadata, "split_seed": 43}
    changed = compute_export_fingerprint(triples, mask, codes, changed_metadata)
    assert base != changed


def test_fingerprint_rejects_metadata_that_already_has_a_fingerprint_field():
    triples, mask, codes, metadata = _fingerprint_inputs()
    with pytest.raises(ValueError, match="export_fingerprint"):
        compute_export_fingerprint(triples, mask, codes, {**metadata, "export_fingerprint": "x"})


# --- Locked-test manifest fingerprint (binds the exact 498-pair manifest) ---


def _manifest_content():
    return {
        "schema_version": 1,
        "export_fingerprint": "abc123",
        "primekg_fingerprint": {"sha256": "deadbeef", "size_bytes": 42},
        "split_seed": 42,
        "test_disease_ids": ["DIS1", "DIS2"],
        "test_indications": [
            {"disease_id": "DIS1", "drug_id": "D1", "disease_name": "DIS1", "drug_name": "D1"},
            {"disease_id": "DIS2", "drug_id": "D2", "disease_name": "DIS2", "drug_name": "D2"},
        ],
    }


def test_manifest_fingerprint_is_identical_for_independently_built_identical_content():
    content = _manifest_content()
    fp_a = compute_locked_test_manifest_fingerprint(dict(content))
    fp_b = compute_locked_test_manifest_fingerprint(dict(content))
    assert fp_a == fp_b


def test_manifest_fingerprint_is_order_independent_for_pair_records():
    content = _manifest_content()
    reordered = {**content, "test_indications": list(reversed(content["test_indications"]))}
    assert compute_locked_test_manifest_fingerprint(content) == compute_locked_test_manifest_fingerprint(reordered)


def test_manifest_fingerprint_changes_when_one_held_out_pair_changes():
    content = _manifest_content()
    base = compute_locked_test_manifest_fingerprint(dict(content))
    changed = dict(content)
    changed["test_indications"] = [
        content["test_indications"][0],
        {**content["test_indications"][1], "drug_id": "D999"},
    ]
    assert base != compute_locked_test_manifest_fingerprint(changed)


def test_manifest_fingerprint_changes_when_one_test_disease_id_changes():
    content = _manifest_content()
    base = compute_locked_test_manifest_fingerprint(dict(content))
    changed = {**content, "test_disease_ids": ["DIS1", "DIS999"]}
    assert base != compute_locked_test_manifest_fingerprint(changed)


def test_manifest_fingerprint_rejects_content_that_already_has_the_field():
    content = {**_manifest_content(), "locked_test_manifest_fingerprint": "x"}
    with pytest.raises(ValueError, match="locked_test_manifest_fingerprint"):
        compute_locked_test_manifest_fingerprint(content)


def test_real_export_propagates_only_the_manifest_hash_never_the_labels(tmp_path):
    _graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    write_export(bundle, export_dir)
    write_locked_test_manifest(bundle, tmp_path / "locked_test_manifest.json")

    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert "locked_test_manifest_fingerprint" in metadata
    assert isinstance(metadata["locked_test_manifest_fingerprint"], str)
    assert "test_indications" not in metadata  # the hash travels; the labels do not

    manifest = load_locked_test_manifest(tmp_path / "locked_test_manifest.json")
    assert manifest.locked_test_manifest_fingerprint == metadata["locked_test_manifest_fingerprint"]
    assert recompute_manifest_fingerprint(manifest) == metadata["locked_test_manifest_fingerprint"]


# --- Locked-test evaluation gate (checkpoint.is_final_training,
#     export_fingerprint, AND locked_test_manifest_fingerprint) ---


def _fake_manifest(**overrides):
    base = {
        "export_fingerprint": "abc",
        "primekg_fingerprint": {"sha256": "x"},
        "split_seed": 42,
        "test_disease_ids": ("DIS1",),
        "test_indications": (),
    }
    base.update(overrides)
    from repurposemap.embeddings.export import LockedTestManifest

    content = {"schema_version": 1, **base}
    fingerprint = compute_locked_test_manifest_fingerprint(dict(content))
    return LockedTestManifest(
        export_fingerprint=base["export_fingerprint"],
        primekg_fingerprint=base["primekg_fingerprint"],
        split_seed=base["split_seed"],
        test_disease_ids=base["test_disease_ids"],
        test_indications=base["test_indications"],
        locked_test_manifest_fingerprint=fingerprint,
    ), fingerprint


def test_gate_refuses_a_non_final_checkpoint():
    manifest, fingerprint = _fake_manifest()
    checkpoint = {"is_final_training": False, "export_fingerprint": "abc", "locked_test_manifest_fingerprint": fingerprint}
    with pytest.raises(ValueError, match="is_final_training"):
        assert_eligible_for_locked_test_evaluation(checkpoint, manifest)


def test_gate_refuses_a_mismatched_export_fingerprint():
    manifest, fingerprint = _fake_manifest(export_fingerprint="different")
    checkpoint = {"is_final_training": True, "export_fingerprint": "abc", "locked_test_manifest_fingerprint": fingerprint}
    with pytest.raises(ValueError, match="export_fingerprint"):
        assert_eligible_for_locked_test_evaluation(checkpoint, manifest)


def test_gate_refuses_a_correct_export_fingerprint_but_wrong_manifest_content():
    # The exact scenario the manifest fingerprint exists to catch: a
    # manifest that happens to share the right export_fingerprint (e.g.
    # someone passed last week's manifest from the same export) but whose
    # actual held-out pairs differ from what the checkpoint expects.
    manifest, _real_fingerprint = _fake_manifest(test_disease_ids=("DIS1", "DIS2"))
    checkpoint = {
        "is_final_training": True,
        "export_fingerprint": "abc",
        "locked_test_manifest_fingerprint": "not-the-real-fingerprint",
    }
    with pytest.raises(ValueError, match="recomputed fingerprint"):
        assert_eligible_for_locked_test_evaluation(checkpoint, manifest)


def test_gate_allows_a_final_checkpoint_with_a_fully_matching_manifest():
    manifest, fingerprint = _fake_manifest()
    checkpoint = {"is_final_training": True, "export_fingerprint": "abc", "locked_test_manifest_fingerprint": fingerprint}
    assert_eligible_for_locked_test_evaluation(checkpoint, manifest)  # must not raise


def test_gate_via_the_real_export_and_manifest_end_to_end(tmp_path):
    _graph, bundle = _build(tmp_path)
    export_dir = tmp_path / "transe_export"
    manifest_path = tmp_path / "locked_test_manifest.json"
    write_export(bundle, export_dir)
    write_locked_test_manifest(bundle, manifest_path)
    manifest = load_locked_test_manifest(manifest_path)

    good_checkpoint = {
        "is_final_training": True,
        "export_fingerprint": bundle.metadata["export_fingerprint"],
        "locked_test_manifest_fingerprint": bundle.metadata["locked_test_manifest_fingerprint"],
    }
    assert_eligible_for_locked_test_evaluation(good_checkpoint, manifest)  # must not raise

    tampered_checkpoint = {**good_checkpoint, "locked_test_manifest_fingerprint": "wrong"}
    with pytest.raises(ValueError, match="recomputed fingerprint"):
        assert_eligible_for_locked_test_evaluation(tampered_checkpoint, manifest)
