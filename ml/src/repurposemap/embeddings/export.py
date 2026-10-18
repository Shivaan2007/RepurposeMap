"""The local export package: everything GPU-side TransE training needs,
with the locked-test labels structurally kept off it.

``build_export`` runs locally (it needs the loaded PrimeKG graph, same as
every other scoring command). Its output splits into two physically
separate pieces:

- ``transe_export/`` -- ``triples.npz``, ``metadata.json``,
  ``validation_indications.json``. This directory is what gets uploaded to
  Kaggle/Colab. It contains the locked test diseases' **IDs** (needed to
  confirm which diseases were held out of ``final_triples`, for
  provenance), but never their actual true-drug labels.
- ``locked_test_manifest.json`` -- the 498 real locked-test `(drug,
  disease)` pairs. Written by the same export run, to a path the caller
  chooses **outside** ``transe_export/``. Nothing in this module ever
  copies it inside that directory, and nothing in
  ``repurposemap.embeddings.transe_train_gpu`` or
  ``repurposemap.evaluation.transe_baseline``'s validation path reads it.

Everything downstream of ``load_export`` is NetworkX-free: it reconstructs
``EntityIndex``, ``RelationIndex``, ``TypePools``, the triple arrays, and
the drug universe entirely from ``transe_export/``'s three files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from repurposemap.adapters.primekg import PrimeKGFingerprint, compute_file_fingerprint
from repurposemap.embeddings.entity_index import (
    EntityIndex,
    RelationIndex,
    build_entity_index,
    build_relation_index,
    resolve_relation_key,
)
from repurposemap.embeddings.negative_sampling import TypePools, build_type_pools, build_type_pools_from_codes
from repurposemap.embeddings.triples import build_triples
from repurposemap.evaluation.artifact import get_git_commit
from repurposemap.evaluation.indications import INDICATION_RELATION, IndicationRecord, extract_indications
from repurposemap.evaluation.split import split_by_disease
from repurposemap.evaluation.training_graph import build_training_graph
from repurposemap.evaluation.transe_baseline import (
    ValidationSplit,
    indications_for_diseases,
    split_validation_from_training,
)

EXPORT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Canonical export fingerprint
# --------------------------------------------------------------------------


def compute_export_fingerprint(
    final_triples: np.ndarray,
    val_indication_mask: np.ndarray,
    entity_node_type_codes: np.ndarray,
    metadata: dict,
) -> str:
    """A fingerprint over canonical *content*, not over serialized file bytes.

    ``.npz``'s own container format can carry details (ordering, internal
    layout) that don't reflect the actual scientific content, so hashing
    the raw file bytes could make two exports with identical arrays hash
    differently for reasons unrelated to the data. This hashes fixed-dtype,
    fixed-order raw array bytes instead, plus a canonically-serialized
    metadata JSON:

    - ``final_triples``: cast to ``int32``, forced C-contiguous, raw bytes.
    - ``val_indication_mask``: cast to ``bool``, forced C-contiguous, raw bytes.
    - ``entity_node_type_codes``: cast to ``int8``, forced C-contiguous, raw bytes.
    - ``metadata``: ``json.dumps(metadata, sort_keys=True, separators=(",", ":"))``,
      UTF-8 encoded -- deterministic regardless of the dict's construction
      order. ``metadata`` must NOT already contain an ``"export_fingerprint"``
      key (this function raises if it does, rather than silently hashing a
      value that depends on itself).

    Each of the four components is SHA-256'd individually, then the four
    hex digests are joined with ``"|"`` and SHA-256'd once more (a
    hash-of-hashes), which avoids any ambiguity from concatenating raw
    bytes directly (two different splits of bytes between components could
    otherwise concatenate to the same string).

    Two independently generated exports with identical array content and
    identical metadata produce the identical fingerprint; changing any one
    triple, split ID, or provenance field changes it.
    """
    if "export_fingerprint" in metadata:
        raise ValueError("metadata must not already contain 'export_fingerprint' when computing it")

    triples_bytes = np.ascontiguousarray(final_triples, dtype=np.int32).tobytes()
    mask_bytes = np.ascontiguousarray(val_indication_mask, dtype=np.bool_).tobytes()
    codes_bytes = np.ascontiguousarray(entity_node_type_codes, dtype=np.int8).tobytes()
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    component_hashes = [hashlib.sha256(b).hexdigest() for b in (triples_bytes, mask_bytes, codes_bytes, metadata_bytes)]
    return hashlib.sha256("|".join(component_hashes).encode("ascii")).hexdigest()


def _canonical_locked_test_manifest_content(manifest: dict) -> dict:
    """The exact, ordering-independent content ``compute_locked_test_manifest_fingerprint``
    hashes -- pulled out so both the "write" and the "verify at evaluation
    time" paths build it identically from whatever dict they have on hand."""
    return {
        "schema_version": manifest["schema_version"],
        "export_fingerprint": manifest["export_fingerprint"],
        "primekg_fingerprint": manifest["primekg_fingerprint"],
        "split_seed": manifest["split_seed"],
        "test_disease_ids": sorted(manifest["test_disease_ids"]),
        "test_indications": sorted(manifest["test_indications"], key=lambda r: (r["disease_id"], r["drug_id"])),
    }


def compute_locked_test_manifest_fingerprint(manifest: dict) -> str:
    """A fingerprint over the locked-test manifest's canonical scientific content.

    Binds the exact 498-pair manifest a final checkpoint was validated
    against to that checkpoint, the same way ``compute_export_fingerprint``
    binds the training data: ``manifest`` must include at minimum
    ``schema_version``, ``export_fingerprint``, ``primekg_fingerprint``,
    ``split_seed``, ``test_disease_ids``, and ``test_indications`` (each a
    dict with ``disease_id``/``drug_id``/``disease_name``/``drug_name``).
    ``test_disease_ids`` and ``test_indications`` are sorted here before
    serializing, so the result does not depend on the order either list
    happened to be constructed in.

    Serialized as ``json.dumps(..., sort_keys=True, separators=(",", ":"))``,
    UTF-8, then SHA-256'd. ``manifest`` must NOT already contain a
    ``"locked_test_manifest_fingerprint"`` key (raises if it does, so the
    fingerprint never depends on itself).

    This fingerprint is provenance only -- a hash, not the labels
    themselves. It is safe to include in the uploadable ``metadata.json``;
    the 498 real pairs it was computed from are not.
    """
    if "locked_test_manifest_fingerprint" in manifest:
        raise ValueError("manifest must not already contain 'locked_test_manifest_fingerprint' when computing it")
    canonical = _canonical_locked_test_manifest_content(manifest)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Building the export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportArrays:
    final_triples: np.ndarray  # int32, shape (n, 3), C-contiguous
    val_indication_mask: np.ndarray  # bool, shape (n,)
    entity_node_type_codes: np.ndarray  # int8, shape (n_entities,)


@dataclass(frozen=True)
class ExportBundle:
    """Everything one ``build_export`` call produces, before writing to disk."""

    arrays: ExportArrays
    metadata: dict  # includes "export_fingerprint"; never includes test indication pairs
    val_indications: tuple[IndicationRecord, ...]
    test_indications: tuple[IndicationRecord, ...]  # goes ONLY to locked_test_manifest.json


def _record_to_dict(record: IndicationRecord) -> dict:
    return {
        "disease_id": record.disease_id,
        "drug_id": record.drug_id,
        "disease_name": record.disease_name,
        "drug_name": record.drug_name,
    }


def build_export(
    graph: nx.MultiDiGraph,
    *,
    csv_path: str | Path,
    seed: int = 42,
    test_fraction: float = 0.1,
    max_test_diseases: int | None = 100,
    val_fraction: float = 0.1,
    repo_dir: str | Path | None = None,
) -> ExportBundle:
    """Build the full export in memory. Call ``write_export``/``write_locked_test_manifest`` to persist it."""
    indications = extract_indications(graph)
    split = split_by_disease(indications, seed=seed, test_fraction=test_fraction, max_test_diseases=max_test_diseases)
    validation_split: ValidationSplit = split_validation_from_training(split, val_fraction=val_fraction)

    entity_index = build_entity_index(graph)
    relation_index = build_relation_index(graph)
    indication_relation_idx = resolve_relation_key(relation_index, INDICATION_RELATION)

    # Only the locked test diseases' indications are removed -- validation
    # diseases' indications stay IN final_triples (they are real training
    # signal for the final model; tuning_triples, derived later, is what
    # removes them too, graph-free, via val_indication_mask).
    final_graph = build_training_graph(graph, split.test_disease_ids)
    triple_set = build_triples(final_graph, entity_index, relation_index)
    final_triples = triple_set.triples.astype(np.int32)

    type_pools: TypePools = build_type_pools(graph, entity_index)
    node_type_names = sorted(type_pools.pools_by_type.keys())
    name_to_code = {name: code for code, name in enumerate(node_type_names)}
    entity_node_type_codes = np.array([name_to_code[t] for t in type_pools.type_of_index], dtype=np.int8)

    val_disease_indices = frozenset(entity_index.id_to_index[d] for d in validation_split.val_disease_ids)
    heads, relations, tails = final_triples[:, 0], final_triples[:, 1], final_triples[:, 2]
    is_indication = relations == indication_relation_idx
    touches_val_disease = np.isin(heads, list(val_disease_indices)) | np.isin(tails, list(val_disease_indices))
    val_indication_mask = (is_indication & touches_val_disease).astype(np.bool_)

    val_indications = indications_for_diseases(split.train_indications, validation_split.val_disease_ids)
    test_indications = split.test_indications  # locked labels -- manifest only, never metadata

    fingerprint: PrimeKGFingerprint = compute_file_fingerprint(csv_path)
    git_commit, git_dirty = get_git_commit(repo_dir)

    metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "entity_index": {"index_to_id": list(entity_index.index_to_id)},
        "relation_index": {"index_to_id": [list(pair) for pair in relation_index.index_to_id]},
        "node_type_names": node_type_names,
        "indication_relation_idx": indication_relation_idx,
        "indication_relation_key": list(relation_index.index_to_id[indication_relation_idx]),
        "fit_disease_ids": list(validation_split.fit_disease_ids),
        "val_disease_ids": list(validation_split.val_disease_ids),
        "test_disease_ids": list(validation_split.test_disease_ids),  # IDs only -- no drug pairs
        "primekg_fingerprint": {
            "sha256": fingerprint.sha256,
            "size_bytes": fingerprint.size_bytes,
            "retrieved_date": fingerprint.retrieved_date,
            "dataverse_doi": fingerprint.dataverse_doi,
            "release_version": fingerprint.release_version,
            "snapshot_build_date": fingerprint.snapshot_build_date,
        },
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "split_seed": seed,
        "test_fraction": test_fraction,
        "max_test_diseases": max_test_diseases,
        "val_fraction": val_fraction,
    }

    arrays = ExportArrays(
        final_triples=final_triples, val_indication_mask=val_indication_mask, entity_node_type_codes=entity_node_type_codes
    )
    fingerprint_hex = compute_export_fingerprint(arrays.final_triples, arrays.val_indication_mask, arrays.entity_node_type_codes, metadata)
    metadata = {**metadata, "export_fingerprint": fingerprint_hex}

    # The locked-test manifest's own fingerprint -- computed here (where
    # test_indications still exist in memory) so it can be propagated into
    # the uploadable metadata.json as JUST a hash, never the labels. The
    # same value is also written into locked_test_manifest.json itself,
    # by write_locked_test_manifest below, from this single source of truth.
    manifest_content = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_fingerprint": metadata["export_fingerprint"],
        "primekg_fingerprint": metadata["primekg_fingerprint"],
        "split_seed": seed,
        "test_disease_ids": list(validation_split.test_disease_ids),
        "test_indications": [_record_to_dict(r) for r in test_indications],
    }
    manifest_fingerprint = compute_locked_test_manifest_fingerprint(manifest_content)
    metadata = {**metadata, "locked_test_manifest_fingerprint": manifest_fingerprint}

    return ExportBundle(arrays=arrays, metadata=metadata, val_indications=val_indications, test_indications=test_indications)


# --------------------------------------------------------------------------
# Writing / loading transe_export/
# --------------------------------------------------------------------------


def write_export(bundle: ExportBundle, output_dir: str | Path) -> None:
    """Write ``triples.npz``, ``metadata.json``, ``validation_indications.json`` into ``output_dir``.

    Never writes ``locked_test_manifest.json`` here -- call
    ``write_locked_test_manifest`` separately, to a different path, for that.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "triples.npz",
        final_triples=bundle.arrays.final_triples,
        val_indication_mask=bundle.arrays.val_indication_mask,
        entity_node_type_codes=bundle.arrays.entity_node_type_codes,
    )
    (output_dir / "metadata.json").write_text(json.dumps(bundle.metadata, indent=2))
    (output_dir / "validation_indications.json").write_text(
        json.dumps([_record_to_dict(r) for r in bundle.val_indications], indent=2)
    )


def write_locked_test_manifest(bundle: ExportBundle, manifest_path: str | Path) -> None:
    """Write the 498 locked-test labels to ``manifest_path`` -- local-only, never inside ``transe_export/``.

    Includes enough provenance (export fingerprint, PrimeKG fingerprint,
    split seed, and the manifest's own cryptographic fingerprint -- see
    ``compute_locked_test_manifest_fingerprint``) to confirm later, at
    evaluation time, that this exact manifest belongs to the export/
    checkpoint it is being used with, not just that *some* manifest with a
    matching export_fingerprint was supplied.
    """
    manifest_path = Path(manifest_path)
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_fingerprint": bundle.metadata["export_fingerprint"],
        "primekg_fingerprint": bundle.metadata["primekg_fingerprint"],
        "split_seed": bundle.metadata["split_seed"],
        "test_disease_ids": bundle.metadata["test_disease_ids"],
        "test_indications": [_record_to_dict(r) for r in bundle.test_indications],
        # Reused from build_export (the single source of truth), not
        # recomputed here -- both must agree by construction, not by luck.
        "locked_test_manifest_fingerprint": bundle.metadata["locked_test_manifest_fingerprint"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


@dataclass(frozen=True)
class LoadedExport:
    """Everything reconstructed from ``transe_export/``, with no graph object."""

    entity_index: EntityIndex
    relation_index: RelationIndex
    type_pools: TypePools
    final_triples: np.ndarray
    val_indication_mask: np.ndarray
    indication_relation_idx: int
    fit_disease_ids: tuple[str, ...]
    val_disease_ids: tuple[str, ...]
    test_disease_ids: tuple[str, ...]
    val_indications: tuple[IndicationRecord, ...]
    drug_ids: tuple[str, ...]
    metadata: dict

    @property
    def tuning_triples(self) -> np.ndarray:
        """``final_triples`` with validation-disease indication edges removed, derived
        graph-free by one boolean index -- no second leakage implementation."""
        return self.final_triples[~self.val_indication_mask]


def load_export(export_dir: str | Path) -> LoadedExport:
    """Reconstruct everything GPU-side training needs from ``transe_export/``.

    Reads only ``triples.npz``, ``metadata.json``, and
    ``validation_indications.json`` -- never looks for, and does not need,
    ``locked_test_manifest.json`` to exist anywhere.
    """
    export_dir = Path(export_dir)
    npz = np.load(export_dir / "triples.npz")
    final_triples = npz["final_triples"]
    val_indication_mask = npz["val_indication_mask"]
    entity_node_type_codes = npz["entity_node_type_codes"]

    metadata = json.loads((export_dir / "metadata.json").read_text())
    entity_index = EntityIndex.from_dict(metadata["entity_index"])
    relation_index = RelationIndex.from_dict(metadata["relation_index"])
    type_pools = build_type_pools_from_codes(entity_node_type_codes, metadata["node_type_names"])

    val_indications_raw = json.loads((export_dir / "validation_indications.json").read_text())
    val_indications = tuple(IndicationRecord(**record) for record in val_indications_raw)

    drug_code = metadata["node_type_names"].index("drug") if "drug" in metadata["node_type_names"] else None
    if drug_code is None:
        drug_ids: tuple[str, ...] = ()
    else:
        drug_ids = tuple(
            sorted(
                node_id
                for node_id, code in zip(entity_index.index_to_id, entity_node_type_codes, strict=True)
                if int(code) == drug_code
            )
        )

    return LoadedExport(
        entity_index=entity_index,
        relation_index=relation_index,
        type_pools=type_pools,
        final_triples=final_triples,
        val_indication_mask=val_indication_mask,
        indication_relation_idx=metadata["indication_relation_idx"],
        fit_disease_ids=tuple(metadata["fit_disease_ids"]),
        val_disease_ids=tuple(metadata["val_disease_ids"]),
        test_disease_ids=tuple(metadata["test_disease_ids"]),
        val_indications=val_indications,
        drug_ids=drug_ids,
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# The locked-test manifest -- loaded ONLY by the local, final evaluation step
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LockedTestManifest:
    export_fingerprint: str
    primekg_fingerprint: dict
    split_seed: int
    test_disease_ids: tuple[str, ...]
    test_indications: tuple[IndicationRecord, ...]
    locked_test_manifest_fingerprint: str  # as stored in the file -- see recompute_fingerprint() for the trusted check


def load_locked_test_manifest(manifest_path: str | Path) -> LockedTestManifest:
    """Load the local-only manifest. Never called by any tuning/training code path."""
    data = json.loads(Path(manifest_path).read_text())
    return LockedTestManifest(
        export_fingerprint=data["export_fingerprint"],
        primekg_fingerprint=data["primekg_fingerprint"],
        split_seed=data["split_seed"],
        test_disease_ids=tuple(data["test_disease_ids"]),
        test_indications=tuple(IndicationRecord(**record) for record in data["test_indications"]),
        locked_test_manifest_fingerprint=data["locked_test_manifest_fingerprint"],
    )


def recompute_manifest_fingerprint(manifest: LockedTestManifest) -> str:
    """Recompute the canonical fingerprint FROM the manifest's actual loaded
    content -- not the ``locked_test_manifest_fingerprint`` field stored
    inside the file, which is read but never trusted on its own. This is
    what ``assert_eligible_for_locked_test_evaluation`` compares against
    the checkpoint's expected value, so a corrupted or hand-edited
    manifest file (even one with its stored fingerprint field left
    unchanged) is still caught.
    """
    content = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_fingerprint": manifest.export_fingerprint,
        "primekg_fingerprint": manifest.primekg_fingerprint,
        "split_seed": manifest.split_seed,
        "test_disease_ids": list(manifest.test_disease_ids),
        "test_indications": [_record_to_dict(r) for r in manifest.test_indications],
    }
    return compute_locked_test_manifest_fingerprint(content)


# --------------------------------------------------------------------------
# Locked-test evaluation gate (Stage D §7/§10) -- an additional guard, not
# the primary mechanism: the primary guarantee is that the Kaggle-bound
# export package never contains locked-test labels at all (see above).
# This is the second, independent check applied once the labels ARE
# available again, locally.
# --------------------------------------------------------------------------


def assert_eligible_for_locked_test_evaluation(checkpoint: dict, manifest: LockedTestManifest) -> None:
    """Raise unless ``checkpoint`` and ``manifest`` are both eligible for a
    locked-test run. Three independent checks, each raising on its own:

    1. ``checkpoint.get("is_final_training")`` is ``True`` -- refuses a
       checkpoint from an interrupted or tuning-phase run.
    2. ``manifest.export_fingerprint == checkpoint.get("export_fingerprint")``
       -- refuses a manifest that does not provably come from the same
       export the checkpoint was trained from.
    3. ``recompute_manifest_fingerprint(manifest) ==
       checkpoint.get("locked_test_manifest_fingerprint")`` -- refuses a
       manifest whose actual content (test diseases, held-out pairs) does
       not match what the checkpoint recorded as expected, even if check 2
       passed -- e.g. a wrong, corrupted, or hand-edited manifest that
       happens to share the right export_fingerprint.
    """
    if not checkpoint.get("is_final_training", False):
        raise ValueError("checkpoint is not marked is_final_training=true; refusing locked-test evaluation")

    checkpoint_export_fingerprint = checkpoint.get("export_fingerprint")
    if manifest.export_fingerprint != checkpoint_export_fingerprint:
        raise ValueError(
            "locked-test manifest's export_fingerprint does not match the checkpoint's "
            f"({manifest.export_fingerprint!r} != {checkpoint_export_fingerprint!r}); refusing to evaluate"
        )

    expected_manifest_fingerprint = checkpoint.get("locked_test_manifest_fingerprint")
    actual_manifest_fingerprint = recompute_manifest_fingerprint(manifest)
    if actual_manifest_fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "locked-test manifest's recomputed fingerprint does not match the checkpoint's expected value "
            f"({actual_manifest_fingerprint!r} != {expected_manifest_fingerprint!r}); refusing to evaluate "
            "(the manifest may be the wrong one, or its content may have changed)"
        )
