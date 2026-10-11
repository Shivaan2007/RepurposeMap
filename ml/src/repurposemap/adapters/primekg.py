"""Adapter for the real PrimeKG edge file, ``kg.csv``.

PrimeKG stores one row per edge with twelve columns. The node key is the
``*_index`` column, which is unique for each node in the file. The ``*_id``
columns are the source's own identifiers. They are not unique on their own:
some IDs are shared across node types, and some gene IDs are shared within a
type. So the index is the graph key, and the ID is kept as ``external_id``.

Edge direction is kept exactly as stored. PrimeKG already lists every relation
in both directions, so this adapter does not add or remove any edges except
exact duplicate rows.

Nothing is invented here. A value that is missing from the file is an error,
not a default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PRIMEKG_COLUMNS: tuple[str, ...] = (
    "relation",
    "display_relation",
    "x_index",
    "x_id",
    "x_type",
    "x_name",
    "x_source",
    "y_index",
    "y_id",
    "y_type",
    "y_name",
    "y_source",
)

DEFAULT_PRIMEKG_CSV = Path("data") / "raw" / "primekg" / "kg.csv"

# Rows read per chunk. Chunks keep the peak memory of the text parse low.
CHUNK_ROWS = 1_000_000

# Bytes read per chunk when hashing. Independent of CHUNK_ROWS: this is a plain
# byte stream, not CSV parsing.
HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class PrimeKGFingerprint:
    """Dataset provenance for one PrimeKG file, attached to benchmark results.

    ``sha256`` and ``size_bytes`` are computed read-only from the file itself
    (see ``compute_file_fingerprint``) and are stable: hashing the same file
    twice gives the same digest. ``retrieved_date`` and ``dataverse_doi`` are
    recorded by hand (see ml/README.md's data section) because PrimeKG's
    distribution does not embed them in the file.

    ``release_version`` and ``snapshot_build_date`` are ``None`` because
    neither is available anywhere local to this repository. This was already
    attempted once (see ml/README.md): the Harvard Dataverse page at
    ``dataverse_doi`` could not be reached from this environment when the
    file was retrieved. That page is the source to check to fill these in;
    do not infer either from the local file's filesystem timestamp, which
    reflects when a copy last changed on this disk, not when the dataset was
    built or released.
    """

    sha256: str
    size_bytes: int
    retrieved_date: str = "2026-10-05"
    dataverse_doi: str = "doi:10.7910/DVN/IXA7BM"
    release_version: str | None = None
    snapshot_build_date: str | None = None


def compute_file_fingerprint(path: str | Path) -> PrimeKGFingerprint:
    """Hash ``path`` and record its size. Read-only; never modifies the file.

    Not called by ``load_primekg``: hashing a file this size costs real
    seconds, and every other command (``stats``, ``search``, ``paths``,
    ``rank-drugs``) must not pay that cost on every load. Call this
    explicitly, once, when building a benchmark result artifact.

    Raises:
        FileNotFoundError: the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"PrimeKG CSV not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return PrimeKGFingerprint(sha256=digest.hexdigest(), size_bytes=file_path.stat().st_size)


def load_primekg(csv_path: str | Path) -> nx.MultiDiGraph:
    """Read PrimeKG's ``kg.csv`` and return a directed multigraph.

    Nodes are keyed by PrimeKG's node index, as a string. Each node carries:
    ``name``, ``node_type``, ``external_id`` (PrimeKG's ``*_id``) and
    ``provenance`` (PrimeKG's ``*_source``).

    Edges run from ``x`` to ``y`` as stored. The edge key is the pair
    ``(relation, display_relation)``, so one node pair can carry several
    relations, and two labels for the same relation stay separate. Each edge
    has ``relation`` and ``display_relation`` attributes.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: a required column is missing, a required value is empty,
            an index is not an integer, a node index has conflicting
            attributes, or the file contains no edges.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"PrimeKG CSV not found: {path}")
    _check_header(path)

    nodes, edges, relation_names, display_names = _read_chunks(path)
    if edges.empty:
        raise ValueError(f"{path}: contains no edges")
    return _build_graph(nodes, edges, relation_names, display_names)


def _check_header(path: Path) -> None:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path}: file is empty, expected a header row") from exc
    missing = [column for column in PRIMEKG_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{path}: missing PrimeKG column(s): {', '.join(missing)}")


def _read_chunks(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Parse the file chunk by chunk into node rows and integer-coded edge rows."""
    relation_vocab: dict[str, int] = {}
    display_vocab: dict[str, int] = {}
    node_parts: list[pd.DataFrame] = []
    edge_parts: list[pd.DataFrame] = []
    rows_before = 0

    reader = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=list(PRIMEKG_COLUMNS),
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        chunk = chunk.apply(lambda column: column.str.strip())
        _reject_empty_values(path, chunk, rows_before)
        rows_before += len(chunk)

        node_parts.append(_nodes_from_chunk(path, chunk))
        edge_parts.append(
            pd.DataFrame(
                {
                    "relation": _encode(chunk["relation"], relation_vocab),
                    "display": _encode(chunk["display_relation"], display_vocab),
                    "x": _to_int(path, chunk["x_index"], "x_index"),
                    "y": _to_int(path, chunk["y_index"], "y_index"),
                }
            )
        )

    nodes = _merge_nodes(path, node_parts)
    edges = pd.concat(edge_parts, ignore_index=True).drop_duplicates(ignore_index=True)
    return nodes, edges, list(relation_vocab), list(display_vocab)


def _reject_empty_values(path: Path, chunk: pd.DataFrame, rows_before: int) -> None:
    empty = (chunk == "").any(axis=1)
    if empty.any():
        # Line 1 is the header, so data row i sits on line i + 2.
        first_line = rows_before + int(np.argmax(empty.to_numpy())) + 2
        raise ValueError(
            f"{path}: {int(empty.sum())} row(s) have empty values (first at line {first_line})"
        )


def _nodes_from_chunk(path: Path, chunk: pd.DataFrame) -> pd.DataFrame:
    """Return the unique nodes in one chunk, from both the x and the y side."""
    sides = []
    for side in ("x", "y"):
        side_frame = chunk[
            [f"{side}_index", f"{side}_id", f"{side}_type", f"{side}_name", f"{side}_source"]
        ].set_axis(["node_index", "external_id", "node_type", "name", "provenance"], axis=1)
        sides.append(side_frame)
    nodes = pd.concat(sides, ignore_index=True).drop_duplicates(ignore_index=True)
    nodes["node_index"] = _to_int(path, nodes["node_index"], "node index")
    return nodes


def _merge_nodes(path: Path, node_parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine chunk-level node rows and check each index has one set of attributes."""
    nodes = pd.concat(node_parts, ignore_index=True).drop_duplicates(ignore_index=True)
    attributes = ["external_id", "node_type", "name", "provenance"]
    conflicts = nodes.groupby("node_index")[attributes].nunique().max(axis=1)
    if (conflicts > 1).any():
        bad = int(conflicts[conflicts > 1].index[0])
        raise ValueError(f"{path}: node index {bad} has conflicting attributes")
    return nodes.drop_duplicates(subset="node_index", ignore_index=True)


def _encode(values: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Map strings to small integer codes, growing ``vocab`` as new values appear."""
    for value in values.unique():
        vocab.setdefault(value, len(vocab))
    return values.map(vocab).to_numpy(dtype=np.int64)


def _to_int(path: Path, values: pd.Series, column: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        bad = values[numeric.isna()].iloc[0]
        raise ValueError(f"{path}: {column} is not an integer: {bad!r}")
    return numeric.to_numpy(dtype=np.int64)


def _build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    relation_names: list[str],
    display_names: list[str],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    # Node keys are strings, so that they match the string IDs the CLI takes.
    keys = {index: str(index) for index in nodes["node_index"]}
    graph.add_nodes_from(
        (
            keys[row.node_index],
            {
                "name": row.name,
                "node_type": row.node_type,
                "external_id": row.external_id,
                "provenance": row.provenance,
            },
        )
        for row in nodes.itertuples(index=False)
    )

    pair_keys = {
        (r, d): (relation_names[r], display_names[d])
        for r, d in set(zip(edges["relation"].tolist(), edges["display"].tolist(), strict=True))
    }
    graph.add_edges_from(
        (
            keys[x],
            keys[y],
            pair_keys[(r, d)],
            {"relation": relation_names[r], "display_relation": display_names[d]},
        )
        for x, y, r, d in zip(
            edges["x"].tolist(),
            edges["y"].tolist(),
            edges["relation"].tolist(),
            edges["display"].tolist(),
            strict=True,
        )
    )
    return graph
