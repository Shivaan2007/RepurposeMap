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
