"""Load the internal edge-table schema into a NetworkX graph.

The internal schema has one row per relationship:

    source_name, source_type, relation, target_name, target_type

It is a stand-in for the real PrimeKG format, which will get its own adapter
once the real column layout has been inspected.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_name",
    "source_type",
    "relation",
    "target_name",
    "target_type",
)


def load_graph(csv_path: str | Path) -> nx.MultiDiGraph:
    """Read an edge-table CSV and return it as a directed multigraph.

    Nodes are keyed by entity name and carry a ``node_type`` attribute.
    Edges run from source to target and carry a ``relation`` attribute.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: a required column is missing, a required value is empty,
            the file has no edges, or one name is used with two node types.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph CSV not found: {path}")
    edges = _read_edge_table(path)
    return _build_graph(edges)


def _read_edge_table(path: Path) -> pd.DataFrame:
    """Read the CSV as text and validate its required columns and values."""
    try:
        # Read everything as text and keep "NA" and similar strings as real values.
        # Only truly empty cells should count as missing.
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path}: file is empty, expected a header row") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")

    df = df.loc[:, list(REQUIRED_COLUMNS)]
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].str.strip()

    empty_rows = df.index[(df == "").any(axis=1)]
    if len(empty_rows) > 0:
        # Line 1 is the header, so data row i sits on line i + 2.
        first_line = int(empty_rows[0]) + 2
        raise ValueError(
            f"{path}: {len(empty_rows)} row(s) have empty required values "
            f"(first at line {first_line})"
        )

    df = df.drop_duplicates(ignore_index=True)
    if df.empty:
        raise ValueError(f"{path}: contains no edges")
    return df


def _build_graph(edges: pd.DataFrame) -> nx.MultiDiGraph:
    """Build the graph from a validated edge table."""
    node_types = _collect_node_types(edges)

    graph = nx.MultiDiGraph()
    for name, node_type in node_types.items():
        # The synthetic format has no IDs, so the name is both the key and the display name.
        graph.add_node(name, node_type=node_type, name=name)
    for row in edges.itertuples(index=False):
        # Using the relation as the parallel-edge key keeps one edge per
        # (source, relation, target) and lets the same pair carry several relations.
        graph.add_edge(row.source_name, row.target_name, key=row.relation, relation=row.relation)
    return graph


def _collect_node_types(edges: pd.DataFrame) -> dict[str, str]:
    """Map each entity name to its type, rejecting names that have two types."""
    sources = edges[["source_name", "source_type"]].set_axis(["name", "type"], axis=1)
    targets = edges[["target_name", "target_type"]].set_axis(["name", "type"], axis=1)
    pairs = pd.concat([sources, targets]).drop_duplicates(ignore_index=True)

    types_per_name = pairs.groupby("name")["type"].nunique()
    conflicting = types_per_name[types_per_name > 1].index.tolist()
    if conflicting:
        raise ValueError(
            "Entity name(s) appear with more than one type: " + ", ".join(conflicting[:5])
        )
    return dict(zip(pairs["name"], pairs["type"], strict=True))
