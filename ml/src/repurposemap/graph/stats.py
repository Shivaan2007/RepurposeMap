"""Summary statistics for a loaded graph.

Every function returns structured data. Printing is left to the caller.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class GraphStats:
    """Summary counts for a graph. Dictionaries are sorted by key."""

    n_nodes: int
    n_edges: int
    node_type_counts: dict[str, int]
    relation_counts: dict[str, int]


def compute_stats(graph: nx.MultiDiGraph) -> GraphStats:
    """Return total node and edge counts plus counts by node type and relation."""
    return GraphStats(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        node_type_counts=count_nodes_by_type(graph),
        relation_counts=count_edges_by_relation(graph),
    )


def count_nodes_by_type(graph: nx.MultiDiGraph) -> dict[str, int]:
    """Count nodes for each ``node_type`` value, sorted by type name."""
    counts = Counter(node_type for _, node_type in graph.nodes(data="node_type"))
    return dict(sorted(counts.items()))


def count_edges_by_relation(graph: nx.MultiDiGraph) -> dict[str, int]:
    """Count edges for each ``relation`` value, sorted by relation name."""
    counts = Counter(relation for _, _, relation in graph.edges(data="relation"))
    return dict(sorted(counts.items()))
