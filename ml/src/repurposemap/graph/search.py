"""Find graph entities by name."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

DEFAULT_SEARCH_LIMIT = 10


@dataclass(frozen=True)
class EntityMatch:
    """One entity returned by a name search."""

    name: str
    node_type: str


def search_entities(
    graph: nx.MultiDiGraph,
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[EntityMatch]:
    """Return entities whose names contain ``query``, ignoring case.

    Exact name matches come first. The rest are sorted by name, so results
    are deterministic. At most ``limit`` results are returned.

    Raises:
        ValueError: the query is empty or whitespace, or ``limit`` is below 1.
    """
    needle = query.strip().casefold()
    if not needle:
        raise ValueError("search query must not be empty")
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    matches = [
        EntityMatch(name=name, node_type=node_type)
        for name, node_type in graph.nodes(data="node_type")
        if needle in name.casefold()
    ]
    matches.sort(key=lambda match: (match.name.casefold() != needle, match.name.casefold(), match.name))
    return matches[:limit]
