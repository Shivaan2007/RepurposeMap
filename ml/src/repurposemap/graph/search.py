"""Find graph entities by name and return their canonical IDs."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

DEFAULT_SEARCH_LIMIT = 10


@dataclass(frozen=True)
class EntityMatch:
    """One entity returned by a name search.

    ``id`` is the canonical graph key. ``external_id`` is the identifier the
    source dataset uses for the entity, such as a DrugBank or MONDO ID. It is
    empty when the source has none.
    """

    id: str
    name: str
    node_type: str
    external_id: str = ""


def search_entities(
    graph: nx.MultiDiGraph,
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[EntityMatch]:
    """Return entities whose names contain ``query``, ignoring case.

    Exact name matches come first. The rest are sorted by name, then ID, so
    results are deterministic. At most ``limit`` results are returned.

    Raises:
        ValueError: the query is empty or whitespace, or ``limit`` is below 1.
    """
    needle = query.strip().casefold()
    if not needle:
        raise ValueError("search query must not be empty")
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    matches = [match for match in _all_matches(graph) if needle in match.name.casefold()]
    matches.sort(key=lambda match: (match.name.casefold() != needle, match.name.casefold(), match.id))
    return matches[:limit]


def find_exact_name_matches(graph: nx.MultiDiGraph, name: str) -> list[EntityMatch]:
    """Return every entity whose name equals ``name``, ignoring case and outer spaces.

    Several entities can share a name. The result is sorted by ID so that it
    is deterministic. An empty list means no entity has that name.
    """
    needle = name.strip().casefold()
    if not needle:
        raise ValueError("entity name must not be empty")
    matches = [match for match in _all_matches(graph) if match.name.casefold() == needle]
    return sorted(matches, key=lambda match: (match.name.casefold(), match.id))


def entity_match(graph: nx.MultiDiGraph, node_id: str) -> EntityMatch:
    """Describe one node by its canonical ID."""
    if node_id not in graph:
        raise ValueError(f"entity not found: {node_id!r}")
    return _match_for(node_id, graph.nodes[node_id])


def _all_matches(graph: nx.MultiDiGraph) -> list[EntityMatch]:
    return [_match_for(node_id, attrs) for node_id, attrs in graph.nodes(data=True)]


def _match_for(node_id: str, attrs: dict) -> EntityMatch:
    # Synthetic graphs key nodes by name and have no separate name attribute.
    return EntityMatch(
        id=str(node_id),
        name=attrs.get("name", str(node_id)),
        node_type=attrs["node_type"],
        external_id=attrs.get("external_id", ""),
    )
