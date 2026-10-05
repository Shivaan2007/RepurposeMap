"""Find short directed paths between two entities, using bounded search.

Paths follow edges only in their stored direction. Parallel edges between the
same two nodes (different relations) produce separate paths, so no relation
is dropped.

The search never enumerates every simple path in the graph. It runs a
backward breadth-first pass from the target, limited to ``max_length`` hops,
and uses those distances to prune a depth-limited walk from the source. The
walk stops when ``max_paths`` paths are found, or when a time or edge-check
budget runs out. In the last case the result is marked as truncated.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product

import networkx as nx

DEFAULT_MAX_PATH_LENGTH = 4
DEFAULT_MAX_PATHS = 10
DEFAULT_TIME_LIMIT_S = 20.0
DEFAULT_MAX_EDGE_CHECKS = 5_000_000

# How often the walk checks the clock. Checking on every edge is wasteful.
_CLOCK_CHECK_INTERVAL = 4096


@dataclass(frozen=True)
class PathStep:
    """One directed hop: ``source --relation--> target``.

    ``source`` and ``target`` are canonical node IDs. The ``*_name`` fields
    hold human-readable names for display. They are left out of equality, so
    two steps with the same IDs compare equal even if one has no name set.
    ``display_relation`` is the source's readable relation label, empty when
    the source has none.
    """

    source: str
    source_type: str
    relation: str
    target: str
    target_type: str
    source_name: str = field(default="", compare=False)
    target_name: str = field(default="", compare=False)
    display_relation: str = ""


@dataclass(frozen=True)
class PathSearchResult:
    """Paths found by a bounded search, and whether the search was cut short.

    ``truncated`` is True when a time or edge-check limit stopped the search
    before it finished. In that case an empty ``paths`` list does not prove
    that no path exists. ``stop_reason`` names the limit that was hit.
    """

    paths: list[list[PathStep]]
    truncated: bool = False
    stop_reason: str | None = None


def find_paths(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    *,
    max_length: int = DEFAULT_MAX_PATH_LENGTH,
    max_paths: int = DEFAULT_MAX_PATHS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    max_edge_checks: int = DEFAULT_MAX_EDGE_CHECKS,
) -> list[list[PathStep]]:
    """Return up to ``max_paths`` directed paths from ``source`` to ``target``.

    Paths are ordered shortest first. Within one length, the order follows the
    graph's insertion order and is deterministic. Each path has at most
    ``max_length`` hops.

    An empty list means no path was found. Use ``find_paths_detailed`` to
    tell "no path" apart from "search stopped at a limit".

    Raises:
        ValueError: an entity is not in the graph, source equals target, or a
            limit is below 1.
    """
    return find_paths_detailed(
        graph,
        source,
        target,
        max_length=max_length,
        max_paths=max_paths,
        time_limit_s=time_limit_s,
        max_edge_checks=max_edge_checks,
    ).paths


def find_paths_detailed(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    *,
    max_length: int = DEFAULT_MAX_PATH_LENGTH,
    max_paths: int = DEFAULT_MAX_PATHS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    max_edge_checks: int = DEFAULT_MAX_EDGE_CHECKS,
) -> PathSearchResult:
    """Like ``find_paths``, but also report whether a search limit was hit.

    Raises:
        ValueError: an entity is not in the graph, source equals target, or a
            limit is below 1.
    """
    _validate_endpoints(graph, source, target)
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")
    if max_paths < 1:
        raise ValueError(f"max_paths must be at least 1, got {max_paths}")
    if time_limit_s <= 0:
        raise ValueError(f"time_limit_s must be positive, got {time_limit_s}")
    if max_edge_checks < 1:
        raise ValueError(f"max_edge_checks must be at least 1, got {max_edge_checks}")

    budget = _Budget(time_limit_s, max_edge_checks)
    # distance[v] is v's hop distance to the target, known for every node within
    # `known_depth` hops. A path of length L needs distances up to L - 1, so the
    # map grows one level at a time, and only as far as the current length needs.
    # Short paths are therefore found before the wider backward pass starts.
    distance: dict[str, int] = {target: 0}
    frontier = [target]
    known_depth = 0

    results: list[list[PathStep]] = []
    for length in range(1, max_length + 1):
        while known_depth < length - 1 and budget.stop_reason is None:
            known_depth += 1
            frontier = _extend_distances(graph, distance, frontier, known_depth, budget)
        if budget.stop_reason is not None:
            break
        for node_path in _node_paths(graph, source, target, length, distance, budget):
            for steps in _expand_relations(graph, node_path):
                results.append(steps)
                if len(results) == max_paths:
                    return PathSearchResult(paths=results)
        if budget.stop_reason is not None:
            break
    return PathSearchResult(
        paths=results,
        truncated=budget.stop_reason is not None,
        stop_reason=budget.stop_reason,
    )


def format_path(steps: list[PathStep]) -> str:
    """Render a path for people to read, one entity and relation per line.

    Entities are shown by name. A step with no name set shows its ID instead.
    """
    if not steps:
        return ""
    lines = [_display_name(steps[0].source, steps[0].source_name)]
    for step in steps:
        relation = step.relation
        if step.display_relation:
            relation = f"{relation} [{step.display_relation}]"
        lines.append(f"  --{relation}-->")
        lines.append(_display_name(step.target, step.target_name))
    return "\n".join(lines)


class _Budget:
    """Tracks the time and edge-check limits shared by one search."""

    def __init__(self, time_limit_s: float, max_edge_checks: int) -> None:
        self._deadline = time.monotonic() + time_limit_s
        self._max_checks = max_edge_checks
        self._checks = 0
        self.stop_reason: str | None = None

    def charge(self) -> bool:
        """Count one edge check. Return False once a limit has been hit."""
        if self.stop_reason is not None:
            return False
        self._checks += 1
        if self._checks > self._max_checks:
            self.stop_reason = "edge check limit reached"
        elif self._checks % _CLOCK_CHECK_INTERVAL == 0 and time.monotonic() > self._deadline:
            self.stop_reason = "time limit reached"
        return self.stop_reason is None


def _validate_endpoints(graph: nx.MultiDiGraph, source: str, target: str) -> None:
    if source not in graph:
        raise ValueError(f"source entity not found: {source!r}")
    if target not in graph:
        raise ValueError(f"target entity not found: {target!r}")
    if source == target:
        raise ValueError("source and target must be different entities")


def _extend_distances(
    graph: nx.MultiDiGraph,
    distance: dict[str, int],
    frontier: list[str],
    hops: int,
    budget: _Budget,
) -> list[str]:
    """Add the nodes exactly ``hops`` steps from the target, reached backwards from ``frontier``.

    Returns the newly added nodes, which form the next frontier. If the budget
    runs out part-way, the map is incomplete and the caller must stop.
    """
    added: list[str] = []
    for node in frontier:
        for predecessor in graph.pred[node]:
            if not budget.charge():
                return added
            if predecessor not in distance:
                distance[predecessor] = hops
                added.append(predecessor)
    return added


def _node_paths(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    length: int,
    distance: dict[str, int],
    budget: _Budget,
) -> Iterator[list[str]]:
    """Yield node sequences of exactly ``length`` hops, with no repeated node."""
    yield from _walk(graph, [source], target, length, distance, budget)


def _walk(
    graph: nx.MultiDiGraph,
    path: list[str],
    target: str,
    remaining: int,
    distance: dict[str, int],
    budget: _Budget,
) -> Iterator[list[str]]:
    node = path[-1]
    for successor in graph.succ[node]:
        if not budget.charge():
            return
        if successor in path:
            continue
        if remaining == 1:
            if successor == target:
                yield path + [successor]
            continue
        # The target can only appear at the end of a path, so do not walk through it.
        if successor == target:
            continue
        if distance.get(successor, remaining) > remaining - 1:
            continue
        yield from _walk(graph, path + [successor], target, remaining - 1, distance, budget)


def _expand_relations(graph: nx.MultiDiGraph, node_path: list[str]) -> Iterator[list[PathStep]]:
    """Yield every relation-level path that follows one node sequence.

    Between two nodes there can be several edges, one per relation. Each
    combination of choices across the hops is a separate path. Edge options are
    sorted by relation, then display label, so the order is deterministic.
    """
    options_per_hop: list[list[PathStep]] = []
    for hop_source, hop_target in zip(node_path, node_path[1:], strict=False):
        edge_attrs = sorted(
            graph[hop_source][hop_target].values(),
            key=lambda attrs: (attrs["relation"], attrs.get("display_relation", "")),
        )
        options_per_hop.append(
            [_make_step(graph, hop_source, hop_target, attrs) for attrs in edge_attrs]
        )
    for combination in product(*options_per_hop):
        yield list(combination)


def _make_step(graph: nx.MultiDiGraph, source: str, target: str, attrs: dict) -> PathStep:
    return PathStep(
        source=source,
        source_type=graph.nodes[source]["node_type"],
        relation=attrs["relation"],
        target=target,
        target_type=graph.nodes[target]["node_type"],
        source_name=graph.nodes[source].get("name", ""),
        target_name=graph.nodes[target].get("name", ""),
        display_relation=attrs.get("display_relation", ""),
    )


def _display_name(node_id: str, name: str) -> str:
    return name or node_id
