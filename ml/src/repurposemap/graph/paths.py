"""Find short paths between two entities while preserving edge direction and relation names.

Paths follow edges only in their stored direction. Parallel edges between the
same two nodes (different relations) produce separate paths, so no relation
is dropped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

import networkx as nx

DEFAULT_MAX_PATH_LENGTH = 4
DEFAULT_MAX_PATHS = 10


@dataclass(frozen=True)
class PathStep:
    """One directed hop: ``source --relation--> target``."""

    source: str
    source_type: str
    relation: str
    target: str
    target_type: str


def find_paths(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    *,
    max_length: int = DEFAULT_MAX_PATH_LENGTH,
    max_paths: int = DEFAULT_MAX_PATHS,
) -> list[list[PathStep]]:
    """Return up to ``max_paths`` directed paths from ``source`` to ``target``.

    Paths are ordered shortest first. Within one length, the order follows the
    graph's insertion order and is deterministic. Each path has at most
    ``max_length`` hops.

    An empty list means no path exists within ``max_length``.

    Raises:
        ValueError: an entity is not in the graph, source equals target, or a
            limit is below 1.
    """
    _validate_endpoints(graph, source, target)
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}")
    if max_paths < 1:
        raise ValueError(f"max_paths must be at least 1, got {max_paths}")

    results: list[list[PathStep]] = []
    seen_node_paths: set[tuple[str, ...]] = set()
    for length in range(1, max_length + 1):
        # Walk node sequences of exactly this length. Shorter ones are skipped,
        # since they were already returned in an earlier pass.
        for node_path in nx.all_simple_paths(graph, source, target, cutoff=length):
            # On a MultiDiGraph, NetworkX yields a node sequence once per parallel
            # edge. Skip repeats here and expand relations ourselves.
            key = tuple(node_path)
            if len(node_path) != length + 1 or key in seen_node_paths:
                continue
            seen_node_paths.add(key)
            for steps in _expand_relations(graph, node_path):
                results.append(steps)
                if len(results) == max_paths:
                    return results
    return results


def format_path(steps: list[PathStep]) -> str:
    """Render a path for people to read, one entity and relation per line."""
    if not steps:
        return ""
    lines = [steps[0].source]
    for step in steps:
        lines.append(f"  --{step.relation}-->")
        lines.append(step.target)
    return "\n".join(lines)


def _validate_endpoints(graph: nx.MultiDiGraph, source: str, target: str) -> None:
    if source not in graph:
        raise ValueError(f"source entity not found: {source!r}")
    if target not in graph:
        raise ValueError(f"target entity not found: {target!r}")
    if source == target:
        raise ValueError("source and target must be different entities")


def _expand_relations(graph: nx.MultiDiGraph, node_path: list[str]) -> Iterator[list[PathStep]]:
    """Yield every relation-level path that follows one node sequence.

    Between two nodes there can be several edges, one per relation. Each
    combination of choices across the hops is a separate path.
    """
    options_per_hop: list[list[PathStep]] = []
    for hop_source, hop_target in zip(node_path, node_path[1:], strict=False):
        edge_attrs = sorted(graph[hop_source][hop_target].values(), key=lambda attrs: attrs["relation"])
        options_per_hop.append(
            [_make_step(graph, hop_source, hop_target, attrs["relation"]) for attrs in edge_attrs]
        )
    for combination in product(*options_per_hop):
        yield list(combination)


def _make_step(graph: nx.MultiDiGraph, source: str, target: str, relation: str) -> PathStep:
    return PathStep(
        source=source,
        source_type=graph.nodes[source]["node_type"],
        relation=relation,
        target=target,
        target_type=graph.nodes[target]["node_type"],
    )
