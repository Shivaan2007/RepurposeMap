"""Tiny graphs in PrimeKG's attribute schema, for tests that must not load the full file."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

# (relation, display_relation, source_id, target_id)
EdgeSpec = tuple[str, str, str, str]


def primekg_style_graph(types: dict[str, str], edges: Iterable[EdgeSpec]) -> nx.MultiDiGraph:
    """Build a graph whose nodes carry name, node_type and external_id, as the PrimeKG adapter does.

    Node names equal their IDs, which keeps test paths easy to read.
    Edges are keyed by (relation, display_relation), matching the adapter.
    """
    graph = nx.MultiDiGraph()
    for node_id, node_type in types.items():
        graph.add_node(node_id, name=node_id, node_type=node_type, external_id="", provenance="")
    for relation, display, source, target in edges:
        graph.add_edge(source, target, key=(relation, display), relation=relation, display_relation=display)
    return graph
