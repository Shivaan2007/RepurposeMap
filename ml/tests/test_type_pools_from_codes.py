"""TypePools reconstructed graph-free, from codes -- equivalence with the graph-based builder."""

from __future__ import annotations

import numpy as np

from graph_builders import primekg_style_graph
from repurposemap.embeddings.entity_index import build_entity_index
from repurposemap.embeddings.negative_sampling import build_type_pools, build_type_pools_from_codes

TARGET = ("drug_protein", "target")


def _graph():
    types = {"D1": "drug", "D2": "drug", "P1": "protein", "DIS1": "disease"}
    edges = [(*TARGET, "D1", "P1")]
    return primekg_style_graph(types, edges)


def test_codes_based_pools_match_graph_based_pools_exactly():
    graph = _graph()
    entity_index = build_entity_index(graph)
    graph_pools = build_type_pools(graph, entity_index)

    node_type_names = sorted(graph_pools.pools_by_type.keys())
    name_to_code = {name: i for i, name in enumerate(node_type_names)}
    codes = np.array([name_to_code[t] for t in graph_pools.type_of_index], dtype=np.int8)

    codes_pools = build_type_pools_from_codes(codes, node_type_names)

    assert codes_pools.type_of_index == graph_pools.type_of_index
    assert codes_pools.pools_by_type == graph_pools.pools_by_type


def test_codes_based_pools_are_deterministic_and_sorted_by_type_name():
    codes = np.array([1, 0, 1, 2], dtype=np.int8)
    names = ["disease", "drug", "protein"]
    pools = build_type_pools_from_codes(codes, names)

    assert list(pools.pools_by_type.keys()) == sorted(pools.pools_by_type.keys())
    # codes = [1, 0, 1, 2] -> types = [drug, disease, drug, protein]
    assert pools.pools_by_type["drug"] == (0, 2)
    assert pools.pools_by_type["disease"] == (1,)
    assert pools.pools_by_type["protein"] == (3,)
