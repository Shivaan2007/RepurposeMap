"""Hub handling: penalise intermediate nodes that connect to almost everything.

A hub is measured by its number of distinct neighbours over the edges the relation
policy allows. The count is taken on allowed edges only, so drug-drug edges do
not make a drug look like a hub. Thresholds come from the PrimeKG degree
distribution (see ml/README.md), not from node names.

Only intermediate nodes of a path are penalised. The two endpoints are the
question being asked, so a popular drug or disease is not penalised for being
popular.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import networkx as nx

from repurposemap.scoring.relation_policy import RelationPolicy, policy_view

# Measured on PrimeKG after the policy filter: p99 of distinct allowed neighbours
# is about 300. Above that a node is a hub.
HUB_NEIGHBOUR_THRESHOLD = 300
# About the top 0.1% of PrimeKG nodes (151 nodes above 2000). Above that a node
# is an extreme hub, such as "protein binding" or "multi-cellular organism".
EXTREME_HUB_NEIGHBOUR_THRESHOLD = 2000

HUB_PENALTY = 0.25
EXTREME_HUB_PENALTY = 0.75


class NeighbourCounts(Mapping[str, int]):
    """Distinct neighbour count per node, over edges the policy allows.

    Neighbours are counted in either direction, so a node's edges to the same
    neighbour count once. Counts are computed on first use and cached. Scoring
    only looks up the nodes on a path, so the full graph is never scanned.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        policy: RelationPolicy,
        extra_filter: Callable[[str, str, tuple[str, str]], bool] | None = None,
    ) -> None:
        self._view = policy_view(graph, policy, extra_filter=extra_filter)
        self._cache: dict[str, int] = {}

    def __getitem__(self, node: str) -> int:
        if node not in self._cache:
            if node not in self._view:
                raise KeyError(node)
            self._cache[node] = len(set(self._view.succ[node]) | set(self._view.pred[node]))
        return self._cache[node]

    def __iter__(self) -> Iterator[str]:
        return iter(self._view.nodes)

    def __len__(self) -> int:
        return self._view.number_of_nodes()


def hub_penalty(neighbour_count: int) -> float:
    """Penalty for one intermediate node with ``neighbour_count`` allowed neighbours."""
    if neighbour_count > EXTREME_HUB_NEIGHBOUR_THRESHOLD:
        return EXTREME_HUB_PENALTY
    if neighbour_count > HUB_NEIGHBOUR_THRESHOLD:
        return HUB_PENALTY
    return 0.0
