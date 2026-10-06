"""Drug-side promiscuity penalty: penalise drugs with very many protein targets.

A drug with many binding targets gets more paths to a disease almost by
construction, because each target is another chance to reach a disease gene.
That makes the drug score favour promiscuous compounds, such as zinc or copper,
over drugs with a few known mechanisms. The path hub penalty cannot catch this,
because it only looks at intermediate nodes, and a drug is an endpoint.

The count is the number of distinct proteins linked to the drug by a ``target``
edge (``drug_protein`` with display ``target``). Enzyme, transporter and carrier
roles are not counted. Thresholds come from PrimeKG's drug distribution: the 99th
percentile of target count is 26, and about 100 is well above that.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import networkx as nx

TARGET_RELATION = "drug_protein"
TARGET_DISPLAY = "target"

# Above the 99th percentile of PrimeKG drugs (26 targets).
PROMISCUOUS_TARGET_THRESHOLD = 26
# Far above the 99th percentile. Only a handful of PrimeKG drugs reach this.
VERY_PROMISCUOUS_TARGET_THRESHOLD = 100

PROMISCUITY_PENALTY = 0.25
VERY_PROMISCUITY_PENALTY = 0.75


class DrugTargetCounts(Mapping[str, int]):
    """Distinct ``target`` proteins per node. Counted on first use, then cached.

    Only drug nodes are counted. Any other node gets 0, and so does a drug with
    no target edge.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph
        self._cache: dict[str, int] = {}

    def __getitem__(self, node: str) -> int:
        if node not in self._cache:
            if node not in self._graph:
                raise KeyError(node)
            self._cache[node] = self._count(node)
        return self._cache[node]

    def __iter__(self) -> Iterator[str]:
        return iter(self._graph.nodes)

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def _count(self, node: str) -> int:
        if self._graph.nodes[node].get("node_type") != "drug":
            return 0
        targets = {
            target
            for target, edges in self._graph.succ[node].items()
            if any(
                attrs["relation"] == TARGET_RELATION and attrs.get("display_relation") == TARGET_DISPLAY
                for attrs in edges.values()
            )
        }
        return len(targets)


def drug_promiscuity_penalty(target_count: int) -> float:
    """Penalty subtracted from a drug's score for ``target_count`` distinct targets."""
    if target_count > VERY_PROMISCUOUS_TARGET_THRESHOLD:
        return VERY_PROMISCUITY_PENALTY
    if target_count > PROMISCUOUS_TARGET_THRESHOLD:
        return PROMISCUITY_PENALTY
    return 0.0
