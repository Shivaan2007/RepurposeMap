"""Extract known drug-disease indication records from a graph.

Only the ``indication`` relation counts as a known treatment. Contraindications,
drug-drug interactions and weak graph associations (gene, phenotype, anatomy
links and the rest) are never treated as treatment positives, however strong
they look in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

INDICATION_RELATION = "indication"


@dataclass(frozen=True, order=True)
class IndicationRecord:
    """One known drug-disease indication. Sorted by disease, then drug, then ID."""

    disease_id: str
    drug_id: str
    disease_name: str = ""
    drug_name: str = ""


def extract_indications(graph: nx.MultiDiGraph) -> tuple[IndicationRecord, ...]:
    """Return every (drug, disease) pair linked by an ``indication`` edge.

    Canonical node IDs and display names are both kept. The direction checked
    is drug -> disease; PrimeKG stores the relation in both directions, so this
    does not miss any pair. The result is deduplicated and sorted, so it is
    deterministic.

    Raises:
        ValueError: a node referenced by an indication edge is missing its
            ``node_type`` or ``name`` attribute.
    """
    seen: set[tuple[str, str]] = set()
    records: list[IndicationRecord] = []
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("relation") != INDICATION_RELATION:
            continue
        if graph.nodes[source].get("node_type") != "drug" or graph.nodes[target].get("node_type") != "disease":
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        records.append(
            IndicationRecord(
                disease_id=target,
                drug_id=source,
                disease_name=graph.nodes[target].get("name", target),
                drug_name=graph.nodes[source].get("name", source),
            )
        )
    return tuple(sorted(records))
