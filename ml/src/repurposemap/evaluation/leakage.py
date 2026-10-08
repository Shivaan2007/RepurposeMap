"""Explicit leakage checks for the disease-based split and training graph.

These fail loudly: a violation raises, instead of silently producing an
optimistic evaluation. Run ``assert_no_leakage`` once per evaluation, right
after the training graph is built and before any ranking happens.
"""

from __future__ import annotations

import networkx as nx

from repurposemap.evaluation.split import DiseaseSplit

INDICATION_RELATION = "indication"


def find_remaining_test_indications(graph: nx.MultiDiGraph, split: DiseaseSplit) -> list[str]:
    """List every indication edge in ``graph`` that touches a test disease.

    An empty list means the training graph correctly removed them all.
    """
    test_ids = frozenset(split.test_disease_ids)
    violations = []
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("relation") != INDICATION_RELATION:
            continue
        if source in test_ids or target in test_ids:
            violations.append(f"indication edge {source} -> {target} touches a test disease")
    return violations


def find_remaining_test_pairs(graph: nx.MultiDiGraph, split: DiseaseSplit) -> list[str]:
    """List every held-out (drug, disease) test pair still connected by an indication edge."""
    violations = []
    for record in split.test_indications:
        if graph.has_edge(record.drug_id, record.disease_id):
            attrs = graph.get_edge_data(record.drug_id, record.disease_id)
            if any(a.get("relation") == INDICATION_RELATION for a in attrs.values()):
                violations.append(f"test pair {record.drug_name!r} -> {record.disease_name!r} still has an indication edge")
    return violations


def find_disease_set_overlap(split: DiseaseSplit) -> list[str]:
    """List any disease ID present in both the train and test sets."""
    overlap = set(split.train_disease_ids) & set(split.test_disease_ids)
    return [f"disease {disease_id!r} is in both train and test" for disease_id in sorted(overlap)]


def assert_no_leakage(training_graph: nx.MultiDiGraph, split: DiseaseSplit) -> None:
    """Run every leakage check. Raises ValueError, listing every violation, if any fail.

    Checks: no test disease has a remaining indication edge, no held-out test
    pair is still connected, and the train and test disease sets do not overlap.
    """
    violations = [
        *find_disease_set_overlap(split),
        *find_remaining_test_indications(training_graph, split),
        *find_remaining_test_pairs(training_graph, split),
    ]
    if violations:
        raise ValueError("leakage detected:\n" + "\n".join(f"  - {v}" for v in violations))
