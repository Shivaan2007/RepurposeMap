"""Build the training graph for disease-based evaluation.

Every indication edge of a held-out (test) disease is removed, in both
directions. Everything else stays: gene, protein, pathway, phenotype, anatomy
links, and even the indication edges of train diseases. The result simulates
"a disease with its biological knowledge, but no treatment information" for
every test disease, while leaving a model free to learn from train diseases'
real treatments.

The path baseline never uses indication edges anyway, because the relation
policy excludes them everywhere (see repurposemap.scoring.relation_policy). This
module removes them structurally as well, so the training graph is safe for any
future model, including one that would otherwise treat indication edges as
training signal.

``build_training_graph`` is used once, to verify there is no leakage before
ranking starts. For the per-disease ranking itself, use
``test_disease_indication_filter`` instead, and pass it as ``extra_filter`` to
``repurposemap.scoring.relation_policy.policy_view`` (or as ``extra_edge_filter``
to ``rank_drugs``), so its check is folded into the *same* filter closure
``rank_drugs`` already builds, instead of adding a second, separately nested
``subgraph_view`` on top. Stacking two views makes every edge access during a
path search pay for two filter layers instead of one; a profiled run showed
this doubling to be the dominant cost of disease-based evaluation, not the
path search itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import networkx as nx

INDICATION_RELATION = "indication"


def test_disease_indication_filter(
    graph: nx.MultiDiGraph, test_disease_ids: Iterable[str]
) -> Callable[[str, str, object], bool]:
    """Return the predicate ``build_training_graph`` wraps in a view.

    True means "keep this edge": its relation is not ``indication``, or
    neither endpoint is a test disease ID.
    """
    test_ids = frozenset(test_disease_ids)

    def keep(u: str, v: str, key: object) -> bool:
        attrs = graph[u][v][key]
        if attrs.get("relation") != INDICATION_RELATION:
            return True
        return u not in test_ids and v not in test_ids

    return keep


def build_training_graph(graph: nx.MultiDiGraph, test_disease_ids: Iterable[str]) -> nx.MultiDiGraph:
    """Return a view of ``graph`` with test diseases' indication edges removed.

    The result is a lazy view; no edges are copied. An edge is removed only
    when its relation is ``indication`` and at least one endpoint is a test
    disease ID, so non-indication edges touching a test disease are kept.

    This builds its own ``subgraph_view``, so use it for one-off checks (such
    as the leakage check) and not in a per-disease ranking hot loop; see
    ``test_disease_indication_filter`` for that case.
    """
    return nx.subgraph_view(graph, filter_edge=test_disease_indication_filter(graph, test_disease_ids))
