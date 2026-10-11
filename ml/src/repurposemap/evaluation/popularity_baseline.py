"""Popularity baseline: rank the complete drug universe by raw graph degree.

This is deliberately independent of the path baseline's own candidate
discovery. `rank_drugs` returns a bounded, non-exhaustive candidate list, so
tying a second baseline to it would mean that baseline could never do better
than the path search's own reach — and for the 13 of 100 diseases in the
completed run where the path search found zero candidates, it would have no
chance at all. Popularity instead ranks every drug node in the graph, for
every held-out disease, independent of whether the path search ever reached
it.

The degree used is raw total degree (in + out, every relation, parallel
edges counted), unfiltered by the relation policy. This is a deliberate
contrast with the path baseline, not a variant of it: a drug's many
drug_drug or side-effect edges are meant to count here even though the path
score excludes them as evidence. The definition is fixed before any result
is seen, and is not tuned against benchmark performance.

Degree is read from the training graph
(`repurposemap.evaluation.training_graph.build_training_graph`) — the same
already-tested, leakage-safe view used for the leakage check, not a second,
separately composed filter path. It has every test disease's indication
edges removed in both directions, so popularity cannot be inflated by a
held-out treatment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx


def compute_popularity_scores(training_graph: nx.MultiDiGraph, drug_ids: Iterable[str]) -> dict[str, int]:
    """Raw total degree of every ID in ``drug_ids``, read from ``training_graph``.

    Computed once per evaluation run. The result does not depend on which
    disease is being evaluated: ``training_graph`` already excludes every
    test disease's indication edges, not just the one currently being scored.
    """
    return {drug_id: training_graph.degree(drug_id) for drug_id in drug_ids}


def popularity_ranking(popularity_scores: Mapping[str, int], candidate_ids: Iterable[str]) -> list[str]:
    """Order ``candidate_ids`` by ``popularity_scores``, descending, ties broken by ID."""
    return sorted(candidate_ids, key=lambda drug_id: (-popularity_scores.get(drug_id, 0), drug_id))
