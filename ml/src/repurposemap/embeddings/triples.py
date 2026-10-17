"""Integer triples for TransE, built from a graph the caller already chose.

This module does not know about diseases, splits, or leakage. It turns
whatever graph it is handed into ``(head_index, relation_index,
tail_index)`` triples, using an ``EntityIndex``/``RelationIndex`` already
built for that graph. In production, the caller passes
``repurposemap.evaluation.training_graph.build_training_graph(graph,
removed_disease_ids)`` -- the same leakage-safe view every other baseline
trains/evaluates against -- so a held-out disease's indication edges (in
either direction) are never turned into triples; this module has no second,
competing leakage check of its own, and relies entirely on being handed the
right graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from repurposemap.embeddings.entity_index import EntityIndex, RelationIndex


@dataclass(frozen=True)
class TripleSet:
    """``triples`` is an ``(n, 3)`` int64 array of ``(head_idx, relation_idx, tail_idx)``.

    ``n_self_loops_dropped`` counts edges where ``head == tail``, excluded
    from ``triples``. A self-loop carries no relational information for
    TransE (scoring it would need ``e_r`` to behave like a zero vector
    regardless of which relation it is), so it is dropped by default -- but
    the count is reported, not silenced, so an unexpectedly large number is
    visible rather than silently absorbed.
    """

    triples: np.ndarray
    n_self_loops_dropped: int

    def __len__(self) -> int:
        return len(self.triples)


def build_triples(graph: nx.MultiDiGraph, entity_index: EntityIndex, relation_index: RelationIndex) -> TripleSet:
    """One triple per edge in ``graph``, using the given indices.

    Two edges between the same node pair under different ``(relation,
    display_relation)`` keys produce two distinct triples -- they are
    semantically different edges, never merged. Exact duplicate rows are
    already deduplicated upstream, by the PrimeKG loader
    (``repurposemap.adapters.primekg``), not here.
    """
    rows: list[tuple[int, int, int]] = []
    n_self_loops_dropped = 0
    for source, target, data in graph.edges(data=True):
        if source == target:
            n_self_loops_dropped += 1
            continue
        head = entity_index.id_to_index[source]
        tail = entity_index.id_to_index[target]
        relation = relation_index.id_to_index[(data["relation"], data["display_relation"])]
        rows.append((head, relation, tail))

    triples = np.array(rows, dtype=np.int64).reshape(-1, 3)
    return TripleSet(triples=triples, n_self_loops_dropped=n_self_loops_dropped)
