"""Deterministic entity and relation indexing for TransE.

Node identity everywhere else in this codebase is the PrimeKG node-index
string (see ``repurposemap.adapters.primekg``); determinism throughout
``repurposemap.evaluation`` comes from plain ``sorted()`` on those string
keys, never from Python's dict/set iteration order. This module builds the
one thing that does not already exist: a dense integer index TransE's
embedding tables need, built with the exact same discipline -- ``sorted()``
first, then ``enumerate()`` -- so the same graph produces the same index
regardless of node or edge insertion order.

Relations are keyed by the full ``(relation, display_relation)`` pair, the
same key the graph itself uses for parallel edges
(``repurposemap.adapters.primekg``), so two distinct relation labels are
never collapsed into one relation embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class EntityIndex:
    """Deterministic ``entity_id -> dense integer index``, and back.

    ``index_to_id[id_to_index[node_id]] == node_id`` for every node_id.
    """

    id_to_index: dict[str, int]
    index_to_id: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.index_to_id)

    def to_dict(self) -> dict:
        """Plain-JSON-serializable form, for a model checkpoint.

        Only ``index_to_id`` needs to be stored -- ``id_to_index`` is fully
        determined by it via ``enumerate()`` -- so round-tripping through
        this form and rebuilding ``id_to_index`` is exact, not approximate.
        """
        return {"index_to_id": list(self.index_to_id)}

    @classmethod
    def from_dict(cls, data: dict) -> EntityIndex:
        index_to_id = tuple(data["index_to_id"])
        return cls(id_to_index={node_id: i for i, node_id in enumerate(index_to_id)}, index_to_id=index_to_id)


def build_entity_index(graph: nx.MultiDiGraph) -> EntityIndex:
    """One deterministic index over every node in ``graph``.

    Sorted by node ID string before indexing, so two graphs with identical
    node sets built in different insertion orders produce an identical
    index.
    """
    ids = tuple(sorted(graph.nodes))
    return EntityIndex(id_to_index={node_id: i for i, node_id in enumerate(ids)}, index_to_id=ids)


RelationKey = tuple[str, str]


@dataclass(frozen=True)
class RelationIndex:
    """Deterministic ``(relation, display_relation) -> dense integer index``, and back."""

    id_to_index: dict[RelationKey, int]
    index_to_id: tuple[RelationKey, ...]

    def __len__(self) -> int:
        return len(self.index_to_id)

    def to_dict(self) -> dict:
        """Plain-JSON-serializable form (tuples become 2-element lists)."""
        return {"index_to_id": [list(pair) for pair in self.index_to_id]}

    @classmethod
    def from_dict(cls, data: dict) -> RelationIndex:
        index_to_id = tuple((pair[0], pair[1]) for pair in data["index_to_id"])
        return cls(id_to_index={pair: i for i, pair in enumerate(index_to_id)}, index_to_id=index_to_id)


def build_relation_index(graph: nx.MultiDiGraph) -> RelationIndex:
    """One deterministic index over every ``(relation, display_relation)`` pair in ``graph``."""
    pairs = tuple(sorted({(d["relation"], d["display_relation"]) for _, _, d in graph.edges(data=True)}))
    return RelationIndex(id_to_index={pair: i for i, pair in enumerate(pairs)}, index_to_id=pairs)


def resolve_relation_key(relation_index: RelationIndex, relation: str) -> int:
    """The unique relation index whose pair's ``relation`` half equals ``relation``.

    ``RelationIndex`` keys relations by the full ``(relation,
    display_relation)`` pair, so code that needs "the indication relation"
    must not look up a loose string -- that is not a valid key. This
    searches the index instead of assuming any particular tuple shape (e.g.
    that it is exactly ``("indication", "indication")``), and raises rather
    than guessing if the match is not unique.

    Raises:
        ValueError: zero, or more than one, pair has this ``relation`` half.
    """
    matches = [pair for pair in relation_index.id_to_index if pair[0] == relation]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one relation pair with relation={relation!r}, found {matches}")
    return relation_index.id_to_index[matches[0]]
