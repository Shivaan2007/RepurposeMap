"""Type-aware negative sampling for TransE training.

For a positive ``(h, r, t)``, the TransE margin objective needs a
"corrupted" negative triple formed by replacing ``h`` or ``t`` with a
different entity. Naive random replacement (any other entity in the whole
graph) typically produces nonsensical corruptions in a heterogeneous
biomedical graph -- e.g. a disease tail replaced by a protein -- which are
so obviously wrong that they carry little training signal. This module
instead replaces the corrupted endpoint with another entity of the *same*
``node_type``, a substantially more informative negative, reusing the same
node_type partitioning already used elsewhere in this codebase (e.g. the
drug universe in ``repurposemap.evaluation.disease_eval``).

Bounded retries, not infinite loops, and never a known positive. A
same-type candidate pool can be small, and the positive-triple set can be
dense enough relative to it that naive reject-and-resample could in
principle loop forever. Every corruption is bounded by ``max_retries``
attempts on the chosen side; if none succeeds, one more bounded attempt is
made on the *other* side; if that also fails, bounded untyped uniform
corruption over the whole entity set is tried as a last resort. If none of
these finds a triple outside the known-positive set, ``corrupt()`` returns
``None`` -- it never knowingly hands back a known positive as a training
negative, not even as a last resort. The caller (the training loop) is
responsible for skipping a ``None`` result rather than training on it.
Every fallback and every unresolved call is counted
(``n_type_aware_fallbacks``, ``n_unresolved_collisions``), so a training run
leaning on them heavily is visible, not silently masked.

Known-positive membership is checked via ``PositiveTripleStore``
(``repurposemap.embeddings.positive_store``), a compact sorted-array
structure, not a Python ``set``/``frozenset`` -- at real-PrimeKG scale
(~8.1 million training triples) a set of boxed Python ints can run to
several hundred MB or more, which is a real risk on an 8 GB machine already
holding the loaded graph.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from repurposemap.embeddings.entity_index import EntityIndex
from repurposemap.embeddings.positive_store import PositiveTripleStore, build_positive_store


@dataclass(frozen=True)
class TypePools:
    """``node_type -> tuple of entity indices of that type`` (sorted, deterministic)."""

    pools_by_type: dict[str, tuple[int, ...]]
    type_of_index: tuple[str, ...]


def _pools_from_type_sequence(type_of_index: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Shared grouping logic for both ``TypePools`` constructors below."""
    pools: dict[str, list[int]] = {}
    for index, node_type in enumerate(type_of_index):
        pools.setdefault(node_type, []).append(index)
    return {node_type: tuple(indices) for node_type, indices in sorted(pools.items())}


def build_type_pools(graph: nx.MultiDiGraph, entity_index: EntityIndex) -> TypePools:
    """Partition every entity by its ``node_type``, in index order, from a graph."""
    type_of_index = tuple(graph.nodes[node_id].get("node_type", "") for node_id in entity_index.index_to_id)
    return TypePools(pools_by_type=_pools_from_type_sequence(type_of_index), type_of_index=type_of_index)


def build_type_pools_from_codes(entity_node_type_codes, node_type_names: list[str]) -> TypePools:
    """The same ``TypePools`` representation, reconstructed with no graph.

    A second constructor, not a competing implementation: this and
    ``build_type_pools`` both populate the identical dataclass, from the
    same underlying grouping logic (``_pools_from_type_sequence``). Used by
    GPU-side training, which only has ``entity_node_type_codes`` (a small
    int8 array, one code per entity) and ``node_type_names`` (what each
    code means) from the export package -- never a NetworkX graph.
    """
    type_of_index = tuple(node_type_names[int(code)] for code in entity_node_type_codes)
    return TypePools(pools_by_type=_pools_from_type_sequence(type_of_index), type_of_index=type_of_index)


class TypeAwareNegativeSampler:
    """Draws one type-aware negative per call, or ``None`` if none can be found.

    Deterministic under a fixed ``seed``: uses its own ``random.Random``
    instance, never global ``random`` state, matching the discipline already
    used by ``repurposemap.evaluation.split.split_by_disease``.
    """

    def __init__(
        self,
        type_pools: TypePools,
        n_entities: int,
        n_relations: int,
        positive_triples,
        *,
        max_retries: int = 10,
        seed: int = 0,
    ) -> None:
        self._type_pools = type_pools
        self._n_entities = n_entities
        self._n_relations = n_relations
        # `positive_triples` may be a prebuilt PositiveTripleStore -- the
        # intended usage at real-PrimeKG scale, so the ~8.1M-row sorted
        # code array is built once and reused, not rebuilt per sampler --
        # or a raw triple collection (small lists/arrays, as in tests),
        # which is wrapped into a store here.
        if isinstance(positive_triples, PositiveTripleStore):
            self._positive_store = positive_triples
        else:
            self._positive_store = build_positive_store(positive_triples, n_relations, n_entities)
        self._max_retries = max_retries
        self._rng = random.Random(seed)
        self.n_type_aware_fallbacks = 0
        self.n_unresolved_collisions = 0

    @property
    def random_state(self):
        """``random.Random.getstate()`` -- for checkpointing this sampler's RNG."""
        return self._rng.getstate()

    @random_state.setter
    def random_state(self, state) -> None:
        self._rng.setstate(state)

    def _is_positive(self, h: int, r: int, t: int) -> bool:
        return self._positive_store.contains(h, r, t)

    def _try_type_aware(self, h: int, r: int, t: int, corrupt_head: bool) -> tuple[int, int, int] | None:
        original = h if corrupt_head else t
        node_type = self._type_pools.type_of_index[original]
        pool = self._type_pools.pools_by_type.get(node_type, ())
        if len(pool) <= 1:
            return None  # no alternative of this type exists at all
        for _ in range(self._max_retries):
            candidate = self._rng.choice(pool)
            if candidate == original:
                continue
            negative = (candidate, r, t) if corrupt_head else (h, r, candidate)
            if not self._is_positive(*negative):
                return negative
        return None

    def _uniform_fallback(self, h: int, r: int, t: int, corrupt_head: bool) -> tuple[int, int, int] | None:
        original = h if corrupt_head else t
        for _ in range(self._max_retries):
            candidate = self._rng.randrange(self._n_entities)
            if candidate == original:
                continue
            negative = (candidate, r, t) if corrupt_head else (h, r, candidate)
            if not self._is_positive(*negative):
                return negative
        # Bounded, not infinite -- but never a known positive either: if
        # every attempt within the retry budget collided with a positive,
        # this returns None rather than handing one back.
        return None

    def corrupt(self, h: int, r: int, t: int) -> tuple[int, int, int] | None:
        """One negative triple for positive ``(h, r, t)``, or ``None``.

        Corrupts head or tail (50/50, standard TransE practice), type-aware
        first, falling back to the other side, then to untyped uniform
        corruption, all within bounded retries -- never an unbounded loop,
        and never a return value that collides with a known positive
        triple. ``None`` means no valid negative was found within the
        retry budget; ``n_unresolved_collisions`` is incremented so this is
        visible, and the caller must skip this draw rather than train on a
        known positive mislabeled as a negative.
        """
        corrupt_head = self._rng.random() < 0.5
        negative = self._try_type_aware(h, r, t, corrupt_head)
        if negative is not None:
            return negative
        negative = self._try_type_aware(h, r, t, not corrupt_head)
        if negative is not None:
            return negative
        self.n_type_aware_fallbacks += 1
        negative = self._uniform_fallback(h, r, t, corrupt_head)
        if negative is not None:
            return negative
        self.n_unresolved_collisions += 1
        return None
