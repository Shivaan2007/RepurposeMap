"""Compact, exact positive-triple membership, for negative-sampling collision
checks at full real-PrimeKG scale.

Why not a Python ``set``/``frozenset`` of ints: each element is a boxed
Python object plus per-entry hash-table overhead, which for the real
leakage-safe training set (~8.1 million triples) can run to several hundred
MB or more -- a real risk on an 8 GB machine already holding the loaded
PrimeKG graph (Stage B measured ~1.3 GB RSS for the graph alone). This
module instead packs every ``(h, r, t)`` triple into a single ``uint64``
code, stores all codes as one sorted NumPy array, and answers membership by
binary search (``np.searchsorted``) -- exact (no false positives or
negatives, unlike a Bloom filter), deterministic, and roughly 8 bytes per
triple plus small NumPy array overhead: ~65 MB for 8.1 million triples.

### The ``uint64`` bound, proven, not assumed

For the real PrimeKG counts (``n_entities = 129,375``, ``n_relations =
33``), the largest possible code is
``(h_max * n_relations + r_max) * n_entities + t_max`` with
``h_max = t_max = n_entities - 1 = 129374`` and ``r_max = n_relations - 1 =
32``::

    (129374 * 33 + 32) * 129375 + 129374 = 552,350,390,624

That is about ``5.52e11`` -- roughly 0.000003% of ``uint64``'s range (max
``~1.8e19``), and comfortably within even ``int64``'s range (``~9.22e18``).
There is no overflow risk at these, or substantially larger, entity/relation
counts. ``tests/test_positive_store.py`` checks this bound directly, rather
than trusting this comment alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def encode_triple(h: int, r: int, t: int, n_relations: int, n_entities: int) -> int:
    """Pack one ``(h, r, t)`` triple into a single integer code.

    Plain Python ints (arbitrary precision), so this scalar form never
    overflows regardless of scale -- used for one-off lookups, e.g. inside
    the negative sampler's per-draw checks. ``encode_triples_array`` is the
    vectorized NumPy form used to build a ``PositiveTripleStore``.
    """
    return (h * n_relations + r) * n_entities + t


def encode_triples_array(triples: np.ndarray, n_relations: int, n_entities: int) -> np.ndarray:
    """Vectorized form of ``encode_triple`` over an ``(n, 3)`` int array, as ``uint64``.

    Cast to ``uint64`` before multiplying, not after, so no intermediate
    step is computed in a narrower dtype that could wrap around.
    """
    h = triples[:, 0].astype(np.uint64)
    r = triples[:, 1].astype(np.uint64)
    t = triples[:, 2].astype(np.uint64)
    return (h * np.uint64(n_relations) + r) * np.uint64(n_entities) + t


@dataclass(frozen=True)
class PositiveTripleStore:
    """Exact, compact ``(h, r, t)`` membership: a sorted, unique ``uint64`` code array."""

    codes: np.ndarray  # sorted, unique, dtype uint64
    n_relations: int
    n_entities: int

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def nbytes(self) -> int:
        """Memory held by the code array, in bytes (~8 bytes per stored triple)."""
        return int(self.codes.nbytes)

    def contains(self, h: int, r: int, t: int) -> bool:
        """Exact membership for one triple, by binary search. O(log n)."""
        if len(self.codes) == 0:
            return False
        code = np.uint64(encode_triple(h, r, t, self.n_relations, self.n_entities))
        idx = np.searchsorted(self.codes, code)
        return bool(idx < len(self.codes) and self.codes[idx] == code)

    def contains_many(self, triples: np.ndarray) -> np.ndarray:
        """Vectorized membership over an ``(n, 3)`` int array; returns a bool array."""
        triples = np.asarray(triples).reshape(-1, 3)
        if len(self.codes) == 0:
            return np.zeros(len(triples), dtype=bool)
        codes = encode_triples_array(triples, self.n_relations, self.n_entities)
        idx = np.clip(np.searchsorted(self.codes, codes), 0, len(self.codes) - 1)
        return self.codes[idx] == codes


def build_positive_store(triples, n_relations: int, n_entities: int) -> PositiveTripleStore:
    """One ``PositiveTripleStore`` from a triple collection.

    Deterministic: ``np.unique`` both deduplicates and sorts, so the same
    set of triples always produces an identical codes array regardless of
    row order or duplicate rows in the input. ``triples`` may be an
    ``(n, 3)`` NumPy array (used as-is, no copy of the row data beyond the
    dtype cast) or any iterable of 3-element rows (e.g. a small list/set,
    as used in tests).
    """
    if isinstance(triples, np.ndarray):
        array = triples.astype(np.int64, copy=False).reshape(-1, 3)
    else:
        rows = list(triples)
        array = np.asarray(rows, dtype=np.int64).reshape(-1, 3) if rows else np.empty((0, 3), dtype=np.int64)

    if len(array) == 0:
        return PositiveTripleStore(codes=np.array([], dtype=np.uint64), n_relations=n_relations, n_entities=n_entities)

    codes = np.unique(encode_triples_array(array, n_relations, n_entities))
    return PositiveTripleStore(codes=codes, n_relations=n_relations, n_entities=n_entities)
