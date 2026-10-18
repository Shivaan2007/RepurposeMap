"""Batched type-aware negative sampling, for full-scale (real-PrimeKG) training.

``TypeAwareNegativeSampler`` (``negative_sampling.py``) draws one negative
per Python call -- correct and already-tested (Stage A-C), but a per-call
Python loop over millions of triples per epoch is the real throughput
bottleneck at full scale, independent of how fast a GPU can do the actual
forward/backward pass. This module vectorizes the common case: drawing a
whole batch's corruption candidates in one NumPy call and checking them
against the complete ``PositiveTripleStore`` in one vectorized
``contains_many`` call, falling back to the existing, already-tested scalar
``corrupt()`` only for the rows a vectorized attempt could not resolve
within its own retry budget.

This is a new, separately-tested sampling implementation, not a guaranteed
drop-in twin of the scalar sampler for the same seed -- batch-level RNG
draws do not correspond 1:1 to a sequence of scalar calls. What it
preserves exactly: type-aware replacement, exact rejection of known
positives via the compact store, bounded retries (never an unbounded
loop), and marking a row ``invalid`` (never emitting a known positive)
when no negative can be found at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler, TypePools
from repurposemap.embeddings.positive_store import PositiveTripleStore


@dataclass
class BatchedSamplerStats:
    """Explicit counters -- a training run leaning on fallbacks heavily is
    visible, not silently masked, same discipline as the scalar sampler.

    Precise semantics (each one is a distinct quantity, not a repackaging
    of another -- verified in ``tests/test_batched_negative_sampling.py``
    on small, hand-countable cases):

    - ``n_requested_negatives``: total rows asked for across every
      ``corrupt_batch`` call -- i.e. ``sum(len(h) for each call)``, the
      denominator everything else should be read against.
    - ``n_initial_collisions``: of those requested negatives, how many hit
      a known positive on their *first* vectorized draw (round 0 only) --
      a measurement of "how often blind sampling lands on a positive,"
      independent of how many retry rounds followed.
    - ``n_retry_rounds``: how many vectorized retry passes actually ran
      with at least one still-unresolved row (i.e. loop iterations, not
      rows) -- bounded by ``max_vectorized_retries`` per call.
    - ``n_retry_candidates_drawn``: the sum of how many *row positions*
      were (re)drawn across every retry round -- a single row that needs
      3 rounds to resolve contributes 3 here, not 1. This is what an
      earlier version of this dataclass called ``n_vectorized_retries``,
      which conflated "a row needed a retry" with "a candidate position
      was drawn in some round" and was ambiguous between the two; this
      name is specific about which one it is.
    - ``n_scalar_fallbacks``: rows that exhausted every vectorized round
      still unresolved and fell through to the scalar sampler.
    - ``n_unresolved``: of those scalar-fallback rows, how many even the
      scalar sampler could not resolve (``corrupt()`` returned ``None``) --
      these rows are masked out of the loss by the caller, never trained
      on as a fabricated negative.
    """

    n_requested_negatives: int = 0
    n_initial_collisions: int = 0
    n_retry_rounds: int = 0
    n_retry_candidates_drawn: int = 0
    n_scalar_fallbacks: int = 0
    n_unresolved: int = 0

    def to_dict(self) -> dict:
        return {
            "n_requested_negatives": self.n_requested_negatives,
            "n_initial_collisions": self.n_initial_collisions,
            "n_retry_rounds": self.n_retry_rounds,
            "n_retry_candidates_drawn": self.n_retry_candidates_drawn,
            "n_scalar_fallbacks": self.n_scalar_fallbacks,
            "n_unresolved": self.n_unresolved,
        }


class BatchedTypeAwareNegativeSampler:
    """Draws one negative per row of a batch, vectorized, with a scalar fallback.

    Deterministic for a fixed seed and a fixed, restored RNG state
    (``get_state()``/``set_state()``) -- the two RNG sources involved
    (a ``numpy.random.Generator`` for the vectorized path, and the
    existing scalar ``TypeAwareNegativeSampler`` for the fallback path) are
    both exposed for checkpointing.
    """

    def __init__(
        self,
        type_pools: TypePools,
        n_entities: int,
        n_relations: int,
        positive_store: PositiveTripleStore,
        *,
        max_vectorized_retries: int = 5,
        max_scalar_retries: int = 10,
        seed: int = 0,
    ) -> None:
        self._type_pools = type_pools
        self._n_entities = n_entities
        self._n_relations = n_relations
        self._positive_store = positive_store
        self._max_vectorized_retries = max_vectorized_retries
        self._rng = np.random.default_rng(seed)
        # The scalar fallback is the exact, already-tested Stage A-C
        # sampler -- reused, not reimplemented -- seeded independently
        # (derived offset) so its own sequence is deterministic too.
        self._scalar_sampler = TypeAwareNegativeSampler(
            type_pools, n_entities, n_relations, positive_store, max_retries=max_scalar_retries, seed=seed + 1
        )
        self._type_of_index = np.array(type_pools.type_of_index)
        self.stats = BatchedSamplerStats()

    def get_state(self) -> dict:
        """Both RNG sources' state, for checkpointing."""
        return {
            "numpy_bit_generator_state": self._rng.bit_generator.state,
            "scalar_random_state": self._scalar_sampler.random_state,
        }

    def set_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["numpy_bit_generator_state"]
        self._scalar_sampler.random_state = state["scalar_random_state"]

    def corrupt_batch(
        self, h: np.ndarray, r: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One negative per row of ``(h, r, t)`` (parallel int arrays).

        Returns ``(neg_h, neg_r, neg_t, valid_mask)``: ``neg_r`` is always
        ``r`` unchanged (only the head or tail is corrupted, standard
        TransE practice); ``valid_mask[i]`` is ``False`` exactly where no
        negative could be found within every retry budget -- those rows'
        ``neg_h``/``neg_t`` hold a harmless placeholder (the original
        triple's own value) and must be excluded from the loss by the
        caller, never trained on as if they were real negatives.
        """
        batch_n = len(h)
        self.stats.n_requested_negatives += batch_n
        corrupt_head = self._rng.random(batch_n) < 0.5
        original = np.where(corrupt_head, h, t)
        node_types = self._type_of_index[original]

        neg_h, neg_t = h.copy(), t.copy()
        valid_mask = np.zeros(batch_n, dtype=bool)
        unresolved = np.ones(batch_n, dtype=bool)

        for round_index in range(self._max_vectorized_retries):
            pending = np.where(unresolved)[0]
            if len(pending) == 0:
                break
            self.stats.n_retry_rounds += 1
            self.stats.n_retry_candidates_drawn += len(pending)

            candidates = np.empty(len(pending), dtype=np.int64)
            for i, row in enumerate(pending):
                pool = self._type_pools.pools_by_type.get(node_types[row], ())
                candidates[i] = (
                    original[row] if len(pool) <= 1 else pool[self._rng.integers(0, len(pool))]
                )  # len<=1: no real alternative; sentinel forces a collision below -> scalar fallback

            cand_h = np.where(corrupt_head[pending], candidates, h[pending])
            cand_t = np.where(corrupt_head[pending], t[pending], candidates)
            cand_r = r[pending]
            is_same_as_original = candidates == original[pending]
            is_positive = self._positive_store.contains_many(np.stack([cand_h, cand_r, cand_t], axis=1))

            if round_index == 0:
                self.stats.n_initial_collisions += int(np.count_nonzero(is_positive & ~is_same_as_original))
            resolved_now = ~is_same_as_original & ~is_positive

            resolved_rows = pending[resolved_now]
            neg_h[resolved_rows] = cand_h[resolved_now]
            neg_t[resolved_rows] = cand_t[resolved_now]
            valid_mask[resolved_rows] = True
            unresolved[resolved_rows] = False

        for row in np.where(unresolved)[0]:
            self.stats.n_scalar_fallbacks += 1
            result = self._scalar_sampler.corrupt(int(h[row]), int(r[row]), int(t[row]))
            if result is None:
                self.stats.n_unresolved += 1
                continue
            nh, _nr, nt = result
            neg_h[row] = nh
            neg_t[row] = nt
            valid_mask[row] = True

        return neg_h, r.copy(), neg_t, valid_mask
