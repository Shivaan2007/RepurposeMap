"""Batched type-aware negative sampling (Stage D)."""

from __future__ import annotations

import numpy as np

from repurposemap.embeddings.batched_negative_sampling import BatchedTypeAwareNegativeSampler
from repurposemap.embeddings.negative_sampling import TypePools
from repurposemap.embeddings.positive_store import build_positive_store

TYPE_OF_INDEX = ("drug", "drug", "drug", "protein", "protein", "disease")
POOLS = {"drug": (0, 1, 2), "protein": (3, 4), "disease": (5,)}


def _type_pools() -> TypePools:
    return TypePools(pools_by_type=POOLS, type_of_index=TYPE_OF_INDEX)


def _sampler(positives, seed=1, **kwargs) -> BatchedTypeAwareNegativeSampler:
    store = build_positive_store(positives, n_relations=2, n_entities=6)
    return BatchedTypeAwareNegativeSampler(_type_pools(), n_entities=6, n_relations=2, positive_store=store, seed=seed, **kwargs)


def test_corrupted_entities_match_original_node_types():
    sampler = _sampler([[0, 0, 3]])
    h = np.array([0, 1, 2] * 50)
    r = np.zeros_like(h)
    t = np.array([3, 4, 3] * 50)

    neg_h, neg_r, neg_t, valid = sampler.corrupt_batch(h, r, t)

    assert np.array_equal(neg_r, r)
    for i in range(len(h)):
        if not valid[i]:
            continue
        if neg_h[i] != h[i]:
            assert TYPE_OF_INDEX[neg_h[i]] == TYPE_OF_INDEX[h[i]]
        if neg_t[i] != t[i]:
            assert TYPE_OF_INDEX[neg_t[i]] == TYPE_OF_INDEX[t[i]]


def test_known_positives_are_never_emitted_as_valid_negatives():
    positives = np.array([[0, 0, 3], [1, 0, 4], [2, 0, 3]])
    store = build_positive_store(positives, n_relations=2, n_entities=6)
    sampler = BatchedTypeAwareNegativeSampler(_type_pools(), n_entities=6, n_relations=2, positive_store=store, seed=3)

    h = np.array([0, 1, 2] * 100)
    r = np.zeros_like(h)
    t = np.array([3, 4, 3] * 100)
    neg_h, neg_r, neg_t, valid = sampler.corrupt_batch(h, r, t)

    for i in range(len(h)):
        if valid[i]:
            assert not store.contains(int(neg_h[i]), int(neg_r[i]), int(neg_t[i]))


def test_deterministic_for_a_fixed_seed_and_fresh_state():
    positives = [[0, 0, 3]]
    sampler_a = _sampler(positives, seed=7)
    sampler_b = _sampler(positives, seed=7)
    h = np.array([0, 1, 2, 0, 1])
    r = np.zeros_like(h)
    t = np.array([3, 4, 3, 4, 3])

    result_a = sampler_a.corrupt_batch(h, r, t)
    result_b = sampler_b.corrupt_batch(h, r, t)
    for arr_a, arr_b in zip(result_a, result_b, strict=True):
        assert np.array_equal(arr_a, arr_b)


def test_state_round_trip_reproduces_subsequent_draws():
    positives = [[0, 0, 3]]
    sampler = _sampler(positives, seed=11)
    h = np.array([0, 1, 2])
    r = np.zeros_like(h)
    t = np.array([3, 4, 3])

    sampler.corrupt_batch(h, r, t)  # advance the RNG state once
    saved_state = sampler.get_state()
    next_from_original = sampler.corrupt_batch(h, r, t)

    resumed = _sampler(positives, seed=999)  # different seed -- state restore must override it
    resumed.set_state(saved_state)
    next_from_resumed = resumed.corrupt_batch(h, r, t)

    for arr_a, arr_b in zip(next_from_original, next_from_resumed, strict=True):
        assert np.array_equal(arr_a, arr_b)


def test_unresolved_rows_are_marked_invalid_not_a_fabricated_negative():
    # Every possible triple over this 2-entity, 1-relation universe is a
    # known positive -- no valid negative can exist at all.
    pools = TypePools(pools_by_type={"a": (0, 1)}, type_of_index=("a", "a"))
    all_possible = [[a, 0, b] for a in (0, 1) for b in (0, 1)]
    store = build_positive_store(all_possible, n_relations=1, n_entities=2)
    sampler = BatchedTypeAwareNegativeSampler(
        pools, n_entities=2, n_relations=1, positive_store=store, max_vectorized_retries=2, max_scalar_retries=3, seed=0
    )

    h = np.array([0, 1] * 10)
    r = np.zeros_like(h)
    t = np.array([1, 0] * 10)
    neg_h, neg_r, neg_t, valid = sampler.corrupt_batch(h, r, t)

    assert not valid.any()
    stats = sampler.stats
    assert stats.n_requested_negatives == len(h)
    # Every row has exactly two possible draws here -- "same as original"
    # (not a collision) or "the other entity" (a known positive by
    # construction, since every triple in this 2-entity universe is
    # positive) -- so no row EVER resolves, in any round: both counts are
    # therefore exact, not just bounded.
    assert stats.n_retry_rounds == 2  # == max_vectorized_retries exactly
    assert stats.n_retry_candidates_drawn == len(h) * 2  # every row reprocessed every round
    assert stats.n_scalar_fallbacks == len(h)
    assert stats.n_unresolved == len(h)


def test_retry_rounds_and_candidates_drawn_are_exactly_one_when_capped_at_one():
    # max_vectorized_retries=1 deterministically runs exactly one round
    # regardless of whether that round resolves anything -- an invariant
    # of the loop bound itself, not of the RNG outcome, so this holds for
    # any seed.
    sampler = _sampler([[0, 0, 3]], seed=2, max_vectorized_retries=1)
    h = np.array([0, 1, 2] * 10)
    r = np.zeros_like(h)
    t = np.array([3, 4, 3] * 10)
    sampler.corrupt_batch(h, r, t)

    stats = sampler.stats
    assert stats.n_requested_negatives == len(h)
    assert stats.n_retry_rounds == 1
    assert stats.n_retry_candidates_drawn == len(h)  # exactly one pass over the full batch, never inflated


def test_initial_collisions_is_exactly_zero_with_an_empty_positive_store():
    # No positive exists anywhere in this universe -- no draw can ever be
    # a collision, deterministically, regardless of seed.
    pools = TypePools(pools_by_type={"a": (0, 1, 2)}, type_of_index=("a", "a", "a"))
    store = build_positive_store([], n_relations=1, n_entities=3)
    sampler = BatchedTypeAwareNegativeSampler(pools, n_entities=3, n_relations=1, positive_store=store, seed=4)
    h = np.array([0, 1, 2] * 10)
    r = np.zeros_like(h)
    t = np.array([1, 2, 0] * 10)
    sampler.corrupt_batch(h, r, t)

    assert sampler.stats.n_initial_collisions == 0


def test_stats_counters_are_consistent_and_nonnegative():
    positives = [[0, 0, 3]]
    sampler = _sampler(positives, seed=5)
    h = np.array([0, 1, 2] * 20)
    r = np.zeros_like(h)
    t = np.array([3, 4, 3] * 20)
    sampler.corrupt_batch(h, r, t)

    stats = sampler.stats.to_dict()
    assert all(v >= 0 for v in stats.values())
    assert stats["n_requested_negatives"] == len(h)
    assert stats["n_retry_candidates_drawn"] >= stats["n_requested_negatives"]  # at least one full pass
    assert stats["n_retry_rounds"] >= 1
