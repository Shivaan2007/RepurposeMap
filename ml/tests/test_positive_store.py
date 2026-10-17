"""Compact, exact positive-triple membership (Stage C, correction #A).

Compares against a plain Python ``set`` on small synthetic data -- the
``set`` is the trusted reference here, not the implementation under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from repurposemap.embeddings.positive_store import (
    PositiveTripleStore,
    build_positive_store,
    encode_triple,
    encode_triples_array,
)

# Real PrimeKG counts, from Stage B's smoke run.
REAL_N_ENTITIES = 129_375
REAL_N_RELATIONS = 33


def test_uint64_bound_holds_for_the_real_primekg_counts():
    """The exact bound from the module docstring, computed here, not assumed."""
    h_max = t_max = REAL_N_ENTITIES - 1
    r_max = REAL_N_RELATIONS - 1
    max_code = encode_triple(h_max, r_max, t_max, REAL_N_RELATIONS, REAL_N_ENTITIES)

    assert max_code == 552_350_390_624
    assert max_code < np.iinfo(np.uint64).max
    assert max_code < np.iinfo(np.int64).max  # comfortable margin, not just barely safe


def test_membership_matches_a_python_set_on_random_synthetic_data():
    rng = np.random.default_rng(0)
    n_entities, n_relations = 50, 6
    all_triples = rng.integers(0, n_entities, size=(400, 3))
    all_triples[:, 1] = rng.integers(0, n_relations, size=400)
    positive_rows = all_triples[:200]
    query_rows = all_triples[100:]  # overlaps with positives, plus some that are not

    reference = {tuple(int(x) for x in row) for row in positive_rows}
    store = build_positive_store(positive_rows, n_relations, n_entities)

    for row in query_rows:
        h, r, t = (int(x) for x in row)
        assert store.contains(h, r, t) == ((h, r, t) in reference)


def test_contains_many_matches_scalar_contains_for_every_row():
    rng = np.random.default_rng(1)
    n_entities, n_relations = 30, 4
    positives = rng.integers(0, n_entities, size=(150, 3))
    positives[:, 1] = rng.integers(0, n_relations, size=150)
    queries = rng.integers(0, n_entities, size=(80, 3))
    queries[:, 1] = rng.integers(0, n_relations, size=80)

    store = build_positive_store(positives, n_relations, n_entities)
    vectorized = store.contains_many(queries)
    scalar = np.array([store.contains(int(h), int(r), int(t)) for h, r, t in queries])

    assert np.array_equal(vectorized, scalar)


def test_first_last_and_missing_values():
    n_entities, n_relations = 10, 3
    triples = np.array([[0, 0, 0], [9, 2, 9], [5, 1, 5]])
    store = build_positive_store(triples, n_relations, n_entities)

    # First and last in sorted-code order must be found.
    assert store.contains(0, 0, 0)
    assert store.contains(9, 2, 9)
    assert store.contains(5, 1, 5)
    # A value definitely not present (every coordinate shifted).
    assert not store.contains(1, 0, 1)
    assert not store.contains(8, 2, 8)


def test_duplicate_input_rows_collapse_to_one_stored_entry():
    n_entities, n_relations = 10, 3
    triples = np.array([[1, 0, 2], [1, 0, 2], [1, 0, 2], [3, 1, 4]])
    store = build_positive_store(triples, n_relations, n_entities)

    assert len(store) == 2  # the triplicated row counts once
    assert store.contains(1, 0, 2)
    assert store.contains(3, 1, 4)


def test_memory_and_count_are_surfaced():
    n_entities, n_relations = 20, 5
    triples = np.array([[i, 0, i + 1] for i in range(10)])
    store = build_positive_store(triples, n_relations, n_entities)

    assert len(store) == 10
    assert store.nbytes == 10 * 8  # uint64 = 8 bytes each, no duplicates here


def test_empty_positive_set_contains_nothing():
    store = build_positive_store(np.empty((0, 3), dtype=np.int64), n_relations=5, n_entities=20)
    assert len(store) == 0
    assert store.nbytes == 0
    assert not store.contains(0, 0, 0)
    assert not any(store.contains_many(np.array([[0, 0, 0], [1, 1, 1]])))


def test_build_positive_store_accepts_a_small_non_array_iterable():
    # As used by small synthetic-graph tests: a plain set of int tuples.
    store = build_positive_store({(1, 0, 2), (3, 1, 4)}, n_relations=5, n_entities=10)
    assert len(store) == 2
    assert store.contains(1, 0, 2)
    assert not store.contains(2, 0, 1)


def test_build_positive_store_is_deterministic_regardless_of_row_order():
    n_entities, n_relations = 15, 4
    triples_a = np.array([[1, 0, 2], [5, 1, 6], [3, 2, 4]])
    triples_b = triples_a[::-1].copy()  # same rows, reversed order

    store_a = build_positive_store(triples_a, n_relations, n_entities)
    store_b = build_positive_store(triples_b, n_relations, n_entities)

    assert np.array_equal(store_a.codes, store_b.codes)


def test_encode_triples_array_matches_the_scalar_formula():
    n_entities, n_relations = 12, 4
    triples = np.array([[1, 2, 3], [11, 3, 11], [0, 0, 0]])
    vectorized = encode_triples_array(triples, n_relations, n_entities)
    scalar = [encode_triple(int(h), int(r), int(t), n_relations, n_entities) for h, r, t in triples]
    assert list(vectorized) == scalar


def test_positive_triple_store_is_a_frozen_dataclass_with_expected_fields():
    store = PositiveTripleStore(codes=np.array([1, 2, 3], dtype=np.uint64), n_relations=5, n_entities=10)
    with pytest.raises(Exception):
        store.n_relations = 6  # frozen -- must not be mutable
