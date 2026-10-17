"""Validation split: carved from training diseases only, never touching the locked test split."""

from __future__ import annotations

import pytest

from graph_builders import primekg_style_graph
from repurposemap.evaluation.indications import extract_indications
from repurposemap.evaluation.split import split_by_disease
from repurposemap.evaluation.training_graph import build_training_graph
from repurposemap.evaluation.transe_baseline import (
    VALIDATION_SEED_OFFSET,
    ValidationSplit,
    build_tuning_graph,
    split_validation_from_training,
    validate_validation_split,
)

INDICATION = ("indication", "indication")


def _graph_with_n_diseases(n: int):
    # One drug, one disease, one indication edge each -- n independent pairs.
    types = {}
    edges = []
    for i in range(n):
        drug, disease = f"D{i}", f"DIS{i}"
        types[drug] = "drug"
        types[disease] = "disease"
        edges.append((*INDICATION, drug, disease))
    return primekg_style_graph(types, edges)


def _split_for(n_diseases: int, *, seed: int = 42, max_test_diseases: int = 10):
    graph = _graph_with_n_diseases(n_diseases)
    indications = extract_indications(graph)
    split = split_by_disease(indications, seed=seed, test_fraction=0.1, max_test_diseases=max_test_diseases)
    return graph, split


def test_validation_diseases_come_only_from_the_training_portion():
    _graph, split = _split_for(100, max_test_diseases=10)
    validation_split = split_validation_from_training(split, val_fraction=0.1)

    train_set = set(split.train_disease_ids)
    assert set(validation_split.val_disease_ids) <= train_set
    assert set(validation_split.fit_disease_ids) <= train_set
    assert validation_split.test_disease_ids == split.test_disease_ids  # untouched, unchanged


def test_fit_val_test_are_pairwise_disjoint_and_union_reconstructs_train():
    _graph, split = _split_for(100, max_test_diseases=10)
    validation_split = split_validation_from_training(split, val_fraction=0.1)

    fit = set(validation_split.fit_disease_ids)
    val = set(validation_split.val_disease_ids)
    test = set(validation_split.test_disease_ids)

    assert fit & val == set()
    assert fit & test == set()
    assert val & test == set()
    assert fit | val == set(split.train_disease_ids)


def test_exact_counts_for_a_100_test_disease_1263_like_split():
    # Mirrors the real benchmark's scale closely enough to check rounding:
    # 1000 train diseases, 10% validation -> 100.
    _graph, split = _split_for(1100, max_test_diseases=100)
    assert len(split.train_disease_ids) == 1000
    assert len(split.test_disease_ids) == 100

    validation_split = split_validation_from_training(split, val_fraction=0.1)
    assert len(validation_split.val_disease_ids) == 100
    assert len(validation_split.fit_disease_ids) == 900


def test_validation_split_is_deterministic_for_a_fixed_seed():
    _graph, split = _split_for(100, max_test_diseases=10)
    a = split_validation_from_training(split, val_fraction=0.1)
    b = split_validation_from_training(split, val_fraction=0.1)
    assert a == b


def test_validation_sampling_uses_a_seed_derived_from_but_distinct_from_the_primary_split():
    _graph, split = _split_for(100, max_test_diseases=10)
    other_split = split_by_disease(
        extract_indications(_graph_with_n_diseases(100)), seed=split.seed + VALIDATION_SEED_OFFSET, test_fraction=0.1, max_test_diseases=10
    )
    validation_split = split_validation_from_training(split, val_fraction=0.1)
    # The validation sample should not simply equal "whatever seed+OFFSET
    # would pick as a disease-level test split" -- it is a distinct sampling
    # procedure (sampling from train_disease_ids, not from all diseases).
    assert set(validation_split.val_disease_ids) != set(other_split.test_disease_ids)


def test_validate_validation_split_raises_on_fit_val_overlap():
    _graph, split = _split_for(100, max_test_diseases=10)
    bad = ValidationSplit(
        fit_disease_ids=split.train_disease_ids,  # deliberately includes everything
        val_disease_ids=(split.train_disease_ids[0],),  # overlaps with fit
        test_disease_ids=split.test_disease_ids,
    )
    with pytest.raises(ValueError, match="fit and validation"):
        validate_validation_split(bad, split)


def test_validate_validation_split_raises_on_val_test_overlap():
    _graph, split = _split_for(100, max_test_diseases=10)
    bad = ValidationSplit(
        fit_disease_ids=split.train_disease_ids[1:],
        val_disease_ids=(split.test_disease_ids[0],),  # a locked test disease, wrongly reused
        test_disease_ids=split.test_disease_ids,
    )
    with pytest.raises(ValueError, match="validation and test"):
        validate_validation_split(bad, split)


def test_validate_validation_split_raises_when_union_does_not_reconstruct_train():
    _graph, split = _split_for(100, max_test_diseases=10)
    bad = ValidationSplit(
        fit_disease_ids=split.train_disease_ids[:-1],  # drops one train disease entirely
        val_disease_ids=(),
        test_disease_ids=split.test_disease_ids,
    )
    with pytest.raises(ValueError, match="does not reconstruct"):
        validate_validation_split(bad, split)


def test_tuning_graph_removes_indications_for_both_validation_and_test_diseases():
    graph, split = _split_for(100, max_test_diseases=10)
    validation_split = split_validation_from_training(split, val_fraction=0.1)

    tuning_graph = build_tuning_graph(graph, validation_split)

    removed_ids = set(validation_split.val_disease_ids) | set(validation_split.test_disease_ids)
    for disease_id in removed_ids:
        drug_id = disease_id.replace("DIS", "D")
        assert not tuning_graph.has_edge(drug_id, disease_id)
        assert not tuning_graph.has_edge(disease_id, drug_id)


def test_tuning_graph_keeps_fit_diseases_indications():
    graph, split = _split_for(100, max_test_diseases=10)
    validation_split = split_validation_from_training(split, val_fraction=0.1)
    tuning_graph = build_tuning_graph(graph, validation_split)

    fit_disease = validation_split.fit_disease_ids[0]
    fit_drug = fit_disease.replace("DIS", "D")
    assert tuning_graph.has_edge(fit_drug, fit_disease)


def test_tuning_graph_is_at_least_as_strict_as_the_final_training_graph():
    # The tuning graph removes MORE (val + test) than the final training
    # graph (test only) -- so every edge missing from the final training
    # graph is also missing from the tuning graph, but not vice versa.
    graph, split = _split_for(100, max_test_diseases=10)
    validation_split = split_validation_from_training(split, val_fraction=0.1)

    tuning_graph = build_tuning_graph(graph, validation_split)
    final_training_graph = build_training_graph(graph, split.test_disease_ids)

    assert tuning_graph.number_of_edges() <= final_training_graph.number_of_edges()
