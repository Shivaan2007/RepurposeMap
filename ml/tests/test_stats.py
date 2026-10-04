from pathlib import Path

from repurposemap.graph import (
    GraphStats,
    compute_stats,
    count_edges_by_relation,
    count_nodes_by_type,
    load_graph,
)


def test_compute_stats_returns_structured_totals(sample_csv: Path):
    stats = compute_stats(load_graph(sample_csv))
    assert isinstance(stats, GraphStats)
    assert stats.n_nodes == 15
    assert stats.n_edges == 23


def test_node_type_counts_are_correct(sample_csv: Path):
    counts = count_nodes_by_type(load_graph(sample_csv))
    assert counts == {
        "disease": 4,
        "drug": 4,
        "gene/protein": 5,
        "pathway": 2,
    }


def test_relation_counts_are_correct(sample_csv: Path):
    counts = count_edges_by_relation(load_graph(sample_csv))
    assert counts == {
        "disease_involves_pathway": 2,
        "drug_indicated_for_disease": 4,
        "drug_targets_protein": 5,
        "protein_associated_with_disease": 6,
        "protein_interacts_with_protein": 2,
        "protein_participates_in_pathway": 3,
        "protein_regulates_pathway": 1,
    }


def test_type_and_relation_counts_sum_to_totals(sample_csv: Path):
    stats = compute_stats(load_graph(sample_csv))
    assert sum(stats.node_type_counts.values()) == stats.n_nodes
    assert sum(stats.relation_counts.values()) == stats.n_edges


def test_count_dictionaries_are_sorted_by_key(sample_csv: Path):
    stats = compute_stats(load_graph(sample_csv))
    assert list(stats.node_type_counts) == sorted(stats.node_type_counts)
    assert list(stats.relation_counts) == sorted(stats.relation_counts)
