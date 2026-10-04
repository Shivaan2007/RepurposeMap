from collections.abc import Callable
from pathlib import Path

import networkx as nx
import pytest

from repurposemap.graph import REQUIRED_COLUMNS, load_graph

HEADER = "source_name,source_type,relation,target_name,target_type"
KINASE = "DEMO Protein Kinase 1"
OMEGA = "DEMO Pathway Omega"


def test_sample_graph_loads_as_multidigraph(sample_csv: Path):
    graph = load_graph(sample_csv)
    assert isinstance(graph, nx.MultiDiGraph)


def test_sample_graph_has_expected_node_and_edge_counts(sample_csv: Path):
    graph = load_graph(sample_csv)
    assert graph.number_of_nodes() == 15
    assert graph.number_of_edges() == 23


def test_sample_graph_has_a_disconnected_component(sample_csv: Path):
    graph = load_graph(sample_csv)
    assert nx.number_weakly_connected_components(graph) == 2
    assert set(graph.neighbors("DEMO Drug Epsilon")) == {"DEMO Disease Iota"}


def test_sample_pair_is_connected_by_two_relations(sample_csv: Path):
    graph = load_graph(sample_csv)
    relations = {data["relation"] for data in graph[KINASE][OMEGA].values()}
    assert relations == {"protein_participates_in_pathway", "protein_regulates_pathway"}


def test_node_types_are_preserved(sample_csv: Path):
    graph = load_graph(sample_csv)
    assert graph.nodes["DEMO Drug Alpha"]["node_type"] == "drug"
    assert graph.nodes["DEMO Protein Kinase 1"]["node_type"] == "gene/protein"
    assert graph.nodes["DEMO Disease Zeta"]["node_type"] == "disease"
    assert graph.nodes["DEMO Pathway Omega"]["node_type"] == "pathway"


def test_relation_types_are_preserved_on_edges(sample_csv: Path):
    graph = load_graph(sample_csv)
    relations = {data["relation"] for _, _, data in graph.edges(data=True)}
    assert relations == {
        "drug_targets_protein",
        "drug_indicated_for_disease",
        "protein_associated_with_disease",
        "protein_participates_in_pathway",
        "protein_regulates_pathway",
        "protein_interacts_with_protein",
        "disease_involves_pathway",
    }


def test_same_pair_can_carry_multiple_relations(write_csv: Callable[[str], Path]):
    path = write_csv(
        f"{HEADER}\n"
        "DEMO A,drug,rel_one,DEMO B,disease\n"
        "DEMO A,drug,rel_two,DEMO B,disease\n"
    )
    graph = load_graph(path)
    assert graph.number_of_edges("DEMO A", "DEMO B") == 2


def test_exact_duplicate_rows_are_collapsed(write_csv: Callable[[str], Path]):
    path = write_csv(
        f"{HEADER}\n"
        "DEMO A,drug,rel_one,DEMO B,disease\n"
        "DEMO A,drug,rel_one,DEMO B,disease\n"
    )
    graph = load_graph(path)
    assert graph.number_of_edges() == 1


def test_missing_required_column_raises_clear_error(write_csv: Callable[[str], Path]):
    path = write_csv("source_name,source_type,relation,target_name\nDEMO A,drug,rel,DEMO B\n")
    with pytest.raises(ValueError, match="missing required column.*target_type"):
        load_graph(path)


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_each_required_column_is_checked(column: str, write_csv: Callable[[str], Path]):
    columns = [c for c in REQUIRED_COLUMNS if c != column]
    path = write_csv(",".join(columns) + "\n" + ",".join(["x"] * len(columns)) + "\n")
    with pytest.raises(ValueError, match=f"missing required column.*{column}"):
        load_graph(path)


def test_empty_required_value_raises_clear_error(write_csv: Callable[[str], Path]):
    path = write_csv(
        f"{HEADER}\n"
        "DEMO A,drug,rel_one,DEMO B,disease\n"
        "DEMO A,drug,,DEMO C,disease\n"
    )
    with pytest.raises(ValueError, match="empty required values.*line 3"):
        load_graph(path)


def test_whitespace_only_value_counts_as_missing(write_csv: Callable[[str], Path]):
    path = write_csv(f"{HEADER}\nDEMO A,drug,rel_one,   ,disease\n")
    with pytest.raises(ValueError, match="empty required values"):
        load_graph(path)


def test_literal_na_string_is_kept_as_a_name(write_csv: Callable[[str], Path]):
    path = write_csv(f"{HEADER}\nNA,drug,rel_one,DEMO B,disease\n")
    graph = load_graph(path)
    assert "NA" in graph.nodes


def test_header_only_file_raises_no_edges_error(write_csv: Callable[[str], Path]):
    path = write_csv(f"{HEADER}\n")
    with pytest.raises(ValueError, match="no edges"):
        load_graph(path)


def test_empty_file_raises_clear_error(write_csv: Callable[[str], Path]):
    path = write_csv("")
    with pytest.raises(ValueError, match="expected a header row"):
        load_graph(path)


def test_name_with_two_types_raises_clear_error(write_csv: Callable[[str], Path]):
    path = write_csv(
        f"{HEADER}\n"
        "DEMO A,drug,rel_one,DEMO B,disease\n"
        "DEMO B,pathway,rel_two,DEMO C,disease\n"
    )
    with pytest.raises(ValueError, match="more than one type.*DEMO B"):
        load_graph(path)


def test_nonexistent_file_raises_clear_error(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Graph CSV not found"):
        load_graph(missing)
