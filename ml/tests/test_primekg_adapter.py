"""Tests for the PrimeKG adapter, run on a tiny synthetic fixture.

The fixture uses PrimeKG's real 12-column schema and node-type and relation
vocabulary, but every name and identifier in it is invented (the "DEMO" names).
No real PrimeKG rows are included, so the tests need neither the full 936 MB
file nor any redistributed dataset content.
"""

from collections.abc import Callable
from pathlib import Path

import networkx as nx
import pytest

from repurposemap.adapters import load_primekg
from repurposemap.graph import find_paths, find_paths_detailed, format_path, search_entities

PRIMEKG_HEADER = (
    "relation,display_relation,x_index,x_id,x_type,x_name,x_source,"
    "y_index,y_id,y_type,y_name,y_source"
)
DRUG_MU = "109"
DIS_TAU = "110"
DRUG_KAPPA = "103"
DIS_LAMBDA = "104"
PROT_NU = "101"
PROT_XI = "102"
COMPOUND_PI_DRUG = "105"
PROT_RHO = "107"
COMPOUND_PI_EXPOSURE = "106"
DIS_SIGMA = "108"


@pytest.fixture
def primekg_fixture() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "primekg_mini.csv"


@pytest.fixture
def graph(primekg_fixture: Path) -> nx.MultiDiGraph:
    return load_primekg(primekg_fixture)


def _row(**fields: str) -> str:
    """Build one PrimeKG row. Missing fields default to a valid, unrelated value."""
    defaults = {
        "relation": "protein_protein",
        "display_relation": "ppi",
        "x_index": "1",
        "x_id": "100",
        "x_type": "gene/protein",
        "x_name": "GENE ONE",
        "x_source": "NCBI",
        "y_index": "2",
        "y_id": "200",
        "y_type": "gene/protein",
        "y_name": "GENE TWO",
        "y_source": "NCBI",
    }
    defaults.update(fields)
    order = PRIMEKG_HEADER.split(",")
    return ",".join(defaults[column] for column in order)


def _write(write_csv: Callable[[str], Path], *rows: str) -> Path:
    return write_csv(PRIMEKG_HEADER + "\n" + "\n".join(rows) + "\n")


# Loading and node identity


def test_fixture_loads_as_multidigraph(graph):
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 12
    assert graph.number_of_edges() == 15


def test_nodes_are_keyed_by_primekg_index_not_by_name(graph):
    assert DRUG_MU in graph
    assert "DEMO Drug Mu" not in graph


def test_node_attributes_keep_name_type_and_source_identifiers(graph):
    attrs = graph.nodes[DRUG_MU]
    assert attrs["name"] == "DEMO Drug Mu"
    assert attrs["node_type"] == "drug"
    assert attrs["external_id"] == "DEMO-DRUG-MU"


def test_external_id_and_provenance_come_from_the_file(graph):
    attrs = graph.nodes[DIS_TAU]
    assert attrs["external_id"] == "DEMO-DIS-TAU"
    assert attrs["provenance"] == "MONDO"
    assert graph.nodes[PROT_NU]["provenance"] == "NCBI"


def test_names_are_trimmed_of_outer_spaces(write_csv):
    path = _write(
        write_csv,
        _row(x_index="1", x_name="Padded name ", y_index="2", y_name="Other"),
    )
    graph = load_primekg(path)
    assert graph.nodes["1"]["name"] == "Padded name"


def test_name_shared_by_two_nodes_keeps_both(graph):
    # "DEMO Compound Pi" is a drug (index 105) and an exposure (index 106).
    assert graph.nodes[COMPOUND_PI_DRUG]["name"] == "DEMO Compound Pi"
    assert graph.nodes[COMPOUND_PI_EXPOSURE]["name"] == "DEMO Compound Pi"
    assert graph.nodes[COMPOUND_PI_DRUG]["node_type"] == "drug"
    assert graph.nodes[COMPOUND_PI_EXPOSURE]["node_type"] == "exposure"


def test_same_name_with_different_types_is_not_an_error(write_csv):
    path = _write(
        write_csv,
        _row(x_index="1", x_name="Shared", x_type="drug", y_index="2", y_name="Other", y_type="disease",
             relation="drug_indication", display_relation="indication"),
        _row(x_index="3", x_name="Shared", x_type="pathway", y_index="2", y_name="Other", y_type="disease",
             relation="disease_involves_pathway", display_relation="linked"),
    )
    graph = load_primekg(path)
    assert graph.number_of_nodes() == 3


def test_same_index_with_conflicting_attributes_is_an_error(write_csv):
    path = _write(
        write_csv,
        _row(x_index="1", x_name="First"),
        _row(x_index="1", x_name="Second", y_index="3"),
    )
    with pytest.raises(ValueError, match="node index 1 has conflicting attributes"):
        load_primekg(path)


# Edges and relations


def test_edge_keeps_relation_and_display_relation(graph):
    edge = graph[DRUG_KAPPA][PROT_NU]
    assert len(edge) == 1
    attrs = next(iter(edge.values()))
    assert attrs["relation"] == "drug_protein"
    assert attrs["display_relation"] == "target"


def test_parallel_relations_between_one_pair_are_kept_separate(graph):
    edges = graph[COMPOUND_PI_DRUG][PROT_RHO]
    labels = sorted(attrs["display_relation"] for attrs in edges.values())
    assert labels == ["carrier", "target"]
    assert graph.number_of_edges(COMPOUND_PI_DRUG, PROT_RHO) == 2


def test_relation_direction_is_preserved_without_adding_reverse_edges(graph):
    # The file stores this exposure edge in one direction only.
    assert graph.has_edge(COMPOUND_PI_EXPOSURE, DIS_SIGMA)
    assert not graph.has_edge(DIS_SIGMA, COMPOUND_PI_EXPOSURE)


def test_both_directions_already_in_file_are_both_present(graph):
    assert graph.has_edge(DRUG_MU, DIS_TAU)
    assert graph.has_edge(DIS_TAU, DRUG_MU)


def test_exact_duplicate_rows_are_collapsed(write_csv):
    line = _row(x_index="1", y_index="2")
    path = _write(write_csv, line, line)
    assert load_primekg(path).number_of_edges() == 1


def test_same_pair_with_different_display_is_not_collapsed(write_csv):
    path = _write(
        write_csv,
        _row(relation="drug_protein", display_relation="target", x_index="1", y_index="2",
             x_type="drug", y_type="gene/protein"),
        _row(relation="drug_protein", display_relation="enzyme", x_index="1", y_index="2",
             x_type="drug", y_type="gene/protein"),
    )
    assert load_primekg(path).number_of_edges() == 2


# Validation


def test_missing_file_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="PrimeKG CSV not found"):
        load_primekg(tmp_path / "kg.csv")


def test_missing_primekg_column_is_reported(write_csv):
    path = write_csv("relation,x_index,y_index\nppi,1,2\n")
    with pytest.raises(ValueError, match="missing PrimeKG column"):
        load_primekg(path)


def test_empty_value_is_reported_with_line_number(write_csv):
    path = _write(write_csv, _row(x_index="1", y_index="2"), _row(x_index="3", x_name="", y_index="4"))
    with pytest.raises(ValueError, match="empty values.*line 3"):
        load_primekg(path)


def test_non_integer_index_is_reported(write_csv):
    path = _write(write_csv, _row(x_index="abc"))
    with pytest.raises(ValueError, match="node index is not an integer: 'abc'"):
        load_primekg(path)


def test_header_only_file_has_no_edges(write_csv):
    with pytest.raises(ValueError, match="no edges"):
        load_primekg(write_csv(PRIMEKG_HEADER + "\n"))


# Search, paths and formatting on the PrimeKG-schema graph


def test_search_by_name_returns_id_name_and_type(graph):
    results = search_entities(graph, "DEMO Drug Eta")
    assert len(results) == 1
    match = results[0]
    assert match.id == "111"
    assert match.name == "DEMO Drug Eta"
    assert match.node_type == "drug"
    assert match.external_id == "DEMO-DRUG-ETA"


def test_search_by_name_is_case_insensitive(graph):
    assert search_entities(graph, "DEMO DRUG MU")[0].id == DRUG_MU


def test_path_between_two_hops_uses_ids_and_shows_names(graph):
    paths = find_paths(graph, DRUG_KAPPA, PROT_XI, max_length=2)
    assert len(paths) == 1
    first, second = paths[0]
    assert (first.source, first.target) == (DRUG_KAPPA, PROT_NU)
    assert (second.source, second.target) == (PROT_NU, PROT_XI)
    assert format_path(paths[0]) == (
        "DEMO Drug Kappa\n"
        "  --drug_protein [target]-->\n"
        "DEMO Protein Nu\n"
        "  --protein_protein [ppi]-->\n"
        "DEMO Protein Xi"
    )


def test_path_respects_max_length(graph):
    assert find_paths(graph, DRUG_KAPPA, PROT_XI, max_length=1) == []
    assert len(find_paths(graph, DRUG_KAPPA, PROT_XI, max_length=2)) == 1


def test_path_respects_max_results(graph):
    assert len(find_paths(graph, COMPOUND_PI_DRUG, PROT_RHO, max_paths=1)) == 1


def test_parallel_relations_give_separate_paths(graph):
    paths = find_paths(graph, COMPOUND_PI_DRUG, PROT_RHO, max_length=1)
    labels = sorted(path[0].display_relation for path in paths)
    assert labels == ["carrier", "target"]


def test_no_path_is_empty_and_not_truncated(graph):
    result = find_paths_detailed(graph, DRUG_MU, PROT_XI)
    assert result.paths == []
    assert result.truncated is False
