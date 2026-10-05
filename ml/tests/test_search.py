from pathlib import Path

import pytest

from repurposemap.graph import EntityMatch, load_graph, search_entities


@pytest.fixture
def graph(sample_csv: Path):
    return load_graph(sample_csv)


def test_exact_name_returns_that_entity_with_its_type(graph):
    results = search_entities(graph, "DEMO Drug Alpha")
    assert results == [EntityMatch(id="DEMO Drug Alpha", name="DEMO Drug Alpha", node_type="drug")]


def test_search_ignores_case(graph):
    assert search_entities(graph, "demo drug alpha") == search_entities(graph, "DEMO DRUG ALPHA")
    assert search_entities(graph, "demo drug alpha") == [
        EntityMatch(id="DEMO Drug Alpha", name="DEMO Drug Alpha", node_type="drug")
    ]


def test_partial_name_matches_all_containing_entities(graph):
    results = search_entities(graph, "protein", limit=10)
    assert [match.name for match in results] == [
        "DEMO Protein Enzyme 3",
        "DEMO Protein Kinase 1",
        "DEMO Protein Receptor 2",
        "DEMO Protein X",
        "DEMO Protein Y",
    ]
    assert {match.node_type for match in results} == {"gene/protein"}


def test_search_with_no_match_returns_empty_list(graph):
    assert search_entities(graph, "no such entity anywhere") == []


def test_limit_caps_the_number_of_results(graph):
    results = search_entities(graph, "DEMO", limit=3)
    assert len(results) == 3


def test_default_limit_applies_when_not_given(graph):
    assert len(search_entities(graph, "DEMO")) == 10


def test_exact_match_ranks_first(graph):
    results = search_entities(graph, "DEMO Drug Beta", limit=5)
    assert results[0].name == "DEMO Drug Beta"


def test_results_are_deterministic(graph):
    assert search_entities(graph, "DEMO") == search_entities(graph, "DEMO")


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_raises_clear_error(graph, query: str):
    with pytest.raises(ValueError, match="must not be empty"):
        search_entities(graph, query)


def test_limit_below_one_raises_clear_error(graph):
    with pytest.raises(ValueError, match="limit must be at least 1"):
        search_entities(graph, "DEMO", limit=0)
