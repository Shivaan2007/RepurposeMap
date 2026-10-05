from collections.abc import Callable
from pathlib import Path

import pytest

from repurposemap.graph import PathStep, find_paths, find_paths_detailed, format_path, load_graph
from repurposemap.graph import paths

ALPHA = "DEMO Drug Alpha"
ZETA = "DEMO Disease Zeta"
KINASE = "DEMO Protein Kinase 1"
PROTEIN_X = "DEMO Protein X"
PROTEIN_Y = "DEMO Protein Y"
ETA = "DEMO Disease Eta"
OMEGA = "DEMO Pathway Omega"
EPSILON = "DEMO Drug Epsilon"
IOTA = "DEMO Disease Iota"

DIRECT_STEP = PathStep(
    source=ALPHA,
    source_type="drug",
    relation="drug_indicated_for_disease",
    target=ZETA,
    target_type="disease",
)


@pytest.fixture
def graph(sample_csv: Path):
    return load_graph(sample_csv)


def test_connected_entities_return_direct_relation_path(graph):
    paths = find_paths(graph, ALPHA, ZETA)
    assert [DIRECT_STEP] in paths


def test_two_hop_path_keeps_each_relation_name_in_order(graph):
    paths = find_paths(graph, ALPHA, ZETA)
    two_hop = [
        PathStep(ALPHA, "drug", "drug_targets_protein", KINASE, "gene/protein"),
        PathStep(KINASE, "gene/protein", "protein_associated_with_disease", ZETA, "disease"),
    ]
    assert two_hop in paths


def test_all_sample_paths_from_alpha_to_zeta_are_found(graph):
    # Direct, two 2-hop routes via a protein, and one 3-hop route via Protein X.
    paths = find_paths(graph, ALPHA, ZETA)
    assert len(paths) == 4
    assert [DIRECT_STEP] in paths


def test_path_of_length_three_keeps_every_relation(graph):
    expected = [
        PathStep(ALPHA, "drug", "drug_targets_protein", KINASE, "gene/protein"),
        PathStep(KINASE, "gene/protein", "protein_interacts_with_protein", PROTEIN_X, "gene/protein"),
        PathStep(PROTEIN_X, "gene/protein", "protein_associated_with_disease", ZETA, "disease"),
    ]
    assert expected in find_paths(graph, ALPHA, ZETA, max_length=3)


def test_path_of_length_four_keeps_every_relation(graph):
    expected = [
        PathStep(ALPHA, "drug", "drug_targets_protein", KINASE, "gene/protein"),
        PathStep(KINASE, "gene/protein", "protein_interacts_with_protein", PROTEIN_X, "gene/protein"),
        PathStep(PROTEIN_X, "gene/protein", "protein_interacts_with_protein", PROTEIN_Y, "gene/protein"),
        PathStep(PROTEIN_Y, "gene/protein", "protein_associated_with_disease", ETA, "disease"),
    ]
    assert find_paths(graph, ALPHA, ETA) == [expected]


def test_lowering_max_length_excludes_the_longer_path(graph):
    assert find_paths(graph, ALPHA, ETA, max_length=3) == []
    assert len(find_paths(graph, ALPHA, ETA, max_length=4)) == 1


def test_lower_max_length_excludes_three_hop_routes(graph):
    paths = find_paths(graph, ALPHA, ZETA, max_length=2)
    assert paths
    assert all(len(path) <= 2 for path in paths)


def test_paths_are_returned_shortest_first(graph):
    paths = find_paths(graph, ALPHA, ZETA)
    lengths = [len(path) for path in paths]
    assert lengths == sorted(lengths)


def test_max_length_limits_path_hops(graph):
    paths = find_paths(graph, ALPHA, ZETA, max_length=1)
    assert paths == [[DIRECT_STEP]]


def test_max_paths_limits_number_of_returned_paths(graph):
    paths = find_paths(graph, ALPHA, ZETA, max_paths=1)
    assert paths == [[DIRECT_STEP]]


def test_disconnected_entities_return_no_path(graph):
    # Drug Epsilon and Disease Iota form their own component.
    assert find_paths(graph, EPSILON, ZETA) == []
    assert find_paths(graph, ALPHA, IOTA) == []


def test_direction_is_respected(graph):
    # Edges point from drug to disease, so the reverse direction has no path.
    assert find_paths(graph, ZETA, ALPHA) == []


def test_sample_pair_with_two_relations_yields_two_distinct_paths(graph):
    # Limit to one hop so that longer routes through Disease Zeta are excluded.
    paths = find_paths(graph, KINASE, OMEGA, max_length=1)
    relations = sorted(path[0].relation for path in paths)
    assert relations == ["protein_participates_in_pathway", "protein_regulates_pathway"]


def test_parallel_relations_between_same_pair_are_not_collapsed(
    write_csv: Callable[[str], Path],
):
    path = write_csv(
        "source_name,source_type,relation,target_name,target_type\n"
        "DEMO A,drug,rel_one,DEMO B,disease\n"
        "DEMO A,drug,rel_two,DEMO B,disease\n"
    )
    graph = load_graph(path)
    paths = find_paths(graph, "DEMO A", "DEMO B")
    relations = sorted(path[0].relation for path in paths)
    assert relations == ["rel_one", "rel_two"]


def test_missing_source_raises_clear_error(graph):
    with pytest.raises(ValueError, match="source entity not found"):
        find_paths(graph, "DEMO Drug Missing", ZETA)


def test_missing_target_raises_clear_error(graph):
    with pytest.raises(ValueError, match="target entity not found"):
        find_paths(graph, ALPHA, "DEMO Disease Missing")


def test_source_equal_to_target_raises_clear_error(graph):
    with pytest.raises(ValueError, match="must be different"):
        find_paths(graph, ALPHA, ALPHA)


@pytest.mark.parametrize("kwargs", [{"max_length": 0}, {"max_paths": 0}])
def test_non_positive_limits_raise_clear_error(graph, kwargs: dict[str, int]):
    with pytest.raises(ValueError, match="must be at least 1"):
        find_paths(graph, ALPHA, ZETA, **kwargs)


def test_format_path_shows_entities_and_relation_names(graph):
    text = format_path([DIRECT_STEP])
    assert text == f"{ALPHA}\n  --drug_indicated_for_disease-->\n{ZETA}"


def test_truncated_search_is_reported_not_silently_empty(graph):
    # A zero-length budget stops the search before any path is walked.
    result = find_paths_detailed(graph, ALPHA, ETA, max_edge_checks=1)
    assert result.paths == []
    assert result.truncated is True
    assert result.stop_reason == "edge check limit reached"


def test_completed_search_is_not_marked_truncated(graph):
    result = find_paths_detailed(graph, ALPHA, ZETA)
    assert result.truncated is False
    assert result.stop_reason is None


def test_search_stops_once_max_paths_are_found(graph):
    result = find_paths_detailed(graph, ALPHA, ZETA, max_paths=2)
    assert len(result.paths) == 2
    assert result.truncated is False


class _FakeClock:
    """Each reading advances the clock by 10 seconds, so any time limit is exceeded quickly."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += 10.0
        return self.now


def test_search_stops_when_time_limit_is_exceeded(graph, monkeypatch):
    monkeypatch.setattr(paths, "time", _FakeClock())
    monkeypatch.setattr(paths, "_CLOCK_CHECK_INTERVAL", 1)
    result = find_paths_detailed(graph, ALPHA, ETA, time_limit_s=1)
    assert result.truncated is True
    assert result.stop_reason == "time limit reached"


@pytest.mark.parametrize("kwargs", [{"time_limit_s": 0}, {"max_edge_checks": 0}])
def test_non_positive_budgets_raise_clear_error(graph, kwargs: dict[str, float]):
    with pytest.raises(ValueError, match="must be"):
        find_paths_detailed(graph, ALPHA, ZETA, **kwargs)


def test_search_result_is_deterministic(graph):
    assert find_paths(graph, ALPHA, ZETA) == find_paths(graph, ALPHA, ZETA)


def test_direct_edge_is_found_before_a_deep_search_would_use_the_budget(graph):
    # A small budget is enough to see the direct edge, but not to finish a
    # deeper backward pass. The direct path must still be returned.
    result = find_paths_detailed(graph, ALPHA, ZETA, max_edge_checks=10)
    assert [DIRECT_STEP] in result.paths
