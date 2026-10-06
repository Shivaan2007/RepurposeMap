"""Non-ML baseline that ranks drug candidates for one disease by path quality.

This is a research hypothesis ranking. It is not a treatment recommendation, and
it is not a trained model. A high score means that the graph contains several
short, well-scored paths from the drug to the disease. It does not mean that the
drug treats the disease.

Formula. For a drug ``d`` and a disease ``D``:

    1. Candidate paths: directed paths ``d -> ... -> D`` of at most
       ``max_path_length`` hops, found by bounded search over edges the
       relation policy allows. At most ``paths_per_drug`` shortest paths are kept.
    2. Score each path with repurposemap.scoring.score_path. Invalid paths drop out.
    3. Let ``s_1 >= s_2 >= ...`` be the valid path totals. Then

           path_sum(d)   = sum of s_i for i = 1 .. min(K, n)
           drug_score(d) = path_sum(d) - promiscuity_penalty(d)

       where K = SCORED_PATHS_PER_DRUG (3) and n is the number of valid paths.
       The promiscuity penalty depends on the number of distinct protein
       ``target`` edges the drug has (see repurposemap.scoring.promiscuity). It
       stops drugs with very many targets from winning on target count alone.
    4. Drugs with no valid path are not ranked. Ranking is by drug_score, then by
       the number of valid paths, then by drug name and ID, so the order is
       deterministic.

Candidates are the drug nodes within ``max_path_length`` hops of the disease
along allowed edges. A backward search finds them, so the search never enumerates
every path in the graph. If there are more candidates than ``max_candidates``,
the nearest ones (fewest hops) are kept, ties by ID, and the cut is reported.

Direct drug-disease edges are hidden from the search through ``hidden_pairs``,
so a known indication is never counted as its own evidence.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import networkx as nx

from repurposemap.graph.paths import find_paths_detailed
from repurposemap.scoring.path_quality import RankedPath, rank_paths
from repurposemap.scoring.promiscuity import DrugTargetCounts, drug_promiscuity_penalty
from repurposemap.scoring.relation_policy import RelationPolicy, policy_view

DEFAULT_RANK_PATH_LENGTH = 3
DEFAULT_PATHS_PER_DRUG = 50
SCORED_PATHS_PER_DRUG = 3
DEFAULT_MAX_CANDIDATES = 2000
DEFAULT_TIME_LIMIT_PER_DRUG_S = 5.0
DEFAULT_MAX_EDGE_CHECKS_PER_DRUG = 1_000_000


@dataclass(frozen=True)
class DrugCandidate:
    """One ranked drug. ``best_path`` is the explanation shown to the user."""

    drug_id: str
    drug_name: str
    score: float
    path_sum: float
    promiscuity_penalty: float
    n_targets: int
    n_qualifying_paths: int
    best_path: RankedPath
    search_truncated: bool


@dataclass(frozen=True)
class DrugRanking:
    """Ranked drugs for one disease, plus counts that show how complete the search was."""

    disease_id: str
    disease_name: str
    ranked: tuple[DrugCandidate, ...]
    n_candidates: int
    candidate_cap_hit: bool
    n_without_path: int
    n_truncated_searches: int


def rank_drugs(
    graph: nx.MultiDiGraph,
    disease_id: str,
    *,
    neighbour_counts: Mapping[str, int],
    policy: RelationPolicy | None = None,
    max_path_length: int = DEFAULT_RANK_PATH_LENGTH,
    paths_per_drug: int = DEFAULT_PATHS_PER_DRUG,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    time_limit_per_drug_s: float = DEFAULT_TIME_LIMIT_PER_DRUG_S,
    max_edge_checks_per_drug: int = DEFAULT_MAX_EDGE_CHECKS_PER_DRUG,
    hidden_pairs: Iterable[tuple[str, str]] = (),
) -> DrugRanking:
    """Rank the drugs that reach ``disease_id`` by quality-scored short paths.

    ``neighbour_counts`` comes from ``allowed_neighbour_counts`` for the same graph
    and policy. ``hidden_pairs`` are (source, target) node IDs whose edges are
    hidden, typically both directions of a held-out indication.

    Raises:
        ValueError: the disease is not in the graph or is not a disease node, or a
            limit is below 1.
    """
    policy = policy or RelationPolicy()
    for name, value in (
        ("max_path_length", max_path_length),
        ("paths_per_drug", paths_per_drug),
        ("max_candidates", max_candidates),
        ("max_edge_checks_per_drug", max_edge_checks_per_drug),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if time_limit_per_drug_s <= 0:
        raise ValueError(f"time_limit_per_drug_s must be positive, got {time_limit_per_drug_s}")
    if disease_id not in graph:
        raise ValueError(f"disease entity not found: {disease_id!r}")
    if graph.nodes[disease_id]["node_type"] != "disease":
        raise ValueError(f"entity {disease_id!r} is not a disease node")

    view = policy_view(graph, policy, hidden_pairs)
    reach = _backward_distances(view, disease_id, max_path_length)
    # Nearest drugs first, so that when the cap applies it keeps the closest candidates.
    drug_ids = sorted(
        (node for node in reach if reach[node] >= 1 and _is_drug(graph, node)),
        key=lambda node: (reach[node], node),
    )
    cap_hit = len(drug_ids) > max_candidates
    drug_ids = drug_ids[:max_candidates]

    target_counts = DrugTargetCounts(graph)
    ranked: list[DrugCandidate] = []
    without_path = 0
    truncated = 0
    for drug_id in drug_ids:
        result = find_paths_detailed(
            view,
            drug_id,
            disease_id,
            max_length=max_path_length,
            max_paths=paths_per_drug,
            time_limit_s=time_limit_per_drug_s,
            max_edge_checks=max_edge_checks_per_drug,
        )
        if result.truncated:
            truncated += 1
        paths = rank_paths(result.paths, policy, neighbour_counts)
        if not paths:
            without_path += 1
            continue
        scored = paths[:SCORED_PATHS_PER_DRUG]
        path_sum = sum(path.score.total_score for path in scored)
        n_targets = target_counts[drug_id]
        penalty = drug_promiscuity_penalty(n_targets)
        ranked.append(
            DrugCandidate(
                drug_id=drug_id,
                drug_name=graph.nodes[drug_id]["name"],
                score=path_sum - penalty,
                path_sum=path_sum,
                promiscuity_penalty=penalty,
                n_targets=n_targets,
                n_qualifying_paths=len(paths),
                best_path=paths[0],
                search_truncated=result.truncated,
            )
        )

    ranked.sort(key=lambda c: (-c.score, -c.n_qualifying_paths, c.drug_name.casefold(), c.drug_id))
    return DrugRanking(
        disease_id=disease_id,
        disease_name=graph.nodes[disease_id]["name"],
        ranked=tuple(ranked),
        n_candidates=len(drug_ids),
        candidate_cap_hit=cap_hit,
        n_without_path=without_path,
        n_truncated_searches=truncated,
    )


def _is_drug(graph: nx.MultiDiGraph, node: str) -> bool:
    return graph.nodes[node]["node_type"] == "drug"


def _backward_distances(view: nx.MultiDiGraph, target: str, depth: int) -> dict[str, int]:
    """Hop distance from every node to ``target`` along allowed edges, up to ``depth``."""
    distance = {target: 0}
    queue: deque[str] = deque([target])
    while queue:
        node = queue.popleft()
        if distance[node] == depth:
            continue
        for predecessor in view.pred[node]:
            if predecessor not in distance:
                distance[predecessor] = distance[node] + 1
                queue.append(predecessor)
    return distance
