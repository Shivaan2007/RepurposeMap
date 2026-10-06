"""Small, reproducible check of the drug-ranking baseline on known indications.

For each held-out drug-disease indication:

1. Hide that drug-disease pair in both directions, so the direct edge cannot
   be used as evidence.
2. Rank every candidate drug for the disease.
3. Record the held-out drug's rank (1 is best), or None if it is not ranked.

The metrics are Hits@10 (the share of pairs whose drug ranks in the top 10) and
mean reciprocal rank (MRR). A pair the baseline does not rank scores 0 on both.

This is a sanity check, not a validated benchmark. The pairs are a small sample
chosen with a fixed seed. Ranking a known drug for a known disease does not
prove the baseline works, and the numbers must not be read as predictive
performance. A proper disease-based train/test split is the next step.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx

from repurposemap.ranking.drugs import DrugRanking, rank_drugs
from repurposemap.scoring.relation_policy import RelationPolicy, policy_view

DEFAULT_EVAL_PAIRS = 5
DEFAULT_EVAL_SEED = 0
HITS_K = 10

INDICATION_RELATION = "indication"


@dataclass(frozen=True)
class HeldOutPair:
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str


@dataclass(frozen=True)
class PairOutcome:
    pair: HeldOutPair
    rank: int | None
    n_ranked: int

    @property
    def hit_at_k(self) -> bool:
        return self.rank is not None and self.rank <= HITS_K

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.rank if self.rank is not None else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    seed: int
    outcomes: tuple[PairOutcome, ...]

    @property
    def n_pairs(self) -> int:
        return len(self.outcomes)

    @property
    def hits_at_10(self) -> float:
        return sum(outcome.hit_at_k for outcome in self.outcomes) / self.n_pairs

    @property
    def mean_reciprocal_rank(self) -> float:
        return sum(outcome.reciprocal_rank for outcome in self.outcomes) / self.n_pairs


def indication_pairs(graph: nx.MultiDiGraph) -> list[tuple[str, str]]:
    """Every (drug, disease) pair linked by an indication edge, sorted by ID."""
    pairs = set()
    for source, target, attrs in graph.edges(data=True):
        if attrs["relation"] != INDICATION_RELATION:
            continue
        if graph.nodes[source]["node_type"] == "drug" and graph.nodes[target]["node_type"] == "disease":
            pairs.add((source, target))
    return sorted(pairs)


def select_pairs(graph: nx.MultiDiGraph, n_pairs: int, seed: int) -> list[HeldOutPair]:
    """Choose ``n_pairs`` indication pairs with a fixed seed, so runs are repeatable."""
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be at least 1, got {n_pairs}")
    available = indication_pairs(graph)
    if not available:
        raise ValueError("the graph has no drug-indication-disease edges to evaluate")
    chosen = random.Random(seed).sample(available, min(n_pairs, len(available)))
    return [
        HeldOutPair(
            drug_id=drug,
            drug_name=graph.nodes[drug]["name"],
            disease_id=disease,
            disease_name=graph.nodes[disease]["name"],
        )
        for drug, disease in sorted(chosen)
    ]


def rank_of(drug_id: str, ranking: DrugRanking) -> int | None:
    """1-based position of ``drug_id`` in a ranking, or None if it is not ranked."""
    for position, candidate in enumerate(ranking.ranked, start=1):
        if candidate.drug_id == drug_id:
            return position
    return None


def evaluate_baseline(
    graph: nx.MultiDiGraph,
    pairs: Sequence[HeldOutPair],
    *,
    neighbour_counts: Mapping[str, int],
    policy: RelationPolicy | None = None,
    seed: int = DEFAULT_EVAL_SEED,
    **rank_options,
) -> EvaluationResult:
    """Rank candidates for each held-out pair and collect the metrics.

    ``rank_options`` are passed to ``rank_drugs``, such as ``max_path_length``.

    Raises:
        ValueError: a held-out direct edge is still visible after hiding, which
            would make the evaluation leak the answer.
    """
    policy = policy or RelationPolicy()
    outcomes = []
    for pair in pairs:
        hidden = ((pair.drug_id, pair.disease_id), (pair.disease_id, pair.drug_id))
        if policy_view(graph, policy, hidden).has_edge(pair.drug_id, pair.disease_id):
            raise ValueError(f"direct edge still visible for {pair.drug_name!r} and {pair.disease_name!r}")
        ranking = rank_drugs(
            graph,
            pair.disease_id,
            neighbour_counts=neighbour_counts,
            policy=policy,
            hidden_pairs=hidden,
            **rank_options,
        )
        outcomes.append(PairOutcome(pair=pair, rank=rank_of(pair.drug_id, ranking), n_ranked=len(ranking.ranked)))
    return EvaluationResult(seed=seed, outcomes=tuple(outcomes))
