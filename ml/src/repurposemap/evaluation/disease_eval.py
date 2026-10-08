"""Disease-based evaluation of the path-quality drug-ranking baseline.

This is the proper evaluation Milestone 2's per-pair sanity check was a
placeholder for. It is built to be reusable: everything here operates on a
disease ID, a ranked list of drug IDs, and a rank number. Swapping in a future
TransE or R-GCN model means writing a function that returns a ranked drug-ID
list for a disease; the split, the leakage checks and the metrics do not change.

Pipeline for one evaluation run:

1. Extract every known indication (repurposemap.evaluation.indications).
2. Split by disease ID (repurposemap.evaluation.split).
3. Build one training graph with every test disease's indication edges removed
   (repurposemap.evaluation.training_graph).
4. Check for leakage (repurposemap.evaluation.leakage). A violation raises.
5. For each test disease, rank candidate drugs with the path baseline, and also
   compute a random-order baseline over the same candidates.
6. For each held-out (drug, disease) pair, compute its filtered rank in both
   rankings (repurposemap.evaluation.metrics).

Research use only. This measures whether the baseline can recover a known
treatment after its direct evidence is hidden. It does not evaluate, and must
not be read as evaluating, whether any ranked drug would actually work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx

from repurposemap.evaluation.indications import IndicationRecord, extract_indications
from repurposemap.evaluation.leakage import assert_no_leakage
from repurposemap.evaluation.metrics import hits_at_k, mean_reciprocal_rank, filtered_rank
from repurposemap.evaluation.random_baseline import random_ranking
from repurposemap.evaluation.split import DiseaseSplit, EvaluationConfig, split_by_disease
from repurposemap.evaluation.training_graph import build_training_graph, test_disease_indication_filter
from repurposemap.ranking.drugs import rank_drugs
from repurposemap.scoring.hubs import NeighbourCounts
from repurposemap.scoring.relation_policy import RelationPolicy


@dataclass(frozen=True)
class TreatmentOutcome:
    """One held-out (drug, disease) pair's result in both rankings.

    ``n_candidates``, ``n_truncated_searches``, ``candidate_cap_hit`` and
    ``backward_search_truncated`` describe the one ranking run for this pair's
    disease (from ``DrugRanking``), so they repeat across every true drug of
    the same disease. They do not change what was ranked; they say how
    complete that ranking's search was, so a disease with heavy truncation can
    be told apart from one that searched its candidates fully.
    """

    disease_id: str
    disease_name: str
    drug_id: str
    drug_name: str
    path_rank: int | None
    random_rank: int | None
    n_candidates: int
    n_truncated_searches: int
    candidate_cap_hit: bool
    backward_search_truncated: bool


@dataclass(frozen=True)
class DiseaseEvaluationResult:
    """Everything a report needs: the configuration, the split, and every outcome."""

    config: EvaluationConfig
    split: DiseaseSplit
    outcomes: tuple[TreatmentOutcome, ...]
    runtime_s: float

    @property
    def n_train_diseases(self) -> int:
        return len(self.split.train_disease_ids)

    @property
    def n_test_diseases(self) -> int:
        return len(self.split.test_disease_ids)

    @property
    def n_train_indications(self) -> int:
        return len(self.split.train_indications)

    @property
    def n_test_indications(self) -> int:
        return len(self.split.test_indications)

    def hits_at(self, k: int) -> float:
        return hits_at_k([o.path_rank for o in self.outcomes], k)

    def random_hits_at(self, k: int) -> float:
        return hits_at_k([o.random_rank for o in self.outcomes], k)

    @property
    def mrr(self) -> float:
        return mean_reciprocal_rank([o.path_rank for o in self.outcomes])

    @property
    def random_mrr(self) -> float:
        return mean_reciprocal_rank([o.random_rank for o in self.outcomes])

    def _one_outcome_per_disease(self) -> list[TreatmentOutcome]:
        """One outcome per test disease, since the search-completeness fields repeat."""
        seen: dict[str, TreatmentOutcome] = {}
        for outcome in self.outcomes:
            seen.setdefault(outcome.disease_id, outcome)
        return list(seen.values())

    @property
    def n_diseases_with_candidate_cap_hit(self) -> int:
        """Diseases where more candidates existed than ``max_candidates`` allowed."""
        return sum(o.candidate_cap_hit for o in self._one_outcome_per_disease())

    @property
    def n_diseases_with_truncated_searches(self) -> int:
        """Diseases where at least one candidate's own per-drug search hit its budget."""
        return sum(o.n_truncated_searches > 0 for o in self._one_outcome_per_disease())

    @property
    def n_diseases_with_backward_search_truncated(self) -> int:
        """Diseases where the candidate-finding search itself hit its edge-check budget."""
        return sum(o.backward_search_truncated for o in self._one_outcome_per_disease())


def evaluate_disease_split(
    graph: nx.MultiDiGraph,
    config: EvaluationConfig,
    *,
    policy: RelationPolicy | None = None,
) -> DiseaseEvaluationResult:
    """Run the full disease-based evaluation pipeline and return its result.

    Raises:
        ValueError: the graph has no indication edges, the split is invalid, or
            a leakage check fails.
    """
    policy = policy or RelationPolicy()
    started = time.monotonic()

    indications = extract_indications(graph)
    if not indications:
        raise ValueError("the graph has no drug-indication-disease edges to evaluate")
    split = split_by_disease(
        indications, seed=config.seed, test_fraction=config.test_fraction, max_test_diseases=config.max_test_diseases
    )
    # Built once, purely to verify there is no leakage before ranking starts.
    # The per-disease ranking below does not use this view; see the module
    # docstring and repurposemap.evaluation.training_graph for why.
    training_graph = build_training_graph(graph, split.test_disease_ids)
    assert_no_leakage(training_graph, split)

    # The same exclusion, as a plain predicate instead of a second nested view,
    # folded into rank_drugs' own single filter layer via extra_edge_filter.
    exclude_test_indications = test_disease_indication_filter(graph, split.test_disease_ids)
    neighbour_counts = NeighbourCounts(graph, policy, extra_filter=exclude_test_indications)
    by_disease: dict[str, list[IndicationRecord]] = {}
    for record in split.test_indications:
        by_disease.setdefault(record.disease_id, []).append(record)

    outcomes: list[TreatmentOutcome] = []
    for disease_id in split.test_disease_ids:
        records = by_disease.get(disease_id, [])
        if not records:
            continue
        true_ids = tuple(record.drug_id for record in records)
        ranking = rank_drugs(
            graph,
            disease_id,
            neighbour_counts=neighbour_counts,
            policy=policy,
            max_path_length=config.max_path_length,
            paths_per_drug=config.paths_per_drug,
            max_candidates=config.max_candidates,
            time_limit_per_drug_s=config.time_limit_per_drug_s,
            max_edge_checks_per_drug=config.max_edge_checks_per_drug,
            max_backward_edge_checks=config.max_backward_edge_checks,
            extra_edge_filter=exclude_test_indications,
        )
        ranked_ids = [candidate.drug_id for candidate in ranking.ranked]
        random_ids = random_ranking(ranked_ids, seed=config.seed, disease_id=disease_id)

        for record in records:
            other_true = [drug_id for drug_id in true_ids if drug_id != record.drug_id]
            outcomes.append(
                TreatmentOutcome(
                    disease_id=disease_id,
                    disease_name=record.disease_name,
                    drug_id=record.drug_id,
                    drug_name=record.drug_name,
                    path_rank=filtered_rank(record.drug_id, other_true, ranked_ids),
                    random_rank=filtered_rank(record.drug_id, other_true, random_ids),
                    n_candidates=len(ranked_ids),
                    n_truncated_searches=ranking.n_truncated_searches,
                    candidate_cap_hit=ranking.candidate_cap_hit,
                    backward_search_truncated=ranking.backward_search_truncated,
                )
            )

    return DiseaseEvaluationResult(
        config=config, split=split, outcomes=tuple(outcomes), runtime_s=time.monotonic() - started
    )
