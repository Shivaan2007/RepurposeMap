"""Evaluation of the drug-ranking baseline.

``evaluate_disease_split`` is the proper disease-based evaluation (Milestone 3).
``evaluate_baseline`` is the older per-pair sanity check from Milestone 2. It is
kept for quick manual checks; the CLI's ``evaluate-baseline`` command now uses
the disease-based evaluation.
"""

from repurposemap.evaluation.baseline import (
    DEFAULT_EVAL_PAIRS,
    DEFAULT_EVAL_SEED,
    EvaluationResult,
    HeldOutPair,
    PairOutcome,
    evaluate_baseline,
    indication_pairs,
    rank_of,
    select_pairs,
)
from repurposemap.evaluation.disease_eval import (
    DiseaseEvaluationResult,
    TreatmentOutcome,
    evaluate_disease_split,
)
from repurposemap.evaluation.indications import IndicationRecord, extract_indications
from repurposemap.evaluation.leakage import (
    assert_no_leakage,
    find_disease_set_overlap,
    find_remaining_test_indications,
    find_remaining_test_pairs,
)
from repurposemap.evaluation.metrics import (
    filtered_rank,
    hits_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
)
from repurposemap.evaluation.random_baseline import random_ranking
from repurposemap.evaluation.split import (
    DEFAULT_EVAL_MAX_CANDIDATES,
    DEFAULT_EVAL_MAX_EDGE_CHECKS_PER_DRUG,
    DEFAULT_EVAL_TIME_LIMIT_PER_DRUG_S,
    DiseaseSplit,
    EvaluationConfig,
    split_by_disease,
    validate_split,
)
from repurposemap.evaluation.training_graph import build_training_graph

__all__ = [
    "DEFAULT_EVAL_MAX_CANDIDATES",
    "DEFAULT_EVAL_MAX_EDGE_CHECKS_PER_DRUG",
    "DEFAULT_EVAL_TIME_LIMIT_PER_DRUG_S",
    "DEFAULT_EVAL_PAIRS",
    "DEFAULT_EVAL_SEED",
    "DiseaseEvaluationResult",
    "DiseaseSplit",
    "EvaluationConfig",
    "EvaluationResult",
    "HeldOutPair",
    "IndicationRecord",
    "PairOutcome",
    "TreatmentOutcome",
    "assert_no_leakage",
    "build_training_graph",
    "evaluate_baseline",
    "evaluate_disease_split",
    "extract_indications",
    "filtered_rank",
    "find_disease_set_overlap",
    "find_remaining_test_indications",
    "find_remaining_test_pairs",
    "hits_at_k",
    "indication_pairs",
    "mean_reciprocal_rank",
    "random_ranking",
    "rank_of",
    "reciprocal_rank",
    "select_pairs",
    "split_by_disease",
    "validate_split",
]
