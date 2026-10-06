"""Evaluation harness for the drug-ranking baseline."""

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

__all__ = [
    "DEFAULT_EVAL_PAIRS",
    "DEFAULT_EVAL_SEED",
    "EvaluationResult",
    "HeldOutPair",
    "PairOutcome",
    "evaluate_baseline",
    "indication_pairs",
    "rank_of",
    "select_pairs",
]
