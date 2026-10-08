"""Deterministic disease-based train/test split for evaluation.

Edge-level splits (hiding one drug-disease pair at a time) are only safe for a
baseline that cannot learn from the graph. A model that fits on the graph, such
as TransE or an R-GCN, needs an explicit disease-level split: every treatment
edge of a held-out disease must be absent from the single training graph used
to fit it, not hidden one pair at a time while it trains on everything else.
Splitting by edge risks leaking a disease's other drugs into training; splitting
by disease does not.

The split is on **disease ID**, not on edge. Every indication edge of a test
disease goes to the test set; none of it goes to training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from repurposemap.evaluation.indications import IndicationRecord
from repurposemap.ranking.drugs import DEFAULT_MAX_BACKWARD_EDGE_CHECKS

# Candidates searched per disease during evaluation, and the per-drug search
# budget. These match rank-drugs' own defaults: a controlled, instrumented
# benchmark (see ml/README.md) found that an earlier, much tighter set of
# defaults here was a reaction to a measurement artifact (concurrent diagnostic
# processes competing for memory on an 8 GB machine), not a real cost of the
# search itself. Keeping these generous avoids truncating valid work for no
# real benefit. They are configurable per call if a run needs to be faster.
DEFAULT_EVAL_MAX_CANDIDATES = 500
DEFAULT_EVAL_MAX_EDGE_CHECKS_PER_DRUG = 1_000_000
DEFAULT_EVAL_TIME_LIMIT_PER_DRUG_S = 5.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Every knob the disease-based evaluation uses. No hidden defaults.

    ``top_ks`` are the Hits@K values to report. ``max_test_diseases`` caps the
    number of test diseases regardless of ``test_fraction``, so a run on the
    full graph stays fast. Set it to ``None`` to use ``test_fraction`` alone.
    """

    seed: int = 0
    test_fraction: float = 0.1
    max_test_diseases: int | None = 20
    top_ks: tuple[int, ...] = (1, 3, 10)
    max_path_length: int = 3
    paths_per_drug: int = 50
    max_candidates: int = DEFAULT_EVAL_MAX_CANDIDATES
    time_limit_per_drug_s: float = DEFAULT_EVAL_TIME_LIMIT_PER_DRUG_S
    max_edge_checks_per_drug: int = DEFAULT_EVAL_MAX_EDGE_CHECKS_PER_DRUG
    max_backward_edge_checks: int = DEFAULT_MAX_BACKWARD_EDGE_CHECKS

    def __post_init__(self) -> None:
        if not 0 < self.test_fraction <= 1:
            raise ValueError(f"test_fraction must be in (0, 1], got {self.test_fraction}")
        if self.max_test_diseases is not None and self.max_test_diseases < 1:
            raise ValueError(f"max_test_diseases must be at least 1, got {self.max_test_diseases}")
        if not self.top_ks or any(k < 1 for k in self.top_ks):
            raise ValueError(f"top_ks must be non-empty and positive, got {self.top_ks}")
        if self.max_path_length < 1 or self.paths_per_drug < 1 or self.max_candidates < 1:
            raise ValueError("max_path_length, paths_per_drug and max_candidates must be at least 1")
        if self.max_edge_checks_per_drug < 1 or self.max_backward_edge_checks < 1:
            raise ValueError("max_edge_checks_per_drug and max_backward_edge_checks must be at least 1")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "test_fraction": self.test_fraction,
            "max_test_diseases": self.max_test_diseases,
            "top_ks": list(self.top_ks),
            "max_path_length": self.max_path_length,
            "paths_per_drug": self.paths_per_drug,
            "max_candidates": self.max_candidates,
            "time_limit_per_drug_s": self.time_limit_per_drug_s,
            "max_edge_checks_per_drug": self.max_edge_checks_per_drug,
            "max_backward_edge_checks": self.max_backward_edge_checks,
        }


@dataclass(frozen=True)
class DiseaseSplit:
    """A disease-level train/test partition. Disjoint by construction and validated."""

    train_disease_ids: tuple[str, ...]
    test_disease_ids: tuple[str, ...]
    train_indications: tuple[IndicationRecord, ...]
    test_indications: tuple[IndicationRecord, ...]
    seed: int = field(default=0)


def split_by_disease(
    records: tuple[IndicationRecord, ...],
    *,
    seed: int,
    test_fraction: float,
    max_test_diseases: int | None = None,
) -> DiseaseSplit:
    """Split indication records by disease ID, deterministically for a given seed.

    The number of test diseases is ``round(test_fraction * n_diseases)``, capped
    by ``max_test_diseases`` if given, and at least 1. Every indication edge of a
    test disease goes to the test set. No edge of a train disease is held out.

    Raises:
        ValueError: ``records`` is empty, or the resulting split fails validation.
    """
    diseases = sorted({record.disease_id for record in records})
    if not diseases:
        raise ValueError("no indication records to split")

    n_test = max(1, round(test_fraction * len(diseases)))
    if max_test_diseases is not None:
        n_test = min(n_test, max_test_diseases)
    n_test = min(n_test, len(diseases))

    test_ids = tuple(sorted(random.Random(seed).sample(diseases, n_test)))
    test_set = set(test_ids)
    train_ids = tuple(sorted(d for d in diseases if d not in test_set))

    split = DiseaseSplit(
        train_disease_ids=train_ids,
        test_disease_ids=test_ids,
        train_indications=tuple(sorted(r for r in records if r.disease_id not in test_set)),
        test_indications=tuple(sorted(r for r in records if r.disease_id in test_set)),
        seed=seed,
    )
    validate_split(split)
    return split


def validate_split(split: DiseaseSplit) -> None:
    """Check the split's internal consistency. Raises on the first problem found.

    Checks: train and test disease sets are disjoint, every train indication
    belongs to a train disease, every test indication belongs to a test disease,
    and there is at least one test disease.
    """
    train_set = set(split.train_disease_ids)
    test_set = set(split.test_disease_ids)

    if not test_set:
        raise ValueError("split has no test diseases")
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"train and test disease sets overlap: {sorted(overlap)[:5]}")
    for record in split.train_indications:
        if record.disease_id not in train_set:
            raise ValueError(f"train indication for {record.disease_id!r} is not a train disease")
    for record in split.test_indications:
        if record.disease_id not in test_set:
            raise ValueError(f"test indication for {record.disease_id!r} is not a test disease")
