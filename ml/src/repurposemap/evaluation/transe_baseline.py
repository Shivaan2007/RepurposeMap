"""TransE evaluation glue: validation split, tuning graph, full-universe ranking.

Kept separate from ``disease_eval``'s ``TreatmentOutcome``/
``DiseaseEvaluationResult`` and from ``checkpoint``'s JSONL path-evaluation
checkpoint, per the approved plan: TransE is a separately trained model,
evaluated by a separate full-universe scoring step with no bounded search,
no candidate-cap/truncation bookkeeping, and its own extra outputs (macro/
micro AUPRC) that don't fit the bounded-search outcome shape. This module
reuses, rather than reimplements, everything that already exists:
``DiseaseSplit``/``split_by_disease`` for the disease-level split,
``build_training_graph`` for the leakage-safe view, and ``filtered_rank``/
``hits_at_k``/``mean_reciprocal_rank``/``average_precision``/``macro_auprc``/
``micro_auprc`` for every metric.

This module knows nothing about how a TransE model is trained -- it only
knows how to score one, using the ``repurposemap.embeddings`` package's
``TransE`` model and indices, on diseases the caller gives it (validation
diseases, in this milestone; never the locked 100 test diseases until a
later, explicitly approved stage).
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from repurposemap.evaluation.indications import IndicationRecord
from repurposemap.evaluation.metrics import (
    average_precision,
    filtered_rank,
    hits_at_k,
    macro_auprc,
    mean_reciprocal_rank,
    micro_auprc,
)
from repurposemap.evaluation.split import DiseaseSplit
from repurposemap.evaluation.training_graph import build_training_graph

# Distinct from the primary split's own seed, so the validation sample is
# independent of it rather than correlated by construction.
VALIDATION_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class ValidationSplit:
    """Validation diseases carved out of the *training* portion only.

    ``fit_disease_ids`` / ``val_disease_ids`` partition the original
    ``DiseaseSplit.train_disease_ids`` exactly (every train disease is in
    exactly one of the two). ``test_disease_ids`` is copied from the
    original split unchanged, for convenience when building the tuning
    graph (C) -- it is never touched by this split itself.
    """

    fit_disease_ids: tuple[str, ...]
    val_disease_ids: tuple[str, ...]
    test_disease_ids: tuple[str, ...]


def split_validation_from_training(split: DiseaseSplit, *, val_fraction: float = 0.1) -> ValidationSplit:
    """Carve validation diseases out of ``split.train_disease_ids`` only.

    Deterministic: ``random.Random(split.seed + VALIDATION_SEED_OFFSET)``, a
    local RNG instance distinct from the one the primary split used (never
    global ``random`` state), matching ``split_by_disease``'s own
    discipline. Never samples from, or otherwise touches,
    ``split.test_disease_ids`` -- the locked 100 test diseases are passed
    through unchanged.

    Raises:
        ValueError: the resulting fit/val/test disease sets are not
            pairwise disjoint, or fit | val does not reconstruct
            ``split.train_disease_ids`` exactly. Neither should be possible
            given the construction below; checked anyway; see
            ``tests/test_validation_split.py``.
    """
    train_ids = split.train_disease_ids
    n_val = max(1, round(val_fraction * len(train_ids)))
    n_val = min(n_val, len(train_ids))

    val_ids = tuple(sorted(random.Random(split.seed + VALIDATION_SEED_OFFSET).sample(train_ids, n_val)))
    val_set = set(val_ids)
    fit_ids = tuple(sorted(d for d in train_ids if d not in val_set))

    result = ValidationSplit(fit_disease_ids=fit_ids, val_disease_ids=val_ids, test_disease_ids=split.test_disease_ids)
    validate_validation_split(result, split)
    return result


def validate_validation_split(validation_split: ValidationSplit, split: DiseaseSplit) -> None:
    """Raise if fit/val/test are not pairwise disjoint, or fit|val != train_disease_ids.

    Separate, explicit function (not folded silently into the constructor
    above) so a test can call it directly against a hand-built
    ``ValidationSplit`` to prove it actually catches each violation.
    """
    fit = set(validation_split.fit_disease_ids)
    val = set(validation_split.val_disease_ids)
    test = set(validation_split.test_disease_ids)

    if fit & val:
        raise ValueError(f"fit and validation disease sets overlap: {sorted(fit & val)}")
    if fit & test:
        raise ValueError(f"fit and test disease sets overlap: {sorted(fit & test)}")
    if val & test:
        raise ValueError(f"validation and test disease sets overlap: {sorted(val & test)}")

    reconstructed = fit | val
    expected = set(split.train_disease_ids)
    if reconstructed != expected:
        raise ValueError(
            "fit | val does not reconstruct the original train_disease_ids "
            f"(missing: {sorted(expected - reconstructed)}, extra: {sorted(reconstructed - expected)})"
        )


def build_tuning_graph(graph: nx.MultiDiGraph, validation_split: ValidationSplit) -> nx.MultiDiGraph:
    """The graph used while choosing hyperparameters: validation AND test diseases' indications removed.

    Reuses ``build_training_graph`` unchanged -- no second leakage filter.
    Removing the union of ``val_disease_ids`` and ``test_disease_ids`` means
    a hyperparameter choice made against this graph's validation
    performance never benefited from the model having seen validation
    diseases' real indications, any more than it benefited from seeing the
    locked test diseases'.
    """
    removed_ids = set(validation_split.val_disease_ids) | set(validation_split.test_disease_ids)
    return build_training_graph(graph, removed_ids)


def indications_for_diseases(
    indications: Sequence[IndicationRecord], disease_ids: Iterable[str]
) -> tuple[IndicationRecord, ...]:
    """The subset of ``indications`` whose disease is in ``disease_ids``, sorted.

    A disease's indications always live entirely within ``split.train_indications``
    or entirely within ``split.test_indications`` (never split between the
    two), so for a validation disease (drawn from the training portion),
    this is called on ``split.train_indications``, filtered down to just
    ``val_disease_ids``.
    """
    disease_set = set(disease_ids)
    return tuple(sorted(record for record in indications if record.disease_id in disease_set))


@dataclass(frozen=True)
class TransEOutcome:
    """One held-out (drug, disease) pair's TransE result.

    ``transe_rank`` is the *filtered* rank (other true drugs for the same
    disease excluded first, per ``filtered_rank``) among the full drug
    universe -- never a bounded candidate set.
    """

    disease_id: str
    disease_name: str
    drug_id: str
    drug_name: str
    transe_rank: int | None


@dataclass(frozen=True)
class TransEEvaluationResult:
    """TransE's result on a set of diseases (validation, in this milestone).

    Deliberately not ``DiseaseEvaluationResult``/``TreatmentOutcome`` --
    see the module docstring. ``macro_auprc``/``micro_auprc`` are computed
    once, over every evaluated disease's full-universe score/label vector
    (not per-outcome, since AUPRC is a disease-level, not a per-pair,
    quantity -- see ``evaluate_transe_on_diseases``).
    """

    disease_ids: tuple[str, ...]
    outcomes: tuple[TransEOutcome, ...]
    macro_auprc: float
    micro_auprc: float
    runtime_s: float

    def hits_at(self, k: int) -> float:
        return hits_at_k([outcome.transe_rank for outcome in self.outcomes], k)

    @property
    def mrr(self) -> float:
        return mean_reciprocal_rank([outcome.transe_rank for outcome in self.outcomes])


def evaluate_transe_on_diseases(
    score_fn,
    disease_ids: Sequence[str],
    indications: Sequence[IndicationRecord],
    drug_ids: Sequence[str],
    *,
    disease_names: dict[str, str] | None = None,
    drug_names: dict[str, str] | None = None,
) -> TransEEvaluationResult:
    """Score every disease in ``disease_ids`` against the full ``drug_ids`` universe.

    ``score_fn(disease_id, drug_ids) -> np.ndarray`` returns one continuous
    TransE score per drug in ``drug_ids`` (same order), for the given
    disease -- the caller supplies this (it is the only place a real
    ``TransE`` model, entity index, and resolved indication-relation index
    are needed; this function stays model-agnostic, matching
    ``repurposemap.evaluation.metrics``'s own "decoupled from any one
    model" design).

    For each disease: candidates are ranked by score, descending, and
    ``filtered_rank`` gives each of the disease's true drugs its rank
    (other true drugs for the same disease excluded first) -- for Hits@K/
    MRR. For AUPRC, the *unfiltered* full score/label vector is used
    instead (co-positives are real positives, not noise) -- per the
    approved plan, this is deliberately a different convention from
    Hits@K/MRR's, not an oversight.
    """
    started = time.monotonic()
    disease_names = disease_names or {}
    drug_names = drug_names or {}
    indications_by_disease: dict[str, list[IndicationRecord]] = {}
    for record in indications:
        indications_by_disease.setdefault(record.disease_id, []).append(record)

    outcomes: list[TransEOutcome] = []
    per_disease_auprc: list[float] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for disease_id in disease_ids:
        true_drug_ids = [record.drug_id for record in indications_by_disease.get(disease_id, [])]
        if not true_drug_ids:
            continue

        scores = np.asarray(score_fn(disease_id, drug_ids))
        if scores.shape != (len(drug_ids),):
            raise ValueError(f"score_fn must return one score per drug ({len(drug_ids)}), got shape {scores.shape}")

        order = np.argsort(-scores, kind="stable")
        ranked_ids = [drug_ids[i] for i in order]

        for drug_id in true_drug_ids:
            other_true = [d for d in true_drug_ids if d != drug_id]
            rank = filtered_rank(drug_id, other_true, ranked_ids)
            outcomes.append(
                TransEOutcome(
                    disease_id=disease_id,
                    disease_name=disease_names.get(disease_id, disease_id),
                    drug_id=drug_id,
                    drug_name=drug_names.get(drug_id, drug_id),
                    transe_rank=rank,
                )
            )

        labels = np.array([1 if drug_id in true_drug_ids else 0 for drug_id in drug_ids], dtype=float)
        per_disease_auprc.append(average_precision(scores, labels))
        all_scores.append(scores)
        all_labels.append(labels)

    return TransEEvaluationResult(
        disease_ids=tuple(disease_ids),
        outcomes=tuple(outcomes),
        macro_auprc=macro_auprc(per_disease_auprc),
        micro_auprc=micro_auprc(all_scores, all_labels),
        runtime_s=time.monotonic() - started,
    )
