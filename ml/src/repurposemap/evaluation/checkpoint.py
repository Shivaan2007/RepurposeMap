"""Resumable disease-based evaluation, for runs large enough that losing
progress to an interruption would be expensive.

``run_with_checkpoint`` calls the exact same ``prepare_evaluation`` and
``rank_one_disease`` that ``evaluate_disease_split`` uses. It adds nothing to
what gets ranked or how it is scored; it only saves each disease's result as
it finishes, and skips diseases already saved on a later call with the same
path. A checkpointed run's outcomes are identical to an uninterrupted
``evaluate_disease_split`` call with the same graph and config; only
completion time and recoverability differ.

The checkpoint file is JSON Lines: one line per disease, holding every
outcome for that disease together, so a single disease's record is written
and flushed as one unit. A disease with several true drugs cannot be left
half-recorded by an interruption; at worst, the disease in progress when the
interruption happened is simply absent and gets ranked again on resume.

This file does not check that a checkpoint matches the graph or config it is
being resumed with. Use one checkpoint path per evaluation run; reusing a
path across a different graph or config silently reuses stale results.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import networkx as nx

from repurposemap.evaluation.disease_eval import (
    DiseaseEvaluationResult,
    TreatmentOutcome,
    prepare_evaluation,
    rank_one_disease,
)
from repurposemap.evaluation.split import EvaluationConfig
from repurposemap.scoring.relation_policy import RelationPolicy


def _outcome_to_dict(outcome: TreatmentOutcome) -> dict:
    return {
        "disease_id": outcome.disease_id,
        "disease_name": outcome.disease_name,
        "drug_id": outcome.drug_id,
        "drug_name": outcome.drug_name,
        "path_rank": outcome.path_rank,
        "random_rank": outcome.random_rank,
        "n_candidates": outcome.n_candidates,
        "n_truncated_searches": outcome.n_truncated_searches,
        "candidate_cap_hit": outcome.candidate_cap_hit,
        "backward_search_truncated": outcome.backward_search_truncated,
    }


def read_checkpoint(path: Path) -> dict[str, list[TreatmentOutcome]]:
    """Every disease ID already recorded at ``path``, mapped to its outcomes.

    An empty dict means the file does not exist yet, which is not an error:
    it is the normal state of a fresh run's checkpoint path.
    """
    by_disease: dict[str, list[TreatmentOutcome]] = {}
    if not path.exists():
        return by_disease
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_disease[record["disease_id"]] = [TreatmentOutcome(**item) for item in record["outcomes"]]
    return by_disease


def run_with_checkpoint(
    graph: nx.MultiDiGraph,
    config: EvaluationConfig,
    checkpoint_path: str | Path,
    *,
    policy: RelationPolicy | None = None,
    on_disease_done: Callable[[str, list[TreatmentOutcome]], None] | None = None,
) -> DiseaseEvaluationResult:
    """Like ``evaluate_disease_split``, but resumable from ``checkpoint_path``.

    Every disease already recorded at ``checkpoint_path`` is read from there
    instead of being ranked again. Every other test disease is ranked, its
    outcomes are appended to the file as one flushed line, and
    ``on_disease_done(disease_id, outcomes)`` is called for it, if given, so a
    caller can print progress for a run too long to wait on silently.

    ``runtime_s`` on the result is only the time spent in *this* call: ranking
    skipped diseases does not count again, so it understates total time
    across several resumed calls. The checkpoint file itself has every
    disease's outcomes regardless of which call produced them.

    Raises:
        ValueError: same as ``evaluate_disease_split``.
    """
    policy = policy or RelationPolicy()
    started = time.monotonic()
    path = Path(checkpoint_path)
    context = prepare_evaluation(graph, config, policy)
    already_done = read_checkpoint(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    outcomes: list[TreatmentOutcome] = []
    with path.open("a", encoding="utf-8") as handle:
        for disease_id in context.split.test_disease_ids:
            records = context.by_disease.get(disease_id, [])
            if not records:
                continue
            if disease_id in already_done:
                outcomes.extend(already_done[disease_id])
                continue

            disease_outcomes = rank_one_disease(
                graph, disease_id, records, policy=policy, config=config, context=context
            )
            handle.write(
                json.dumps({"disease_id": disease_id, "outcomes": [_outcome_to_dict(o) for o in disease_outcomes]})
                + "\n"
            )
            handle.flush()
            outcomes.extend(disease_outcomes)
            if on_disease_done is not None:
                on_disease_done(disease_id, disease_outcomes)

    return DiseaseEvaluationResult(
        config=config, split=context.split, outcomes=tuple(outcomes), runtime_s=time.monotonic() - started
    )
