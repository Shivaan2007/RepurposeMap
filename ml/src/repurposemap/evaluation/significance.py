"""Disease-level paired bootstrap significance test.

Implements, exactly, the method frozen (before any R-GCN locked-test number
existed) in ``docs/FINAL_RESULTS.md`` section 3: resample the locked-test
*diseases* with replacement, recompute MRR and Hits@10 for both rankers on
each resample, and report the 95% percentile interval of the (model A minus
model B) difference. This module does not choose or adjust the method -- it
only executes what was already specified.

Takes the per-disease outcome JSON that ``repurposemap rgcn-evaluate`` /
``repurposemap transe-evaluate``'s ``--per-disease-output`` writes: a list of
``{"disease_id": ..., "drug_id": ..., "rank": ...}`` records, one per held-out
(drug, disease) pair. The unit of resampling is the *disease*, not the pair,
per the frozen method's own stated reason (the held-out split is built
disease-by-disease, so some diseases contribute several pairs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from repurposemap.evaluation.metrics import hits_at_k, mean_reciprocal_rank


@dataclass(frozen=True)
class PairedBootstrapResult:
    """One metric's paired-bootstrap result for (model A minus model B)."""

    metric: str
    point_estimate_a: float
    point_estimate_b: float
    point_estimate_diff: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    n_resamples: int
    n_diseases: int


def _outcomes_by_disease(outcomes: Sequence[dict]) -> dict[str, list[int | None]]:
    by_disease: dict[str, list[int | None]] = {}
    for outcome in outcomes:
        by_disease.setdefault(outcome["disease_id"], []).append(outcome["rank"])
    return by_disease


def _metric_fn(metric: str):
    if metric == "mrr":
        return mean_reciprocal_rank
    if metric == "hits_at_10":
        return lambda ranks: hits_at_k(ranks, 10)
    raise ValueError(f"unsupported metric {metric!r}; expected 'mrr' or 'hits_at_10'")


def paired_bootstrap_ci(
    outcomes_a: Sequence[dict],
    outcomes_b: Sequence[dict],
    *,
    metric: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Paired, disease-level bootstrap CI for (model A minus model B) on ``metric``.

    ``outcomes_a`` and ``outcomes_b`` must cover exactly the same set of
    disease IDs -- both models were evaluated on the same locked-test split,
    so a mismatch almost certainly means the wrong artifact was passed, and
    this refuses to silently compare different disease sets.
    """
    by_disease_a = _outcomes_by_disease(outcomes_a)
    by_disease_b = _outcomes_by_disease(outcomes_b)
    disease_ids_a = set(by_disease_a)
    disease_ids_b = set(by_disease_b)
    if disease_ids_a != disease_ids_b:
        raise ValueError(
            "outcomes_a and outcomes_b cover different disease sets -- "
            f"only in A: {sorted(disease_ids_a - disease_ids_b)[:5]}, "
            f"only in B: {sorted(disease_ids_b - disease_ids_a)[:5]}"
        )
    disease_ids = sorted(disease_ids_a)
    if not disease_ids:
        raise ValueError("no diseases to resample")

    metric_fn = _metric_fn(metric)
    point_a = metric_fn([rank for ranks in by_disease_a.values() for rank in ranks])
    point_b = metric_fn([rank for ranks in by_disease_b.values() for rank in ranks])

    rng = np.random.default_rng(seed)
    n = len(disease_ids)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(disease_ids, size=n, replace=True)
        ranks_a = [rank for disease_id in sample for rank in by_disease_a[disease_id]]
        ranks_b = [rank for disease_id in sample for rank in by_disease_b[disease_id]]
        diffs[i] = metric_fn(ranks_a) - metric_fn(ranks_b)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return PairedBootstrapResult(
        metric=metric,
        point_estimate_a=point_a,
        point_estimate_b=point_b,
        point_estimate_diff=point_a - point_b,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        excludes_zero=bool(ci_low > 0 or ci_high < 0),
        n_resamples=n_resamples,
        n_diseases=n,
    )
