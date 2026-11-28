"""The time-travel test's rules, as code. See ``ml/TIME_TRAVEL_PROTOCOL.md``, which fixed every choice here before any
data was collected. Pure functions: no network, no model, no file access.

A *hit* is a co-mention in a title/abstract or a registered trial. It is not evidence that a drug works."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

CUTOFF = "2022-04-25"
CUTOFF_YEAR, CUTOFF_MONTH, CUTOFF_DAY = 2022, 4, 25
SEED = 42
N_DISEASES = 50
TOP_K = 50
N_RESAMPLES = 10_000

_DATE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def sample_diseases(test_disease_ids: Sequence[str], n: int = N_DISEASES, seed: int = SEED) -> list[str]:
    """``n`` diseases drawn uniformly without replacement from the sorted IDs; returned sorted."""

    ids = sorted(test_disease_ids)
    if n > len(ids):
        raise ValueError(f"asked for {n} diseases but only {len(ids)} are available")
    chosen = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
    return sorted(ids[i] for i in chosen)


def random_drugs(drug_ids: Sequence[str], disease_id: str, k: int = TOP_K, seed: int = SEED) -> list[str]:
    """``k`` drugs uniformly without replacement, from ``default_rng([seed, int(disease_id)])``."""

    chosen = np.random.default_rng([seed, int(disease_id)]).choice(len(drug_ids), size=k, replace=False)
    return [drug_ids[i] for i in chosen]


def popularity_top(drug_degree: Mapping[str, int], k: int = TOP_K) -> list[str]:
    """The ``k`` highest-degree drugs, ties broken by drug ID."""

    return sorted(drug_degree, key=lambda drug_id: (-drug_degree[drug_id], drug_id))[:k]


def trial_is_prior(start_date: str | None) -> bool:
    """Whether a study's start date counts as on or before the cutoff.

    Missing or unparseable dates count as prior (unknown, so conservative). A date with month precision only is
    compared at month level, so April 2022 counts as prior; a year-only date compares by year, so 2022 counts as prior.
    """

    if not start_date:
        return True
    match = _DATE.match(start_date.strip())
    if not match:
        return True
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else None
    day = int(match.group(3)) if match.group(3) else None
    if year != CUTOFF_YEAR:
        return year < CUTOFF_YEAR
    if month is None:
        return True
    if month != CUTOFF_MONTH:
        return month < CUTOFF_MONTH
    return True if day is None else day <= CUTOFF_DAY


@dataclass(frozen=True)
class PairEvidence:
    """What was found for one drug-disease pair. ``error`` set means a query failed and the pair is excluded."""

    prior_papers: int = 0
    later_papers: int = 0
    prior_trials: int = 0
    later_trials: int = 0
    error: str | None = None


def classify(evidence: PairEvidence) -> str:
    """``error``, ``prior`` (had evidence by the cutoff), ``new_hit`` or ``new_miss``."""

    if evidence.error:
        return "error"
    if evidence.prior_papers > 0 or evidence.prior_trials > 0:
        return "prior"
    return "new_hit" if (evidence.later_papers > 0 or evidence.later_trials > 0) else "new_miss"


def counts(evidences: Sequence[PairEvidence]) -> dict[str, int]:
    """new pairs, hits (either), paper-only hits, trial-only hits, errors, prior pairs."""

    out = {"pairs": len(evidences), "prior": 0, "new": 0, "hits": 0, "paper_hits": 0, "trial_hits": 0, "errors": 0}
    for evidence in evidences:
        kind = classify(evidence)
        if kind == "error":
            out["errors"] += 1
        elif kind == "prior":
            out["prior"] += 1
        else:
            out["new"] += 1
            out["hits"] += kind == "new_hit"
            out["paper_hits"] += evidence.later_papers > 0
            out["trial_hits"] += evidence.later_trials > 0
    return out


def pooled_rate(per_disease: Sequence[Mapping[str, int]], key: str = "hits") -> float | None:
    """Sum of ``key`` over sum of new pairs, across diseases; ``None`` when there are no new pairs."""

    new = sum(d["new"] for d in per_disease)
    return sum(d[key] for d in per_disease) / new if new else None


def paired_bootstrap_difference(
    a: Sequence[Mapping[str, int]], b: Sequence[Mapping[str, int]], *, n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> dict:
    """Disease-level paired bootstrap of ``pooled_rate(a) - pooled_rate(b)``.

    ``a`` and ``b`` hold one counts dict per disease, in the same disease order. Each resample draws the diseases with
    replacement once and applies the same draw to both, so the rankers are compared on identical diseases. A resample in
    which either ranker has no new pairs is dropped and counted."""

    if len(a) != len(b):
        raise ValueError("both rankers need one counts dict per disease, in the same order")
    n = len(a)
    rng = np.random.default_rng(seed)
    hits_a, new_a = np.array([d["hits"] for d in a]), np.array([d["new"] for d in a])
    hits_b, new_b = np.array([d["hits"] for d in b]), np.array([d["new"] for d in b])
    differences, dropped = [], 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        denominator_a, denominator_b = new_a[idx].sum(), new_b[idx].sum()
        if denominator_a == 0 or denominator_b == 0:
            dropped += 1
            continue
        differences.append(hits_a[idx].sum() / denominator_a - hits_b[idx].sum() / denominator_b)
    if not differences:
        return {"difference": None, "ci_low": None, "ci_high": None, "excludes_zero": None, "resamples_used": 0, "resamples_dropped": dropped}
    low, high = np.percentile(differences, [2.5, 97.5])
    point_a, point_b = pooled_rate(a), pooled_rate(b)
    return {
        "difference": None if point_a is None or point_b is None else point_a - point_b,
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
        "resamples_used": len(differences),
        "resamples_dropped": dropped,
    }
