"""Ranking metrics, decoupled from any one model.

These functions take a plain ordered list of candidate IDs and the ID being
scored. They do not know about paths, drugs or diseases, so the same functions
can score the path baseline today and a TransE or R-GCN ranking later, without
change.

Filtered rank. When a disease has several true drugs, a baseline that puts two
of them first and second should not be punished for "pushing down" the other
true drug. Before finding one true drug's rank, every *other* true drug for the
same disease is removed from the candidate list. This is the standard practice
in knowledge-graph link prediction, called a filtered rank. Each true drug is
scored this way, so a disease with three true drugs contributes three ranks to
the metrics below, one per drug, each computed with the other two filtered out.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def filtered_rank(true_id: str, other_true_ids: Iterable[str], ranked_ids: Sequence[str]) -> int | None:
    """1-based position of ``true_id`` in ``ranked_ids``, after removing ``other_true_ids``.

    Returns None if ``true_id`` is not in ``ranked_ids`` at all.
    """
    exclude = set(other_true_ids) - {true_id}
    filtered = [candidate_id for candidate_id in ranked_ids if candidate_id not in exclude]
    try:
        return filtered.index(true_id) + 1
    except ValueError:
        return None


def hits_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of ``ranks`` that are not None and at most ``k``. 0.0 for an empty sequence."""
    if not ranks:
        return 0.0
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def reciprocal_rank(rank: int | None) -> float:
    """``1/rank``, or 0.0 if ``rank`` is None."""
    return 1 / rank if rank is not None else 0.0


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank over ``ranks``. 0.0 for an empty sequence."""
    if not ranks:
        return 0.0
    return sum(reciprocal_rank(rank) for rank in ranks) / len(ranks)


def average_precision(rank: int | None, n_relevant: int) -> float:
    """Placeholder for a future AUPRC metric. Not used or validated yet.

    Raises:
        NotImplementedError: always. Precision-recall metrics need the full
            score distribution, not just a rank, and are planned for a later
            milestone alongside a trained model that can produce one.
    """
    raise NotImplementedError("AUPRC is planned for a future milestone, once a scored model exists")
