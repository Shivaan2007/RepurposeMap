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

import numpy as np


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


def average_precision(scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray) -> float:
    """Disease-level **Average Precision (AP)** over one disease's full candidate universe.

    This is NOT the trapezoidal area under the precision-recall curve
    (PR-AUC) -- an earlier version of this function computed that, under
    the same name, which is a real and common mix-up; this is the
    corrected, standard (non-interpolated) Average Precision used in
    information retrieval and multi-label ranking, equivalent to
    scikit-learn's ``average_precision_score`` but computed here with only
    NumPy (no sklearn dependency):

        AP = (1 / n_positives) * sum(precision_at_rank_k for every k where label_k == 1)

    i.e. sort candidates by score descending, compute precision at every
    rank, then average precision **only at the ranks that are themselves
    positives** -- not integrated/interpolated between them.

    ``scores`` and ``labels`` are parallel arrays over the *complete*
    candidate universe (not a rank, and not a top-K subset): ``labels[i]``
    is 1 if candidate ``i`` is a true positive, 0 otherwise, and
    ``scores[i]`` is that candidate's continuous model score (higher is
    more plausible). Unlike ``filtered_rank``'s Hits@K/MRR convention,
    co-positive candidates (other true drugs for the same disease) all stay
    in the label vector together -- never filtered out.

    Tie handling (deterministic, but worth knowing about): candidates with
    exactly equal scores are ordered by a *stable* sort, so ties keep their
    original input order rather than being broken by label or randomized.
    This makes AP fully deterministic for a given ``(scores, labels)``
    pair, but means a tie's contribution depends on which order the tied
    rows happened to arrive in -- see
    ``test_average_precision_tie_handling_is_deterministic_but_order_dependent``
    in ``tests/test_auprc.py`` for a concrete example where swapping two
    tied rows' input order changes the result. In practice, continuous
    model scores (TransE, popularity's raw degree) rarely tie exactly.

    Raises:
        ValueError: ``labels`` contains no positives at all -- AP is
            mathematically undefined there, and this never silently
            returns an invented value (such as 0.0) in that case.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must have the same shape, got {scores.shape} and {labels.shape}")
    n_pos = labels.sum()
    if n_pos == 0:
        raise ValueError("average_precision is undefined when labels contains no positives")

    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    ranks = np.arange(1, len(sorted_labels) + 1)
    cumulative_true_positives = np.cumsum(sorted_labels)
    precision_at_rank = cumulative_true_positives / ranks

    # Summing precision_at_rank * sorted_labels keeps only the ranks that
    # are themselves positives, per the AP formula above.
    return float(np.sum(precision_at_rank * sorted_labels) / n_pos)


def macro_auprc(per_disease_auprc: Iterable[float]) -> float:
    """Mean of several diseases' own disease-level ``average_precision`` (AP) values.

    **This is the primary AUPRC-style metric for this project.** It weighs
    every disease equally regardless of how many candidates or positives it
    has, which is the fairer comparison across a benchmark of diseases with
    very different numbers of known true drugs (see ``micro_auprc`` for the
    secondary alternative and why it should not replace this one as the
    headline number).

    0.0 for an empty ``per_disease_auprc`` -- a documented convention for
    "no diseases were evaluated at all", distinct from (and not to be
    confused with) a single disease having zero positives, which
    ``average_precision`` itself raises on rather than silently scoring.
    """
    values = list(per_disease_auprc)
    return sum(values) / len(values) if values else 0.0


def micro_auprc(
    scores_by_disease: Sequence[Sequence[float] | np.ndarray],
    labels_by_disease: Sequence[Sequence[int] | np.ndarray],
) -> float:
    """One global Average Precision over every disease's candidates pooled into a single ranking.

    **Secondary, not the headline metric** -- report alongside
    ``macro_auprc``, never in place of it. Pooling raw model scores across
    diseases assumes those scores are on a comparable scale *between*
    diseases, which is not guaranteed: a model (or a simple baseline like
    raw graph degree) can produce systematically higher or lower score
    ranges for one disease's candidate set than another's for reasons that
    have nothing to do with ranking quality, and pooling then lets a
    disease with an incidentally larger score range dominate the global
    ranking out of proportion to how many diseases or positives it
    represents. ``macro_auprc`` is immune to this because each disease's AP
    is computed independently, on its own scores only, before averaging.
    """
    if not scores_by_disease:
        return 0.0
    pooled_scores = np.concatenate([np.asarray(s, dtype=float) for s in scores_by_disease])
    pooled_labels = np.concatenate([np.asarray(l, dtype=float) for l in labels_by_disease])
    return average_precision(pooled_scores, pooled_labels)
