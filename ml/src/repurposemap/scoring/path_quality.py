"""Interpretable quality score for one directed path.

This is a path-quality heuristic. It is not a confidence or probability, and it
is not a measure of biological truth. Every rule and weight is listed below.

For a path with ``L`` hops, whose intermediate nodes are ``I``:

    length_score   = 1 / L
    relation_score = mean over hops of HOP_WEIGHTS[category of the hop]
                     (preferred 1.0, acceptable 0.5, caution -0.5)
    hub_penalty    = sum over intermediate nodes I of hub_penalty(allowed
                     neighbour count), see repurposemap.scoring.hubs
    total_score    = length_score + relation_score - hub_penalty

A path with any EXCLUDED or unmapped hop is invalid. It gets no total score and
is never ranked. Endpoints are never penalised for being hubs.

Ties are broken deterministically: shorter first, then by the node and relation
sequence. Repeated runs give the same order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from repurposemap.graph.paths import PathStep
from repurposemap.scoring.hubs import hub_penalty
from repurposemap.scoring.relation_policy import HOP_WEIGHTS, RelationCategory, RelationPolicy


@dataclass(frozen=True)
class PathScore:
    """Score breakdown for one path. ``total_score`` is None when the path is invalid."""

    valid: bool
    total_score: float | None
    length_score: float | None
    relation_score: float | None
    hub_penalty: float | None
    hop_categories: tuple[str, ...] = ()
    flags: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "total_score": self.total_score,
            "length_score": self.length_score,
            "relation_score": self.relation_score,
            "hub_penalty": self.hub_penalty,
            "hop_categories": list(self.hop_categories),
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class RankedPath:
    """A valid path with its score. ``raw_index`` is its position in the raw candidates."""

    steps: tuple[PathStep, ...]
    score: PathScore
    raw_index: int


def relation_label(relation: str, display_relation: str) -> str:
    return f"{relation} [{display_relation}]" if display_relation else relation


def score_path(
    steps: Sequence[PathStep],
    policy: RelationPolicy,
    neighbour_counts: Mapping[str, int],
) -> PathScore:
    """Score one path. Invalid paths get a breakdown with the reasons in ``flags``."""
    if not steps:
        raise ValueError("a path must have at least one hop")

    flags: list[str] = []
    categories: list[RelationCategory] = []
    for step in steps:
        label = relation_label(step.relation, step.display_relation)
        if not policy.is_mapped(step.relation, step.display_relation):
            flags.append(f"unmapped relation: {label}")
        category = policy.category(step.relation, step.display_relation)
        categories.append(category)
        if category == RelationCategory.EXCLUDED:
            flags.append(f"excluded relation: {label}")
        elif category == RelationCategory.CAUTION:
            flags.append(f"caution relation: {label}")

    hop_categories = tuple(category.value for category in categories)
    if any(category == RelationCategory.EXCLUDED for category in categories):
        return PathScore(
            valid=False,
            total_score=None,
            length_score=None,
            relation_score=None,
            hub_penalty=None,
            hop_categories=hop_categories,
            flags=tuple(flags),
        )

    length = len(steps)
    length_score = 1 / length
    relation_score = sum(HOP_WEIGHTS[category] for category in categories) / length

    penalty = 0.0
    # Intermediate nodes are the targets of every hop except the last one.
    for step in steps[:-1]:
        count = neighbour_counts.get(step.target, 0)
        step_penalty = hub_penalty(count)
        if step_penalty > 0:
            name = step.target_name or step.target
            flags.append(f"hub intermediate: {name} ({count} allowed neighbours, penalty {step_penalty:g})")
            penalty += step_penalty

    total = length_score + relation_score - penalty
    return PathScore(
        valid=True,
        total_score=total,
        length_score=length_score,
        relation_score=relation_score,
        hub_penalty=penalty,
        hop_categories=hop_categories,
        flags=tuple(flags),
    )


def rank_paths(
    paths: Sequence[Sequence[PathStep]],
    policy: RelationPolicy,
    neighbour_counts: Mapping[str, int],
) -> list[RankedPath]:
    """Score ``paths`` and return only the valid ones, best first.

    The raw candidates are not changed. Each ranked path keeps its position in
    ``paths`` as ``raw_index``. Invalid paths are dropped here, so use
    ``score_path`` directly to see why a path was rejected.
    """
    ranked = []
    for raw_index, steps in enumerate(paths):
        score = score_path(steps, policy, neighbour_counts)
        if score.valid:
            ranked.append(RankedPath(steps=tuple(steps), score=score, raw_index=raw_index))
    ranked.sort(key=_sort_key)
    return ranked


def explain_ranking(ranked: Sequence[RankedPath]) -> list[str]:
    """For each path after the first, say why it ranks below the one before it.

    Only components that differ are named. Returns one line per path after the
    first, in the same order.
    """
    lines = []
    for position in range(1, len(ranked)):
        higher, lower = ranked[position - 1].score, ranked[position].score
        parts = []
        if len(ranked[position - 1].steps) != len(ranked[position].steps):
            parts.append(
                f"{len(ranked[position - 1].steps)} hop(s) vs {len(ranked[position].steps)} hop(s)"
            )
        if higher.relation_score != lower.relation_score:
            parts.append(f"relation score {higher.relation_score:.2f} vs {lower.relation_score:.2f}")
        if higher.hub_penalty != lower.hub_penalty:
            parts.append(f"hub penalty {higher.hub_penalty:.2f} vs {lower.hub_penalty:.2f}")
        if not parts:
            parts.append("tied on score; ordered by node and relation sequence")
        lines.append(f"#{position} ranks above #{position + 1}: " + "; ".join(parts))
    return lines


def _sort_key(ranked: RankedPath) -> tuple:
    signature = tuple(
        (step.source, step.relation, step.display_relation, step.target) for step in ranked.steps
    )
    return (-ranked.score.total_score, len(ranked.steps), signature)
