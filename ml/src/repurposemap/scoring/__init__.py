"""Relation policy, hub handling and path-quality scoring.

These are transparent research heuristics. They are not medically validated rules,
and the scores are not confidence values.
"""

from repurposemap.scoring.hubs import (
    EXTREME_HUB_NEIGHBOUR_THRESHOLD,
    HUB_NEIGHBOUR_THRESHOLD,
    NeighbourCounts,
    hub_penalty,
)
from repurposemap.scoring.promiscuity import (
    DrugTargetCounts,
    drug_promiscuity_penalty,
)
from repurposemap.scoring.path_quality import (
    PathScore,
    RankedPath,
    explain_ranking,
    rank_paths,
    score_path,
)
from repurposemap.scoring.relation_policy import (
    HOP_WEIGHTS,
    PRIMEKG_RELATION_POLICY,
    RelationCategory,
    RelationPolicy,
    policy_view,
)

__all__ = [
    "EXTREME_HUB_NEIGHBOUR_THRESHOLD",
    "HOP_WEIGHTS",
    "HUB_NEIGHBOUR_THRESHOLD",
    "PRIMEKG_RELATION_POLICY",
    "PathScore",
    "RankedPath",
    "RelationCategory",
    "RelationPolicy",
    "DrugTargetCounts",
    "NeighbourCounts",
    "drug_promiscuity_penalty",
    "explain_ranking",
    "hub_penalty",
    "policy_view",
    "rank_paths",
    "score_path",
]
