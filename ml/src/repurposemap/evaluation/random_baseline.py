"""Candidate-conditioned random baseline, for comparison with the path-quality ranking.

Labeled deliberately: this baseline is **candidate-conditioned**. It shuffles
the *same* candidate set the path baseline found for a disease (`rank_drugs`'
output), not the complete drug universe. It answers "how much of the path
baseline's apparent signal is just from finding the right candidates at all,"
not "how would a baseline with no knowledge of the graph do." A path ranking
that cannot beat this is not doing anything useful over its own candidate
discovery.

For a baseline that ranks the complete drug universe, independent of the
path baseline's candidate discovery, see
`repurposemap.evaluation.popularity_baseline`.

The shuffle is deterministic: it is seeded from the evaluation seed and the
disease ID, so the same call always gives the same order, and different
diseases get independent orders.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def random_ranking(candidate_ids: Sequence[str], *, seed: int, disease_id: str) -> list[str]:
    """Return ``candidate_ids`` in a deterministic random order for ``(seed, disease_id)``."""
    shuffled = list(candidate_ids)
    random.Random(f"{seed}:{disease_id}:random-baseline").shuffle(shuffled)
    return shuffled
