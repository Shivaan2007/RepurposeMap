"""Random-order baseline, for comparison with the path-quality ranking.

It shuffles the *same* candidate set the path baseline found for a disease, so
the comparison is fair: both baselines are scored over the same drugs, judged
by the same metrics. A path ranking that cannot beat this is not doing
anything useful.

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
