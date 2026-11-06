"""Deployable, precomputed drug-ranking serving contract.

See ``schemas.py`` for ``CandidateRankingBundle``, ``loader.py`` for how
the live API reads one from disk, and ``export_rankings.py`` for the
offline tool that produces one from an already-approved ranking output.
This package never computes a ranking itself and never fabricates
candidate data -- see each module's own docstring.
"""

from __future__ import annotations

from repurposemap.serving.schemas import (
    SERVING_SCHEMA_VERSION,
    BundleValidationError,
    CandidateRankingBundle,
    CandidateUniverseDescription,
    ExplanationPathRecord,
    ExplanationPathStepRecord,
    RankedCandidateRecord,
    RankingArtifactProvenance,
    validate_bundle,
)

__all__ = [
    "SERVING_SCHEMA_VERSION",
    "BundleValidationError",
    "CandidateRankingBundle",
    "CandidateUniverseDescription",
    "ExplanationPathRecord",
    "ExplanationPathStepRecord",
    "RankedCandidateRecord",
    "RankingArtifactProvenance",
    "validate_bundle",
]
