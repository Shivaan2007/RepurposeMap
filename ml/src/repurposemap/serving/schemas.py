"""Deployable, precomputed drug-ranking serving contract.

Defines ``CandidateRankingBundle``: the on-disk shape a real ranking
output is exported into (see ``export_rankings.py``) and the live API
reads back (see ``loader.py``) to serve ``POST /api/v1/rank`` honestly
once a bundle exists.

Nothing in this module computes or invents a ranking -- it only
describes and validates the shape one takes. A bundle with fabricated
candidate data is a process violation, not something this schema can
prevent by itself; the enforcement is procedural (see
``docs/ARCHITECTURE.md`` and ``RGCN_EVALUATION_PROTOCOL.md``), not
mechanical.

Deliberately separate from ``repurposemap.api.schemas``: this is the
offline-artifact/on-disk schema. The API layer (``api/ranking.py``) maps
a loaded bundle onto its own ``RankingResponse``/``CandidateDrug`` types
rather than serving this schema directly, so the two can evolve
independently -- a request-scoped API field never needs to leak into
what gets written to disk.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SERVING_SCHEMA_VERSION = 1


class RankingArtifactProvenance(BaseModel):
    """Where this bundle's ranking came from -- never left blank or
    invented. ``ranking_method`` names the exact scoring approach (e.g.
    ``"path_quality"``, ``"transe"``, ``"rgcn"``); ``source_experiment_file``
    ties it back to one file under ``ml/experiments/`` when the ranking
    derives from a trained model, and stays ``None`` for the heuristic
    path/popularity baselines, which have no experiment-record
    checkpoint to point to."""

    ranking_method: str
    source_experiment_file: str | None = Field(default=None, description="a filename under ml/experiments/, or None for a baseline with no checkpoint")
    source_commit: str | None = Field(default=None, description="the git commit the ranking code ran at, if known")


class CandidateUniverseDescription(BaseModel):
    """What set of drugs this bundle's ranking was computed over -- the
    caveat every RepurposeMap ranking carries (see
    ``RGCN_EVALUATION_PROTOCOL.md``): a candidate's rank is only
    meaningful relative to this stated universe, never "all drugs that
    exist"."""

    description: str
    size: int | None = None


class ExplanationPathStepRecord(BaseModel):
    """One hop of a stored explanation path. Mirrors
    ``repurposemap.api.schemas.ExplanationStep`` field-for-field so the
    API layer can map one onto the other without reinterpreting
    anything."""

    source_id: str
    source_name: str
    source_type: str
    relation: str
    display_relation: str | None = None
    target_id: str
    target_name: str
    target_type: str


class ExplanationPathRecord(BaseModel):
    """A full mechanistic path stored alongside a ranked candidate.
    ``path_score`` is the same heuristic ``repurposemap.scoring.path_quality``
    defines, not a probability or model confidence."""

    steps: list[ExplanationPathStepRecord]
    path_score: float | None = None


class RankedCandidateRecord(BaseModel):
    """One ranked drug within a bundle. ``rank`` is 1-indexed and must be
    unique and sorted within a bundle (enforced by ``validate_bundle``).

    Evidence and safety data are deliberately not embedded here: those
    are fetched live from PubMed/ClinicalTrials.gov/openFDA/DailyMed per
    PLAN.md, independent of this bundle's own freshness, so a stale
    ranking bundle never also serves stale evidence."""

    drug_id: str
    drug_name: str
    rank: int = Field(ge=1)
    score: float | None = None
    explanation_paths: list[ExplanationPathRecord] = Field(default_factory=list)


class CandidateRankingBundle(BaseModel):
    """A deployable, precomputed ranking for one disease.

    Produced offline (see ``export_rankings.py``) from a real,
    already-approved ranking output -- never constructed with
    placeholder or synthetic candidate data outside of tests, and never
    the output of this schema module itself."""

    schema_version: int = SERVING_SCHEMA_VERSION
    disease_id: str
    disease_name: str
    ranking_method: str
    provenance: RankingArtifactProvenance
    generated_at: str
    candidate_universe: CandidateUniverseDescription
    candidates: list[RankedCandidateRecord]


class BundleValidationError(ValueError):
    """Raised by ``validate_bundle`` when a bundle is structurally
    dishonest (duplicate ranks, out-of-order ranks, a schema-version
    mismatch) -- never silently repaired or re-sorted on the caller's
    behalf."""


def validate_bundle(bundle: CandidateRankingBundle) -> None:
    """Raise ``BundleValidationError`` if ``bundle`` is structurally
    inconsistent. Pydantic's own field validation (types, ``rank >= 1``)
    already ran by the time a ``CandidateRankingBundle`` exists; this
    checks the cross-field invariants pydantic can't express alone."""

    if bundle.schema_version != SERVING_SCHEMA_VERSION:
        raise BundleValidationError(
            f"bundle for disease {bundle.disease_id!r} has schema_version={bundle.schema_version}, "
            f"expected {SERVING_SCHEMA_VERSION}"
        )
    if not bundle.candidates:
        raise BundleValidationError(f"bundle for disease {bundle.disease_id!r} has an empty candidate list")
    ranks = [c.rank for c in bundle.candidates]
    if len(ranks) != len(set(ranks)):
        raise BundleValidationError(f"duplicate rank(s) in bundle for disease {bundle.disease_id!r}: {ranks}")
    if ranks != sorted(ranks):
        raise BundleValidationError(f"candidates in bundle for disease {bundle.disease_id!r} are not sorted by rank: {ranks}")
