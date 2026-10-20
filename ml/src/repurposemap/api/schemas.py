"""Pydantic API contracts for RepurposeMap.

Two kinds of model live here: the small, already-answerable ones
(``HealthResponse``, ``SystemInfoResponse``) backing today's two live
endpoints, and the future candidate-ranking contracts
(``DiseaseQuery`` through ``RankingResponse``) that describe the shape a
real ranking endpoint will return once a model is actually plugged in.
None of the second group is served by a real endpoint yet -- they exist so
the OpenAPI schema, and the generated TypeScript types, are correct and
stable from day one, instead of being redesigned out from under the
frontend later.

Every response here is a research hypothesis contract, never a treatment
recommendation -- see ``RankingResponse.disclaimer``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Live today: /health, /api/v1/system/info
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """What ``GET /health`` returns. Deliberately static -- no graph load,
    no model check, just "is the process up and which API version is this"."""

    service: str
    status: Literal["ok"]
    api_version: str


class ModelStatus(BaseModel):
    """Whether a real ranking model is configured. ``available=False`` is
    the honest default until a specific, eligible model is explicitly
    supplied -- never guessed, never silently assumed."""

    available: bool
    model_type: str | None = Field(default=None, description="e.g. 'path-quality', 'transe' -- None until a model is configured")
    ready_for_inference: bool


class Provenance(BaseModel):
    """Dataset/source provenance -- only fields knowable without touching
    the raw file (see ``repurposemap.adapters.primekg.PrimeKGFingerprint``,
    whose ``sha256``/``size_bytes`` require reading the file and are
    deliberately NOT surfaced here)."""

    source: str
    retrieved_date: str | None = None
    version: str | None = None


class SystemInfoResponse(BaseModel):
    """What ``GET /api/v1/system/info`` returns."""

    application: str
    api_version: str
    model: ModelStatus
    data_provenance: Provenance | None = None


# --------------------------------------------------------------------------
# Future candidate-ranking contracts -- not served by a real endpoint yet.
# --------------------------------------------------------------------------


class DiseaseQuery(BaseModel):
    """A request to rank candidate drugs for one disease. Exactly one of
    ``disease_name``/``disease_id`` should be given -- the same
    name-or-canonical-ID convention already used by the CLI's
    ``rank-drugs`` command."""

    disease_name: str | None = None
    disease_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class SafetyFlag(BaseModel):
    """A caution attached to a candidate -- e.g. a known contraindication
    or interaction. Structural placeholder: no safety-data source is wired
    up yet, so this is never populated with invented content."""

    code: str
    severity: Literal["info", "caution", "exclude"]
    description: str


class EvidenceSummary(BaseModel):
    """One piece of graph evidence linking a candidate drug toward a
    disease -- a relation type and the node types it connects, not a path
    in full (see ``ExplanationPath`` for that)."""

    relation: str
    display_relation: str
    source_node_type: str
    target_node_type: str


class ExplanationStep(BaseModel):
    """One hop of an explanation path, mirroring
    ``repurposemap.scoring.path_quality``'s own path representation."""

    source_id: str
    source_name: str
    relation: str
    target_id: str
    target_name: str


class ExplanationPath(BaseModel):
    """A full mechanistic path from drug to disease, as the CLI's
    ``explain-paths``/``rank-drugs`` commands already compute -- a
    hypothesis to review, not evidence of efficacy."""

    steps: list[ExplanationStep]
    path_score: float | None = None


class CandidateScore(BaseModel):
    """One scoring method's verdict on a candidate -- a disease-based
    evaluation may report several (path quality, popularity, a future
    TransE score), never collapsed into one undifferentiated number."""

    value: float
    method: str = Field(description="e.g. 'path_quality', 'popularity', 'transe'")
    rank: int | None = None


class CandidateDrug(BaseModel):
    """One ranked candidate drug for a ``DiseaseQuery``."""

    drug_id: str
    drug_name: str
    scores: list[CandidateScore] = Field(default_factory=list)
    evidence: list[EvidenceSummary] = Field(default_factory=list)
    explanation_paths: list[ExplanationPath] = Field(default_factory=list)
    safety_flags: list[SafetyFlag] = Field(default_factory=list)


class RankingResponse(BaseModel):
    """The future candidate-ranking response shape. ``model_status`` must
    be ``"not_available"`` -- never silently substituted with a mock or
    random ranking -- until a real, eligible model is explicitly wired in."""

    disease_query: DiseaseQuery
    candidates: list[CandidateDrug] = Field(default_factory=list)
    model_status: Literal["not_available", "ready"]
    provenance: Provenance | None = None
    disclaimer: str = "Research hypothesis ranking. Not a treatment recommendation."
