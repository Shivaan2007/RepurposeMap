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

from repurposemap.biology import DiseaseBiologyModule, DiseaseMatch, TargetMatch, TargetScorecard
from repurposemap.evidence import EvidenceBundle, EvidenceSource

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


class ModelProvenanceInfo(BaseModel):
    """What the experiment/model provenance layer
    (``repurposemap.provenance``) can safely report about model status --
    read from approved ``ml/experiments/*.json`` files only, never a
    checkpoint tensor, never a locked-test label."""

    most_recent_completed_stage: str | None = Field(default=None, description="stage name of the most recently generated valid experiment record")
    inference_eligible: bool = Field(description="whether an eligible final-training experiment record exists AND a checkpoint is explicitly configured")
    eligibility_reason: str = Field(description="human-readable reason behind inference_eligible")


class SystemInfoResponse(BaseModel):
    """What ``GET /api/v1/system/info`` returns."""

    application: str
    api_version: str
    model: ModelStatus
    data_provenance: Provenance | None = None
    model_provenance: ModelProvenanceInfo | None = None


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


# --------------------------------------------------------------------------
# Live today: /api/v1/evidence
# --------------------------------------------------------------------------


class EvidenceQuery(BaseModel):
    """A request to look up real external evidence for one drug/disease pair."""

    drug_name: str
    disease_name: str
    max_results: int = Field(default=5, ge=1, le=20, description="per source, not a total")


class SourceLookupStatus(BaseModel):
    """Whether one of the four evidence/safety sources answered this
    request. A source that failed is reported here with its error, never
    silently folded into an empty-but-successful result -- see
    ``repurposemap.evidence``'s own "never swallowed into an empty result"
    rule, which this endpoint preserves rather than hides from the caller."""

    source: EvidenceSource
    succeeded: bool
    error: str | None = Field(default=None, description="set only when succeeded is False")


class EvidenceLookupResponse(BaseModel):
    """What ``POST /api/v1/evidence`` returns: one live, synchronous lookup
    across PubMed, ClinicalTrials.gov, openFDA and DailyMed (see that
    router's docstring for why this is synchronous for now). ``bundle`` is
    the same ``EvidenceBundle`` the evidence domain already defines --
    presence of a record here is not a claim of efficacy or safety."""

    query: EvidenceQuery
    bundle: EvidenceBundle
    source_status: list[SourceLookupStatus]
    disclaimer: str = "Research evidence lookup. Presence of a record is not proof of efficacy or safety."


# --------------------------------------------------------------------------
# Live today: /api/v1/disease-biology
# --------------------------------------------------------------------------


class DiseaseBiologyQuery(BaseModel):
    """A request to look up the disease-biology module for one disease.
    Exactly one of ``disease_name``/``disease_id`` should be given --
    ``disease_name`` is resolved to an Open Targets ID first (see
    ``DiseaseBiologyResponse.resolved_from``); ``disease_id`` (an EFO or
    MONDO ID) skips that resolution step."""

    disease_name: str | None = None
    disease_id: str | None = None
    max_targets: int = Field(default=10, ge=1, le=50)


class DiseaseBiologyResponse(BaseModel):
    """What ``POST /api/v1/disease-biology`` returns. ``resolved_from`` is
    set only when ``disease_name`` was given and had to be resolved to an
    Open Targets ID first -- it lists the top candidate matches Open
    Targets' own search returned, so a caller can see what else the name
    could have meant, not just the one picked."""

    module: DiseaseBiologyModule
    resolved_from: list[DiseaseMatch] | None = None
    disclaimer: str = "Research evidence integration from Open Targets. Association with a gene is not a causal claim."


# --------------------------------------------------------------------------
# Live today: /api/v1/target-scorecard
# --------------------------------------------------------------------------


class TargetScorecardQuery(BaseModel):
    """A request to look up the target scorecard for one gene/protein
    target (PLAN.md researcher workflow step 2: "Which proteins are
    worth hitting?"). Exactly one of ``gene_symbol``/``ensembl_id``
    should be given -- ``gene_symbol`` is resolved to an Open Targets
    (Ensembl) ID first (see ``TargetScorecardResponse.resolved_from``);
    ``ensembl_id`` skips that resolution step."""

    gene_symbol: str | None = None
    ensembl_id: str | None = None


class TargetScorecardResponse(BaseModel):
    """What ``POST /api/v1/target-scorecard`` returns. ``resolved_from``
    is set only when ``gene_symbol`` was given and had to be resolved to
    an Ensembl ID first."""

    scorecard: TargetScorecard
    resolved_from: list[TargetMatch] | None = None
    disclaimer: str = "Research evidence integration from Open Targets. Tractability is not a guarantee a drug can be developed."
