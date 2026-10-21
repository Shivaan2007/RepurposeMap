"""The ``RepurposingHypothesis`` domain object.

Shows how future ranking + evidence + safety components will assemble
into one researcher-facing object -- a typed model only. Nothing here
generates a hypothesis; that still requires a real model (see
``repurposemap.provenance``, currently ``inference_eligible=False``) and
real evidence lookups (see ``repurposemap.evidence``, no live adapter
yet). Every ``RepurposingHypothesis`` is a research hypothesis for expert
review, never a treatment recommendation -- same convention as
``repurposemap.api.schemas.RankingResponse.disclaimer``."""

from __future__ import annotations

from pydantic import BaseModel, Field

from repurposemap.api.schemas import CandidateScore, DiseaseQuery, ExplanationPath, ModelProvenanceInfo, SafetyFlag
from repurposemap.evidence.models import EvidenceBundle


class RepurposingHypothesis(BaseModel):
    """One candidate drug for one disease, with everything that backs
    it: the model score, the mechanistic explanation paths, the
    evidence bundle, safety flags, and provenance for both the model and
    the evidence. Assembling this for real requires a model
    (``provenance.inference_eligible``) and evidence adapters
    (``repurposemap.evidence.adapters``) that do not exist yet."""

    disease: DiseaseQuery
    candidate_drug_id: str
    candidate_drug_name: str
    model_score: CandidateScore | None = None
    explanation_paths: list[ExplanationPath] = Field(default_factory=list)
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    provenance: ModelProvenanceInfo | None = None
    model_identifier: str | None = None
    generated_at: str | None = None
    disclaimer: str = "Research hypothesis. Not a treatment recommendation."
