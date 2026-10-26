"""Typed contracts for the future evidence/safety aggregation domain.

No adapter in this package performs a live PubMed/ClinicalTrials.gov/
DailyMed/openFDA query -- see ``adapters.py``'s own docstring. These
models exist only so later evidence-lookup work has a stable shape to
fill in, instead of being designed out from under the API/frontend
later.

Every record here traces back to a real external source ID (a PMID, an
NCT number, an openFDA report ID) -- there is no "invented paper" or
"invented trial" shape, and nothing here claims a drug treats a disease
or gives dosing/clinical advice. Connectivity or presence of evidence is
never itself a claim of efficacy."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    """Where a piece of evidence would come from. Matches the sources
    named in the project README's "Planned" section."""

    PUBMED = "pubmed"
    CLINICAL_TRIALS = "clinical_trials"
    DAILYMED = "dailymed"
    OPENFDA = "openfda"


class EvidenceStrength(str, Enum):
    """A coarse, non-clinical categorization of evidence type -- never a
    claim about whether the evidence shows efficacy."""

    UNKNOWN = "unknown"
    CASE_REPORT = "case_report"
    OBSERVATIONAL = "observational"
    CONTROLLED_TRIAL = "controlled_trial"
    SYSTEMATIC_REVIEW = "systematic_review"


class EvidenceRecord(BaseModel):
    """One piece of external evidence, traceable to a real source ID.
    Subclassed per source below for source-specific fields."""

    source: EvidenceSource
    source_id: str = Field(description="the source's own identifier, e.g. a PMID or NCT number -- never fabricated")
    title: str
    summary: str | None = None
    url: str | None = None
    strength: EvidenceStrength = EvidenceStrength.UNKNOWN
    retrieved_at: str | None = None


class PubMedEvidence(EvidenceRecord):
    source: EvidenceSource = EvidenceSource.PUBMED
    pmid: str
    journal: str | None = None
    publication_date: str | None = None
    publication_types: list[str] = Field(
        default_factory=list,
        description="PubMed's own publication-type labels (e.g. 'Clinical Trial'), copied as reported; not interpreted",
    )


class ClinicalTrialEvidence(EvidenceRecord):
    source: EvidenceSource = EvidenceSource.CLINICAL_TRIALS
    nct_id: str
    phase: str | None = None
    status: str | None = None


class SafetyEvidence(BaseModel):
    """One raw safety signal from DailyMed/openFDA, before it is
    distilled into an API-facing ``repurposemap.api.schemas.SafetyFlag``.
    Never a dosing recommendation or clinical judgment -- a pointer to a
    real external record only."""

    source: EvidenceSource
    source_id: str
    description: str
    url: str | None = None
    retrieved_at: str | None = None


class EvidenceBundle(BaseModel):
    """All evidence gathered so far for one candidate drug/disease pair.
    Every list defaults empty -- an empty bundle means "no lookups have
    been run yet", never "no evidence exists"."""

    pubmed: list[PubMedEvidence] = Field(default_factory=list)
    clinical_trials: list[ClinicalTrialEvidence] = Field(default_factory=list)
    safety: list[SafetyEvidence] = Field(default_factory=list)
    generated_at: str | None = None
