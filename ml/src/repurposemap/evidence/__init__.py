"""Evidence/safety domain: typed contracts and service interfaces, plus one
live adapter, PubMed (``pubmed.py``). ClinicalTrials.gov, DailyMed and openFDA
have contracts only, no live adapter yet; see ``adapters.py``'s docstring."""

from __future__ import annotations

from repurposemap.evidence.adapters import ClinicalTrialAdapter, NotImplementedAdapter, PubMedAdapter, SafetyAdapter
from repurposemap.evidence.pubmed import PubMedError, PubMedEUtilsAdapter, PubMedQuery, PubMedSearchResult
from repurposemap.evidence.models import (
    ClinicalTrialEvidence,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStrength,
    PubMedEvidence,
    SafetyEvidence,
)

__all__ = [
    "ClinicalTrialAdapter",
    "ClinicalTrialEvidence",
    "EvidenceBundle",
    "EvidenceRecord",
    "EvidenceSource",
    "EvidenceStrength",
    "NotImplementedAdapter",
    "PubMedAdapter",
    "PubMedError",
    "PubMedEUtilsAdapter",
    "PubMedEvidence",
    "PubMedQuery",
    "PubMedSearchResult",
    "SafetyAdapter",
    "SafetyEvidence",
]
