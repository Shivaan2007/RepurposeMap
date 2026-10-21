"""Evidence/safety domain foundation -- typed contracts and service
interfaces only. No adapter here performs a live PubMed/ClinicalTrials.gov/
DailyMed/openFDA query yet; see ``adapters.py``'s own docstring."""

from __future__ import annotations

from repurposemap.evidence.adapters import ClinicalTrialAdapter, NotImplementedAdapter, PubMedAdapter, SafetyAdapter
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
    "PubMedEvidence",
    "SafetyAdapter",
    "SafetyEvidence",
]
