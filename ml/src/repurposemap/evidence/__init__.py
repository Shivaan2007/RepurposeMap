"""Evidence/safety domain: typed contracts and service interfaces, plus three
live adapters -- PubMed (``pubmed.py``), ClinicalTrials.gov
(``clinicaltrials.py``) and openFDA (``openfda.py``). DailyMed has contracts
only, no live adapter yet; see ``adapters.py``'s docstring."""

from __future__ import annotations

from repurposemap.evidence.adapters import ClinicalTrialAdapter, NotImplementedAdapter, PubMedAdapter, SafetyAdapter
from repurposemap.evidence.clinicaltrials import (
    ClinicalTrialsAdapter,
    ClinicalTrialsError,
    ClinicalTrialsQuery,
    ClinicalTrialsSearchResult,
)
from repurposemap.evidence.openfda import (
    DrugLabelQuery,
    DrugLabelSearchResult,
    OpenFDAError,
    OpenFDALabelAdapter,
)
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
    "ClinicalTrialsAdapter",
    "ClinicalTrialsError",
    "ClinicalTrialsQuery",
    "ClinicalTrialsSearchResult",
    "DrugLabelQuery",
    "DrugLabelSearchResult",
    "EvidenceBundle",
    "EvidenceRecord",
    "EvidenceSource",
    "EvidenceStrength",
    "NotImplementedAdapter",
    "OpenFDAError",
    "OpenFDALabelAdapter",
    "PubMedAdapter",
    "PubMedError",
    "PubMedEUtilsAdapter",
    "PubMedEvidence",
    "PubMedQuery",
    "PubMedSearchResult",
    "SafetyAdapter",
    "SafetyEvidence",
]
