"""Evidence/safety domain: typed contracts and service interfaces, plus four
live adapters -- PubMed (``pubmed.py``), ClinicalTrials.gov
(``clinicaltrials.py``), openFDA (``openfda.py``) and DailyMed
(``dailymed.py``). See ``adapters.py``'s docstring for the shared adapter
protocols."""

from __future__ import annotations

from repurposemap.evidence.adapters import ClinicalTrialAdapter, NotImplementedAdapter, PubMedAdapter, SafetyAdapter
from repurposemap.evidence.clinicaltrials import (
    ClinicalTrialsAdapter,
    ClinicalTrialsError,
    ClinicalTrialsQuery,
    ClinicalTrialsSearchResult,
)
from repurposemap.evidence.dailymed import (
    DailyMedAdapter,
    DailyMedError,
    DailyMedQuery,
    DailyMedSearchResult,
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
    "DailyMedAdapter",
    "DailyMedError",
    "DailyMedQuery",
    "DailyMedSearchResult",
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
