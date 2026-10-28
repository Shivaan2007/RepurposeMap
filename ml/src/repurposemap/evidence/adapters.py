"""Service interfaces for future evidence/safety adapters.

Nothing in this module performs a network call. (The live PubMed adapter is
in ``pubmed.py``, the live ClinicalTrials.gov adapter is in
``clinicaltrials.py``, and the live openFDA adapter is in ``openfda.py``;
each implements the matching protocol below.) Each adapter here is a
``typing.Protocol`` describing the shape a real PubMed/ClinicalTrials.gov/
DailyMed/openFDA adapter would implement, plus a ``NotImplementedAdapter``
base class any concrete first implementation can subclass -- its methods
raise ``NotImplementedError`` rather than returning fabricated data, so a
caller can never mistake "not built yet" for "queried and found nothing".

Building a real adapter later means implementing one of these protocols
against the named API and returning ``EvidenceBundle``-shaped results --
it does not change this module."""

from __future__ import annotations

from typing import Protocol

from repurposemap.evidence.models import ClinicalTrialEvidence, PubMedEvidence, SafetyEvidence


class PubMedAdapter(Protocol):
    """Looks up literature evidence for a drug/disease pair by PubMed
    search terms. A real implementation calls PubMed's E-utilities and
    returns only records with real PMIDs."""

    def search(self, drug_name: str, disease_name: str, max_results: int = 10) -> list[PubMedEvidence]: ...


class ClinicalTrialAdapter(Protocol):
    """Looks up trial evidence for a drug/disease pair. A real
    implementation calls ClinicalTrials.gov's API and returns only
    records with real NCT IDs."""

    def search(self, drug_name: str, disease_name: str, max_results: int = 10) -> list[ClinicalTrialEvidence]: ...


class SafetyAdapter(Protocol):
    """Looks up safety signals for a drug from DailyMed/openFDA. A real
    implementation returns only records tied to a real source report ID
    -- never a dosing recommendation."""

    def search(self, drug_name: str, max_results: int = 10) -> list[SafetyEvidence]: ...


class NotImplementedAdapter:
    """Shared base for adapters that have no real backend wired in yet.
    Every method raises instead of returning an empty list, so a caller
    cannot mistake "not implemented" for "queried, found nothing"."""

    def search(self, *_args: object, **_kwargs: object) -> list[object]:
        raise NotImplementedError(f"{type(self).__name__} has no live backend configured yet")
