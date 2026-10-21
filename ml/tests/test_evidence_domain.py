"""Evidence/safety domain foundation -- typed contracts only, no live
lookups (Phase E)."""

from __future__ import annotations

import pytest

from repurposemap.evidence import (
    ClinicalTrialEvidence,
    EvidenceBundle,
    EvidenceSource,
    EvidenceStrength,
    NotImplementedAdapter,
    PubMedEvidence,
    SafetyEvidence,
)


def test_pubmed_evidence_requires_a_real_pmid_field():
    record = PubMedEvidence(pmid="12345678", title="A real-looking paper title", source_id="12345678")
    assert record.source == EvidenceSource.PUBMED
    assert record.strength == EvidenceStrength.UNKNOWN


def test_clinical_trial_evidence_requires_a_real_nct_id_field():
    record = ClinicalTrialEvidence(nct_id="NCT00000000", title="A trial title", source_id="NCT00000000")
    assert record.source == EvidenceSource.CLINICAL_TRIALS


def test_safety_evidence_has_no_dosing_or_recommendation_field():
    record = SafetyEvidence(source=EvidenceSource.OPENFDA, source_id="report-1", description="a reported adverse event")
    dumped = record.model_dump()
    assert "dosing" not in dumped
    assert "recommendation" not in dumped


def test_empty_evidence_bundle_means_no_lookup_run_yet_not_no_evidence():
    bundle = EvidenceBundle()
    assert bundle.pubmed == []
    assert bundle.clinical_trials == []
    assert bundle.safety == []


def test_not_implemented_adapter_raises_rather_than_returning_empty_results():
    adapter = NotImplementedAdapter()
    with pytest.raises(NotImplementedError):
        adapter.search("aspirin", "headache")
