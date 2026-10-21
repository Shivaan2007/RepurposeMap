"""``RepurposingHypothesis`` -- typed domain/API model only (Phase F).
Nothing here generates a hypothesis; this only checks the shape."""

from __future__ import annotations

from repurposemap.api.schemas import DiseaseQuery
from repurposemap.hypothesis import RepurposingHypothesis


def test_hypothesis_builds_from_minimal_fields_with_safe_defaults():
    hypothesis = RepurposingHypothesis(
        disease=DiseaseQuery(disease_name="example disease"),
        candidate_drug_id="drug-1",
        candidate_drug_name="Example Drug",
    )
    assert hypothesis.model_score is None
    assert hypothesis.explanation_paths == []
    assert hypothesis.evidence.pubmed == []
    assert hypothesis.safety_flags == []
    assert hypothesis.provenance is None
    assert "not a treatment recommendation" in hypothesis.disclaimer.lower()
