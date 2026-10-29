"""Open Targets adapter -- mocked HTTP only, no network.

The JSON below is a SYNTHETIC fixture in the real GraphQL response shape.
Gene symbols, IDs and scores are invented (DEMO ...) and say nothing
about any real gene or disease."""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone

import pytest

from repurposemap.biology.models import Pathway
from repurposemap.biology.open_targets import (
    OpenTargetsAdapter,
    OpenTargetsNotFoundError,
    OpenTargetsParseError,
    OpenTargetsRequestError,
)

RETRIEVED = "2026-10-29T12:00:00Z"


def graphql_response(data=None, errors=None):
    body = {}
    if data is not None:
        body["data"] = data
    if errors is not None:
        body["errors"] = errors
    return json.dumps(body).encode()


def demo_target(ensembl_id, *, symbol="DEMOG1", score=0.8, genetic_score=0.9, pathways=None):
    return {
        "score": score,
        "datatypeScores": [{"id": "genetic_association", "score": genetic_score}, {"id": "literature", "score": 0.5}],
        "target": {
            "id": ensembl_id,
            "approvedSymbol": symbol,
            "approvedName": f"demo gene {symbol}",
            "pathways": pathways if pathways is not None else [{"pathway": "DEMO pathway", "pathwayId": "R-DEMO-1", "topLevelTerm": "Demo term"}],
        },
    }


def demo_disease_response(rows, *, disease_id="MONDO_0000001", count=None):
    return graphql_response(
        {
            "disease": {
                "id": disease_id,
                "name": "demo disease",
                "description": "A demo disease.",
                "associatedTargets": {"count": len(rows) if count is None else count, "rows": rows},
            }
        }
    )


class FakeFetcher:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, body, timeout_s):
        self.calls.append((url, body, timeout_s))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_adapter(fetcher, **kwargs):
    return OpenTargetsAdapter(fetcher=fetcher, now=lambda: datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc), **kwargs)


def test_resolve_disease_returns_candidates_in_order():
    body = graphql_response({"search": {"hits": [
        {"id": "EFO_0000768", "name": "idiopathic pulmonary fibrosis", "score": 5903.7},
        {"id": "MONDO_0002771", "name": "pulmonary fibrosis", "score": 922.6},
    ]}})
    fetcher = FakeFetcher(body)
    matches = make_adapter(fetcher, timeout_s=7.0).resolve_disease("pulmonary fibrosis", max_results=2)
    assert [m.id for m in matches] == ["EFO_0000768", "MONDO_0002771"]
    assert matches[0].name == "idiopathic pulmonary fibrosis"
    assert matches[0].search_score == 5903.7

    (url, raw_body, timeout), = fetcher.calls
    assert url == "https://api.platform.opentargets.org/api/v4/graphql"
    assert timeout == 7.0
    sent = json.loads(raw_body)
    assert sent["variables"] == {"q": "pulmonary fibrosis", "size": 2}
    assert "ResolveDisease" in sent["query"]


@pytest.mark.parametrize("max_results", [0, 51])
def test_resolve_disease_rejects_out_of_range_limits(max_results):
    with pytest.raises(ValueError):
        make_adapter(FakeFetcher()).resolve_disease("x", max_results=max_results)


def test_get_disease_biology_extracts_targets_ranked_and_scored():
    rows = [
        demo_target("ENSG_DEMO_1", symbol="DEMOG1", score=0.85, genetic_score=0.94),
        demo_target("ENSG_DEMO_2", symbol="DEMOG2", score=0.68, genetic_score=0.88, pathways=[]),
    ]
    fetcher = FakeFetcher(demo_disease_response(rows, count=4549))
    module = make_adapter(fetcher).get_disease_biology("MONDO_0000001", max_targets=2)

    assert module.disease_id == "MONDO_0000001"
    assert module.disease_name == "demo disease"
    assert module.disease_description == "A demo disease."
    assert module.total_associated_targets == 4549
    assert len(module.targets) == 2

    first, second = module.targets
    assert first.ensembl_id == "ENSG_DEMO_1" and first.approved_symbol == "DEMOG1"
    assert first.overall_score == 0.85
    assert first.genetic_association_score == 0.94
    assert first.pathways == [Pathway(pathway_id="R-DEMO-1", pathway="DEMO pathway", top_level_term="Demo term")]
    assert second.pathways == []
    assert module.retrieved_at == RETRIEVED


def test_get_disease_biology_reports_none_not_zero_when_no_genetic_score():
    row = demo_target("ENSG_DEMO_3", score=0.5)
    row["datatypeScores"] = [{"id": "literature", "score": 0.5}]  # no genetic_association entry at all
    fetcher = FakeFetcher(demo_disease_response([row]))
    module = make_adapter(fetcher).get_disease_biology("MONDO_0000001")
    assert module.targets[0].genetic_association_score is None


def test_get_disease_biology_raises_not_found_for_a_missing_disease():
    fetcher = FakeFetcher(graphql_response({"disease": None}))
    with pytest.raises(OpenTargetsNotFoundError, match="MONDO_9999999"):
        make_adapter(fetcher).get_disease_biology("MONDO_9999999")


def test_get_disease_biology_sends_genetic_association_ordering():
    fetcher = FakeFetcher(demo_disease_response([]))
    make_adapter(fetcher).get_disease_biology("MONDO_0000001", max_targets=15)
    (_, raw_body, _), = fetcher.calls
    sent = json.loads(raw_body)
    assert sent["variables"] == {"efoId": "MONDO_0000001", "size": 15}
    assert "genetic_association desc" in sent["query"]


@pytest.mark.parametrize("max_targets", [0, 51])
def test_get_disease_biology_rejects_out_of_range_limits(max_targets):
    with pytest.raises(ValueError):
        make_adapter(FakeFetcher()).get_disease_biology("MONDO_0000001", max_targets=max_targets)


def test_graphql_errors_payload_raises_instead_of_an_empty_result():
    body = graphql_response(errors=[{"message": "disease not found"}])
    with pytest.raises(OpenTargetsRequestError, match="disease not found"):
        make_adapter(FakeFetcher(body)).resolve_disease("x")


@pytest.mark.parametrize("body", [b"not json", json.dumps({"nope": 1}).encode(), json.dumps({"data": "x"}).encode()])
def test_malformed_bodies_raise_parse_error(body):
    with pytest.raises(OpenTargetsParseError):
        make_adapter(FakeFetcher(body)).resolve_disease("x")


def test_network_failure_raises_instead_of_returning_an_empty_result():
    adapter = make_adapter(FakeFetcher(urllib.error.URLError("no route")))
    with pytest.raises(OpenTargetsRequestError, match="no route"):
        adapter.resolve_disease("x")
    with pytest.raises(OpenTargetsRequestError):
        make_adapter(FakeFetcher(TimeoutError("timed out"))).resolve_disease("x")


def test_http_error_is_surfaced_with_its_status():
    error = urllib.error.HTTPError("https://example.invalid", 500, "Server Error", None, None)
    with pytest.raises(OpenTargetsRequestError) as raised:
        make_adapter(FakeFetcher(error)).resolve_disease("x")
    assert raised.value.status == 500


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        OpenTargetsAdapter(timeout_s=0)


def demo_scorecard_response(*, tractability=None, prioritisation_items=None, expression_rows=None, target_id="ENSG_DEMO_1"):
    return graphql_response(
        {
            "target": {
                "id": target_id,
                "approvedSymbol": "DEMOG1",
                "approvedName": "demo gene 1",
                "tractability": tractability if tractability is not None else [],
                "prioritisation": {"items": prioritisation_items if prioritisation_items is not None else []},
                "baselineExpression": {"rows": expression_rows if expression_rows is not None else []},
            }
        }
    )


def test_resolve_target_returns_candidates_in_order():
    body = graphql_response({"search": {"hits": [
        {"id": "ENSG_DEMO_1", "name": "DEMOG1", "score": 5466.2},
        {"id": "ENSG_DEMO_2", "name": "DEMOG2", "score": 42.9},
    ]}})
    fetcher = FakeFetcher(body)
    matches = make_adapter(fetcher).resolve_target("DEMOG1", max_results=2)
    assert [m.id for m in matches] == ["ENSG_DEMO_1", "ENSG_DEMO_2"]
    assert matches[0].search_score == 5466.2

    (_, raw_body, _), = fetcher.calls
    sent = json.loads(raw_body)
    assert sent["variables"] == {"q": "DEMOG1", "size": 2}
    assert "ResolveTarget" in sent["query"]


@pytest.mark.parametrize("max_results", [0, 51])
def test_resolve_target_rejects_out_of_range_limits(max_results):
    with pytest.raises(ValueError):
        make_adapter(FakeFetcher()).resolve_target("x", max_results=max_results)


def test_get_target_scorecard_extracts_tractability_and_prioritisation():
    tractability = [{"label": "Approved Drug", "modality": "SM", "value": True}, {"label": "High-Quality Pocket", "modality": "SM", "value": False}]
    items = [{"key": "geneticConstraint", "value": "-0.76"}, {"key": "hasLigand", "value": "1"}]
    fetcher = FakeFetcher(demo_scorecard_response(tractability=tractability, prioritisation_items=items))
    scorecard = make_adapter(fetcher).get_target_scorecard("ENSG_DEMO_1")

    assert scorecard.ensembl_id == "ENSG_DEMO_1"
    assert scorecard.approved_symbol == "DEMOG1"
    assert len(scorecard.tractability) == 2
    assert scorecard.tractability[0].label == "Approved Drug" and scorecard.tractability[0].value is True
    assert scorecard.prioritisation_factors == {"geneticConstraint": -0.76, "hasLigand": 1.0}
    assert scorecard.retrieved_at == RETRIEVED


def test_get_target_scorecard_drops_non_numeric_prioritisation_values():
    items = [{"key": "weird", "value": "not-a-number"}, {"key": "ok", "value": "0.5"}]
    fetcher = FakeFetcher(demo_scorecard_response(prioritisation_items=items))
    scorecard = make_adapter(fetcher).get_target_scorecard("ENSG_DEMO_1")
    assert scorecard.prioritisation_factors == {"ok": 0.5}


def test_get_target_scorecard_ranks_tissues_by_median_and_caps_at_ten():
    rows = [
        {"tissueBiosample": {"biosampleName": f"tissue{i}"}, "median": float(i), "unit": "TPM"}
        for i in range(15)
    ]
    fetcher = FakeFetcher(demo_scorecard_response(expression_rows=rows))
    scorecard = make_adapter(fetcher).get_target_scorecard("ENSG_DEMO_1")
    assert len(scorecard.top_tissue_expression) == 10
    assert [t.tissue for t in scorecard.top_tissue_expression] == [f"tissue{i}" for i in range(14, 4, -1)]
    assert scorecard.top_tissue_expression[0].median == 14.0


def test_get_target_scorecard_drops_rows_without_a_quantified_median():
    rows = [
        {"tissueBiosample": {"biosampleName": "tissueA"}, "median": None, "unit": "PPB"},
        {"tissueBiosample": None, "median": 5.0, "unit": "TPM"},
        {"tissueBiosample": {"biosampleName": "tissueB"}, "median": 3.0, "unit": "TPM"},
    ]
    fetcher = FakeFetcher(demo_scorecard_response(expression_rows=rows))
    scorecard = make_adapter(fetcher).get_target_scorecard("ENSG_DEMO_1")
    assert [t.tissue for t in scorecard.top_tissue_expression] == ["tissueB"]


def test_get_target_scorecard_raises_not_found_for_a_missing_target():
    fetcher = FakeFetcher(graphql_response({"target": None}))
    with pytest.raises(OpenTargetsNotFoundError, match="ENSG_FAKE"):
        make_adapter(fetcher).get_target_scorecard("ENSG_FAKE")
