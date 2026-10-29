"""Open Targets Platform adapter: disease/target-name resolution, the
disease-biology module (PLAN.md researcher workflow step 1), and the
target scorecard (step 2).

GraphQL queries against the public Open Targets Platform API
(``https://api.platform.opentargets.org/api/v4/graphql``, no API key
required):

- ``resolve_disease`` / ``resolve_target``: free-text name -> candidate
  IDs (EFO/MONDO for diseases, Ensembl for targets), via the API's own
  full-text ``search``.
- ``get_disease_biology``: for one disease ID, its targets ranked by
  genetic-association evidence (``orderByScore: "genetic_association
  desc"``), each with its Reactome pathways -- one round trip, no
  separate per-target fetch.
- ``get_target_scorecard``: for one target ID (no disease context),
  its tractability assessments, Open Targets' own target-prioritisation
  factors, and its highest-expressing tissues.

What a result means: Open Targets has integrated evidence (GWAS,
ClinVar, animal models, literature, ...) connecting this gene to this
disease (or, for a scorecard, characterizing the gene itself), scored
and ranked by its own pipeline. This is not a causal claim and never
substitutes for the actual evidence review PLAN.md's researcher
workflow is meant to support.

Failures are never turned into an empty result. A network error, a
non-200 reply, a GraphQL ``errors`` payload, or an unparseable body
raises ``OpenTargetsError``, so "the lookup failed" cannot be mistaken
for "Open Targets has nothing".

    adapter = OpenTargetsAdapter()
    matches = adapter.resolve_disease("pulmonary arterial hypertension")
    module = adapter.get_disease_biology(matches[0].id, max_targets=10)
    scorecard = adapter.get_target_scorecard(module.targets[0].ensembl_id)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone

from repurposemap.biology.models import (
    DiseaseBiologyModule,
    DiseaseMatch,
    Pathway,
    TargetAssociation,
    TargetMatch,
    TargetScorecard,
    TissueExpression,
    TractabilityAssessment,
)

OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
MAX_RESULTS_LIMIT = 50
TOP_TISSUE_EXPRESSION_LIMIT = 10

_RESOLVE_DISEASE_QUERY = """
query ResolveDisease($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: $size}) {
    hits { id name score }
  }
}
"""

_RESOLVE_TARGET_QUERY = """
query ResolveTarget($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: $size}) {
    hits { id name score }
  }
}
"""

_TARGET_SCORECARD_QUERY = """
query TargetScorecard($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    tractability { label modality value }
    prioritisation { items { key value } }
    baselineExpression {
      rows { tissueBiosample { biosampleName } median unit }
    }
  }
}
"""

_DISEASE_BIOLOGY_QUERY = """
query DiseaseBiology($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    description
    associatedTargets(page: {index: 0, size: $size}, orderByScore: "genetic_association desc") {
      count
      rows {
        score
        datatypeScores { id score }
        target {
          id
          approvedSymbol
          approvedName
          pathways { pathway pathwayId topLevelTerm }
        }
      }
    }
  }
}
"""

Fetcher = Callable[[str, bytes, float], bytes]


class OpenTargetsError(Exception):
    """An Open Targets lookup failed. Never swallowed into an empty result."""


class OpenTargetsRequestError(OpenTargetsError):
    """The request did not complete: network error, timeout, HTTP error, or a GraphQL ``errors`` payload."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OpenTargetsParseError(OpenTargetsError):
    """Open Targets replied, but the body was not the JSON that was expected."""


class OpenTargetsNotFoundError(OpenTargetsError):
    """The disease or target ID does not exist in Open Targets. Distinct
    from a request failure: the lookup completed, and Open Targets
    explicitly reported no such entity."""


def _urllib_post(url: str, body: bytes, timeout_s: float) -> bytes:
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"User-Agent": "repurposemap/0.1 (research tool)", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


class OpenTargetsAdapter:
    def __init__(
        self,
        *,
        url: str = OPEN_TARGETS_GRAPHQL_URL,
        timeout_s: float = 10.0,
        fetcher: Fetcher = _urllib_post,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._url = url
        self._timeout_s = timeout_s
        self._fetcher, self._now = fetcher, now

    def _graphql(self, query: str, variables: dict) -> dict:
        body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        try:
            raw = self._fetcher(self._url, body, self._timeout_s)
        except urllib.error.HTTPError as error:
            raise OpenTargetsRequestError(f"Open Targets returned HTTP {error.code}", status=error.code) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise OpenTargetsRequestError(f"Open Targets request failed: {error}") from error

        try:
            payload = json.loads(raw)
        except ValueError as error:
            raise OpenTargetsParseError(f"Open Targets returned invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise OpenTargetsParseError("expected a JSON object")
        if payload.get("errors"):
            raise OpenTargetsRequestError(f"Open Targets reported GraphQL errors: {payload['errors']}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise OpenTargetsParseError("response had no 'data' object")
        return data

    def resolve_disease(self, query: str, max_results: int = 5) -> list[DiseaseMatch]:
        if max_results < 1 or max_results > MAX_RESULTS_LIMIT:
            raise ValueError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}")
        data = self._graphql(_RESOLVE_DISEASE_QUERY, {"q": query, "size": max_results})
        search = data.get("search")
        if not isinstance(search, dict) or "hits" not in search:
            raise OpenTargetsParseError("'search.hits' was missing from the response")
        return [DiseaseMatch(id=hit["id"], name=hit["name"], search_score=hit["score"]) for hit in search["hits"]]

    def get_disease_biology(self, efo_id: str, max_targets: int = 10) -> DiseaseBiologyModule:
        if max_targets < 1 or max_targets > MAX_RESULTS_LIMIT:
            raise ValueError(f"max_targets must be between 1 and {MAX_RESULTS_LIMIT}")
        data = self._graphql(_DISEASE_BIOLOGY_QUERY, {"efoId": efo_id, "size": max_targets})
        disease = data.get("disease")
        if disease is None:
            raise OpenTargetsNotFoundError(f"Open Targets has no disease with ID {efo_id!r}")

        associated = disease.get("associatedTargets") or {}
        rows = associated.get("rows") or []
        targets = []
        for row in rows:
            target = row["target"]
            datatype_scores = {s["id"]: s["score"] for s in row.get("datatypeScores", [])}
            targets.append(
                TargetAssociation(
                    ensembl_id=target["id"],
                    approved_symbol=target["approvedSymbol"],
                    approved_name=target["approvedName"],
                    overall_score=row["score"],
                    genetic_association_score=datatype_scores.get("genetic_association"),
                    pathways=[
                        Pathway(pathway_id=p["pathwayId"], pathway=p["pathway"], top_level_term=p["topLevelTerm"])
                        for p in target.get("pathways", [])
                    ],
                )
            )

        return DiseaseBiologyModule(
            disease_id=disease["id"],
            disease_name=disease["name"],
            disease_description=disease.get("description"),
            total_associated_targets=associated.get("count", 0),
            targets=targets,
            retrieved_at=self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def resolve_target(self, query: str, max_results: int = 5) -> list[TargetMatch]:
        if max_results < 1 or max_results > MAX_RESULTS_LIMIT:
            raise ValueError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}")
        data = self._graphql(_RESOLVE_TARGET_QUERY, {"q": query, "size": max_results})
        search = data.get("search")
        if not isinstance(search, dict) or "hits" not in search:
            raise OpenTargetsParseError("'search.hits' was missing from the response")
        return [TargetMatch(id=hit["id"], name=hit["name"], search_score=hit["score"]) for hit in search["hits"]]

    def get_target_scorecard(self, ensembl_id: str) -> TargetScorecard:
        data = self._graphql(_TARGET_SCORECARD_QUERY, {"ensemblId": ensembl_id})
        target = data.get("target")
        if target is None:
            raise OpenTargetsNotFoundError(f"Open Targets has no target with ID {ensembl_id!r}")

        tractability = [
            TractabilityAssessment(label=t["label"], modality=t["modality"], value=t["value"])
            for t in target.get("tractability", [])
        ]

        prioritisation_factors = {}
        for item in (target.get("prioritisation") or {}).get("items", []):
            try:
                prioritisation_factors[item["key"]] = float(item["value"])
            except (TypeError, ValueError):
                pass  # a non-numeric factor value is dropped, never coerced into a misleading number

        rows = (target.get("baselineExpression") or {}).get("rows", [])
        scored_rows = [
            TissueExpression(tissue=row["tissueBiosample"]["biosampleName"], median=row["median"], unit=row["unit"])
            for row in rows
            if row.get("tissueBiosample") is not None and row.get("median") is not None
        ]
        top_tissue_expression = sorted(scored_rows, key=lambda r: r.median, reverse=True)[:TOP_TISSUE_EXPRESSION_LIMIT]

        return TargetScorecard(
            ensembl_id=target["id"],
            approved_symbol=target["approvedSymbol"],
            approved_name=target["approvedName"],
            tractability=tractability,
            prioritisation_factors=prioritisation_factors,
            top_tissue_expression=top_tissue_expression,
            retrieved_at=self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
