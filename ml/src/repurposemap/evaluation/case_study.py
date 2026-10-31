"""Infrastructure for masked case studies.

A masked case study evaluates one specific, already-known drug-disease
indication: remove that indication edge from the graph in *both*
directions (PrimeKG stores drug -> disease and disease -> drug separately),
rank candidates on the resulting graph, and report where the known drug
landed. The point is to see whether the model rediscovers a known
treatment without having been allowed to see it.

This module holds no real case studies, loads no real graph, and never
touches ``locked_test_manifest.json`` or PrimeKG -- it only defines the
shape a case study takes and the masking step a result cannot be produced
without. Building and running an actual case study (choosing a real
disease/drug pair, loading the real training graph, calling a real ranking
method) is separate, later work that must use a completely separate,
approved, non-locked protocol -- not this module's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from repurposemap.evaluation.indications import INDICATION_RELATION


class MaskingNotConfirmedError(RuntimeError):
    """Raised when a case-study result is requested on a graph that still
    has the known indication edge in it, in either direction."""


@dataclass(frozen=True)
class CaseStudySpec:
    """One case study's identity: which disease, which already-known drug,
    and which ranking method will be run against the masked graph."""

    disease_id: str
    disease_name: str
    known_drug_id: str
    known_drug_name: str
    ranking_method: str
    notes: str = ""


@dataclass
class CaseStudyResult:
    """The outcome of running ``spec`` against a graph that has been
    confirmed masked. Never constructed directly -- see
    ``build_case_study_result``."""

    spec: CaseStudySpec
    rank: int | None
    explanation_paths: tuple = ()
    evidence_summary: dict | None = None
    provenance: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = ()


def mask_known_indication(graph: nx.MultiDiGraph, spec: CaseStudySpec) -> nx.MultiDiGraph:
    """Return a copy of ``graph`` with every ``indication``-relation edge
    between ``spec.known_drug_id`` and ``spec.disease_id`` removed, checked
    in both directions. Never mutates the input graph."""
    masked = graph.copy()
    for source, target in (
        (spec.known_drug_id, spec.disease_id),
        (spec.disease_id, spec.known_drug_id),
    ):
        if not masked.has_edge(source, target):
            continue
        keys_to_remove = [
            key for key, attrs in masked.get_edge_data(source, target).items() if attrs.get("relation") == INDICATION_RELATION
        ]
        for key in keys_to_remove:
            masked.remove_edge(source, target, key)
    return masked


def confirm_masked(graph: nx.MultiDiGraph, spec: CaseStudySpec) -> bool:
    """True iff no ``indication``-relation edge remains between
    ``spec.known_drug_id`` and ``spec.disease_id``, checked in both
    directions. False means the edge is still present somewhere, including
    as a parallel edge ``mask_known_indication`` was never run against."""
    for source, target in (
        (spec.known_drug_id, spec.disease_id),
        (spec.disease_id, spec.known_drug_id),
    ):
        if not graph.has_edge(source, target):
            continue
        attrs = graph.get_edge_data(source, target)
        if any(a.get("relation") == INDICATION_RELATION for a in attrs.values()):
            return False
    return True


def build_case_study_result(
    masked_graph: nx.MultiDiGraph,
    spec: CaseStudySpec,
    rank: int | None,
    *,
    explanation_paths: tuple = (),
    evidence_summary: dict | None = None,
    provenance: dict | None = None,
    limitations: tuple[str, ...] = (),
) -> CaseStudyResult:
    """Build a ``CaseStudyResult``.

    Raises:
        MaskingNotConfirmedError: ``masked_graph`` still has the known
            indication edge between ``spec.known_drug_id`` and
            ``spec.disease_id`` in either direction -- the result would be
            inflated by an edge the ranking method was never meant to see.
    """
    if not confirm_masked(masked_graph, spec):
        raise MaskingNotConfirmedError(
            f"indication edge between drug {spec.known_drug_id!r} and disease {spec.disease_id!r} "
            "is still present in the graph -- call mask_known_indication() first"
        )
    return CaseStudyResult(
        spec=spec,
        rank=rank,
        explanation_paths=tuple(explanation_paths),
        evidence_summary=evidence_summary,
        provenance=dict(provenance or {}),
        limitations=tuple(limitations),
    )
