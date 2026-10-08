"""Transparent relation policy for PrimeKG edges.

This is a research heuristic, not a medically validated rule. It decides how
much a relation should count when a path is scored. It is written down here so
that anyone can see, and dispute, every decision.

Each edge is classified by the pair ``(relation, display_relation)`` as stored
in PrimeKG's ``kg.csv``. The table below lists all 33 pairs present in the
full file. An unmapped pair is excluded, so a new relation can never silently
count as evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum

import networkx as nx


class RelationCategory(StrEnum):
    """How much a relation counts when a path is scored."""

    # Direct molecular or disease-gene mechanism.
    PREFERRED_MECHANISTIC = "preferred_mechanistic"
    # Real biology, but indirect, contextual or broad.
    ACCEPTABLE = "acceptable"
    # Weak, broad, adverse-effect or ontology-hierarchy link. It is penalised.
    CAUTION = "caution"
    # Not usable as path evidence. A path containing one is invalid.
    EXCLUDED = "excluded"


# Score contribution of one hop, by category. The relation score of a path is
# the mean of its hop weights, so it stays between -0.5 and 1.0.
HOP_WEIGHTS: dict[RelationCategory, float] = {
    RelationCategory.PREFERRED_MECHANISTIC: 1.0,
    RelationCategory.ACCEPTABLE: 0.5,
    RelationCategory.CAUTION: -0.5,
}

_P = RelationCategory.PREFERRED_MECHANISTIC
_A = RelationCategory.ACCEPTABLE
_C = RelationCategory.CAUTION
_X = RelationCategory.EXCLUDED

# Keys are (relation, display_relation) exactly as PrimeKG stores them.
PRIMEKG_RELATION_POLICY: dict[tuple[str, str], RelationCategory] = {
    # Drug to protein. "target" is the drug's binding target. The other three
    # roles describe how the drug is handled in the body, not what it acts on.
    ("drug_protein", "target"): _P,
    ("drug_protein", "enzyme"): _A,
    ("drug_protein", "transporter"): _A,
    ("drug_protein", "carrier"): _A,
    # Protein to disease, and gene-level mechanism.
    ("disease_protein", "associated with"): _P,
    ("protein_protein", "ppi"): _P,
    ("pathway_protein", "interacts with"): _P,
    # Gene ontology annotations are broad, so they count for less.
    ("bioprocess_protein", "interacts with"): _A,
    ("molfunc_protein", "interacts with"): _A,
    ("cellcomp_protein", "interacts with"): _A,
    # Phenotypes and tissues. Context, not direct mechanism.
    ("phenotype_protein", "associated with"): _A,
    ("disease_phenotype_positive", "phenotype present"): _A,
    ("anatomy_protein_present", "expression present"): _A,
    ("disease_phenotype_negative", "phenotype absent"): _C,
    ("anatomy_protein_absent", "expression absent"): _C,
    # Adverse effects and environmental exposures. Penalised, as they do not
    # show how a drug treats a disease.
    ("drug_effect", "side effect"): _C,
    ("exposure_disease", "linked to"): _C,
    ("exposure_protein", "interacts with"): _C,
    ("exposure_bioprocess", "interacts with"): _C,
    ("exposure_molfunc", "interacts with"): _C,
    ("exposure_cellcomp", "interacts with"): _C,
    # Ontology hierarchy ("is a kind of"). These connect unrelated things through
    # broad parent terms, so they are the main source of generic shortcuts.
    ("disease_disease", "parent-child"): _C,
    ("phenotype_phenotype", "parent-child"): _C,
    ("anatomy_anatomy", "parent-child"): _C,
    ("bioprocess_bioprocess", "parent-child"): _C,
    ("molfunc_molfunc", "parent-child"): _C,
    ("cellcomp_cellcomp", "parent-child"): _C,
    ("pathway_pathway", "parent-child"): _C,
    ("exposure_exposure", "parent-child"): _C,
    # Drug-drug interactions say nothing about the disease. Treatment labels
    # are excluded because they are the answer being tested. Using them as
    # path evidence would be circular.
    ("drug_drug", "synergistic interaction"): _X,
    ("indication", "indication"): _X,
    ("contraindication", "contraindication"): _X,
    ("off-label use", "off-label use"): _X,
}


class RelationPolicy:
    """Classifies edges. Unmapped ``(relation, display_relation)`` pairs are excluded."""

    def __init__(
        self,
        categories: Mapping[tuple[str, str], RelationCategory] = PRIMEKG_RELATION_POLICY,
    ) -> None:
        self._categories = dict(categories)

    def is_mapped(self, relation: str, display_relation: str = "") -> bool:
        return (relation, display_relation) in self._categories

    def category(self, relation: str, display_relation: str = "") -> RelationCategory:
        return self._categories.get((relation, display_relation), RelationCategory.EXCLUDED)

    def edge_allowed(self, attrs: Mapping[str, str]) -> bool:
        """True when an edge may appear in a candidate path."""
        return self.category(attrs["relation"], attrs.get("display_relation", "")) != RelationCategory.EXCLUDED


def policy_view(
    graph: nx.MultiDiGraph,
    policy: RelationPolicy,
    hidden_pairs: Iterable[tuple[str, str]] = (),
    extra_filter: Callable[[str, str, tuple[str, str]], bool] | None = None,
) -> nx.MultiDiGraph:
    """Return a read-only view with excluded edges and ``hidden_pairs`` removed.

    ``hidden_pairs`` are ``(source, target)`` node IDs whose edges are hidden in
    the stored direction. Evaluation uses this to hide a held-out indication
    edge. The view is lazy, so no edges are copied.

    ``extra_filter``, if given, is ANDed into the same filter closure instead of
    being applied as a second, separately nested ``subgraph_view``. Composing
    filters this way matters: every edge access during a path search must pass
    through every filter layer wrapping the graph, so stacking two
    ``subgraph_view`` calls (one from a caller, one here) roughly doubles the
    per-edge cost of every search. A profiled disease-evaluation run spent most
    of its time in exactly this nested-view overhead, not in the search itself.
    """
    hidden = frozenset(hidden_pairs)

    def keep(u: str, v: str, key: tuple[str, str]) -> bool:
        if (u, v) in hidden:
            return False
        if not policy.edge_allowed(graph[u][v][key]):
            return False
        return extra_filter is None or extra_filter(u, v, key)

    return nx.subgraph_view(graph, filter_edge=keep)
