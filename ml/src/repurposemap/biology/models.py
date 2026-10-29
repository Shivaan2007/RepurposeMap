"""Typed contracts for the disease-biology domain (PLAN.md's researcher
workflow steps 1 and 2: "What goes wrong in this disease?" -- genes,
pathways and tissues linked to the disease, ranked by strength of
genetic evidence -- and "Which proteins are worth hitting?" -- a target
scorecard of genetic evidence, tissue expression and tractability).

Every record traces back to a real Open Targets Platform identifier (an
EFO/MONDO disease ID, an Ensembl gene ID, a Reactome pathway ID) -- there
is no invented gene-disease link. Presence of an association here is a
research signal from Open Targets' own evidence integration, never a
claim that the gene causes the disease."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiseaseMatch(BaseModel):
    """One candidate disease returned when resolving a free-text name to
    an Open Targets disease ID (EFO or MONDO)."""

    id: str
    name: str
    search_score: float = Field(description="Open Targets' own search relevance score, not a biological score")


class Pathway(BaseModel):
    """One Reactome pathway a target participates in."""

    pathway_id: str
    pathway: str
    top_level_term: str


class TargetAssociation(BaseModel):
    """One gene/protein target associated with the disease, with Open
    Targets' own evidence-integration scores. ``genetic_association_score``
    is ``None`` when Open Targets has no genetic-evidence datatype score
    for this target/disease pair -- never defaulted to 0, which would
    read as "evidence against" rather than "no genetic evidence datatype
    scored"."""

    ensembl_id: str
    approved_symbol: str
    approved_name: str
    overall_score: float = Field(description="Open Targets' overall association score (0-1), across all datatypes")
    genetic_association_score: float | None = Field(
        default=None, description="the 'genetic_association' datatype score (0-1), if Open Targets reports one"
    )
    pathways: list[Pathway] = Field(default_factory=list)


class DiseaseBiologyModule(BaseModel):
    """What Open Targets reports for one disease: its own description,
    plus the targets most strongly associated with it, ranked by genetic
    evidence. ``total_associated_targets`` is Open Targets' total match
    count, independent of how many are returned in ``targets``."""

    disease_id: str
    disease_name: str
    disease_description: str | None = None
    total_associated_targets: int
    targets: list[TargetAssociation] = Field(default_factory=list)
    retrieved_at: str


class TargetMatch(BaseModel):
    """One candidate target returned when resolving a free-text gene
    symbol to an Open Targets target ID (Ensembl)."""

    id: str
    name: str
    search_score: float = Field(description="Open Targets' own search relevance score, not a biological score")


class TractabilityAssessment(BaseModel):
    """One tractability criterion Open Targets checked for a target --
    e.g. whether it has an approved drug, a druggable pocket, or a
    high-quality ligand, for a given drug modality (small molecule,
    antibody, PROTAC, other). ``value`` is the criterion's own
    true/false result, not a composite score."""

    label: str
    modality: str
    value: bool


class TissueExpression(BaseModel):
    """The target's expression level in one tissue, from Open Targets'
    baseline expression data (Expression Atlas / HPA / GTEx, as Open
    Targets integrates them). ``median`` is ``None`` when Open Targets
    reports no quantified value for this tissue/unit pair -- never
    defaulted to 0, which would read as "not expressed" rather than
    "not quantified"."""

    tissue: str
    median: float | None = None
    unit: str


class TargetScorecard(BaseModel):
    """What Open Targets reports for one target in isolation (no
    disease context): tractability, Open Targets' own target
    prioritisation factors, and the tissues where it is most highly
    expressed. ``prioritisation_factors`` keeps Open Targets' own factor
    names (e.g. ``geneticConstraint``, ``mouseKOScore``,
    ``tissueSpecificity``) as keys -- see the Open Targets documentation
    for what each one means; this never re-labels or re-scores them."""

    ensembl_id: str
    approved_symbol: str
    approved_name: str
    tractability: list[TractabilityAssessment] = Field(default_factory=list)
    prioritisation_factors: dict[str, float] = Field(default_factory=dict)
    top_tissue_expression: list[TissueExpression] = Field(default_factory=list)
    retrieved_at: str
