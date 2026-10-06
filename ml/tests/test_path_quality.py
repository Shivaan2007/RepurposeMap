"""Relation policy, hub penalty and path-quality score, on hand-built steps. No PrimeKG file needed."""

import pytest

from repurposemap.graph import PathStep
from repurposemap.scoring import (
    PRIMEKG_RELATION_POLICY,
    RelationCategory,
    RelationPolicy,
    explain_ranking,
    hub_penalty,
    rank_paths,
    score_path,
)
from repurposemap.scoring.hubs import EXTREME_HUB_NEIGHBOUR_THRESHOLD, HUB_NEIGHBOUR_THRESHOLD

PREFERRED = ("drug_protein", "target")
ACCEPTABLE = ("drug_protein", "enzyme")
CAUTION = ("drug_effect", "side effect")
EXCLUDED = ("drug_drug", "synergistic interaction")
PREF_GENE_DISEASE = ("disease_protein", "associated with")


def step(relation: tuple[str, str], source: str, target: str, target_name: str = "") -> PathStep:
    return PathStep(
        source=source,
        source_type="x",
        relation=relation[0],
        target=target,
        target_type="x",
        source_name=source,
        target_name=target_name or target,
        display_relation=relation[1],
    )


def one_hop(relation: tuple[str, str]) -> list[PathStep]:
    return [step(relation, "A", "B")]


# Relation policy

# The 33 (relation, display_relation) pairs in PrimeKG's kg.csv, as found by the
# audit of the full file. Hardcoded on purpose: the test must check the policy
# against the real data, not against itself.
PRIMEKG_AUDITED_PAIRS = frozenset({
    ("anatomy_anatomy", "parent-child"),
    ("anatomy_protein_absent", "expression absent"),
    ("anatomy_protein_present", "expression present"),
    ("bioprocess_bioprocess", "parent-child"),
    ("bioprocess_protein", "interacts with"),
    ("cellcomp_cellcomp", "parent-child"),
    ("cellcomp_protein", "interacts with"),
    ("contraindication", "contraindication"),
    ("disease_disease", "parent-child"),
    ("disease_phenotype_negative", "phenotype absent"),
    ("disease_phenotype_positive", "phenotype present"),
    ("disease_protein", "associated with"),
    ("drug_drug", "synergistic interaction"),
    ("drug_effect", "side effect"),
    ("drug_protein", "carrier"),
    ("drug_protein", "enzyme"),
    ("drug_protein", "target"),
    ("drug_protein", "transporter"),
    ("exposure_bioprocess", "interacts with"),
    ("exposure_cellcomp", "interacts with"),
    ("exposure_disease", "linked to"),
    ("exposure_exposure", "parent-child"),
    ("exposure_molfunc", "interacts with"),
    ("exposure_protein", "interacts with"),
    ("indication", "indication"),
    ("molfunc_molfunc", "parent-child"),
    ("molfunc_protein", "interacts with"),
    ("off-label use", "off-label use"),
    ("pathway_pathway", "parent-child"),
    ("pathway_protein", "interacts with"),
    ("phenotype_phenotype", "parent-child"),
    ("phenotype_protein", "associated with"),
    ("protein_protein", "ppi"),
})


def test_policy_maps_exactly_the_audited_primekg_relation_pairs():
    assert len(PRIMEKG_AUDITED_PAIRS) == 33
    assert set(PRIMEKG_RELATION_POLICY) == PRIMEKG_AUDITED_PAIRS


def test_every_audited_pair_has_a_category_and_is_mapped():
    policy = RelationPolicy()
    for relation, display in PRIMEKG_AUDITED_PAIRS:
        assert policy.is_mapped(relation, display)
        assert isinstance(policy.category(relation, display), RelationCategory)


def test_policy_uses_all_four_categories():
    used = set(PRIMEKG_RELATION_POLICY.values())
    assert used == set(RelationCategory)


def test_indication_and_treatment_labels_are_excluded():
    policy = RelationPolicy()
    assert policy.category("indication", "indication") == RelationCategory.EXCLUDED
    assert policy.category("contraindication", "contraindication") == RelationCategory.EXCLUDED
    assert policy.category("off-label use", "off-label use") == RelationCategory.EXCLUDED


def test_unmapped_relation_is_excluded_not_guessed():
    policy = RelationPolicy()
    assert not policy.is_mapped("made_up_relation", "")
    assert policy.category("made_up_relation", "") == RelationCategory.EXCLUDED


# Hop scoring

def test_preferred_relation_scores_full_weight():
    score = score_path(one_hop(PREFERRED), RelationPolicy(), {})
    assert score.valid
    assert score.relation_score == pytest.approx(1.0)
    assert score.length_score == pytest.approx(1.0)
    assert score.hub_penalty == pytest.approx(0.0)
    assert score.total_score == pytest.approx(2.0)


def test_acceptable_relation_scores_half_weight():
    score = score_path(one_hop(ACCEPTABLE), RelationPolicy(), {})
    assert score.relation_score == pytest.approx(0.5)


def test_caution_relation_is_penalised_and_flagged():
    score = score_path(one_hop(CAUTION), RelationPolicy(), {})
    assert score.valid
    assert score.relation_score == pytest.approx(-0.5)
    assert any(flag.startswith("caution relation: drug_effect") for flag in score.flags)
    assert score.total_score < score_path(one_hop(ACCEPTABLE), RelationPolicy(), {}).total_score


def test_excluded_relation_invalidates_the_whole_path():
    steps = [step(PREFERRED, "A", "B"), step(EXCLUDED, "B", "C")]
    score = score_path(steps, RelationPolicy(), {})
    assert not score.valid
    assert score.total_score is None
    assert score.relation_score is None
    assert any(flag.startswith("excluded relation: drug_drug") for flag in score.flags)


def test_unmapped_relation_is_reported_as_unmapped():
    score = score_path(one_hop(("made_up", "")), RelationPolicy(), {})
    assert not score.valid
    assert any(flag.startswith("unmapped relation: made_up") for flag in score.flags)


def test_empty_path_is_rejected():
    with pytest.raises(ValueError):
        score_path([], RelationPolicy(), {})


# Hub penalty

@pytest.mark.parametrize(
    ("neighbours", "expected"),
    [
        (0, 0.0),
        (HUB_NEIGHBOUR_THRESHOLD, 0.0),
        (HUB_NEIGHBOUR_THRESHOLD + 1, 0.25),
        (EXTREME_HUB_NEIGHBOUR_THRESHOLD, 0.25),
        (EXTREME_HUB_NEIGHBOUR_THRESHOLD + 1, 0.75),
    ],
)
def test_hub_penalty_tiers(neighbours, expected):
    assert hub_penalty(neighbours) == pytest.approx(expected)


def test_hub_intermediate_node_is_penalised():
    steps = [step(PREFERRED, "A", "HUB"), step(PREF_GENE_DISEASE, "HUB", "D")]
    counts = {"HUB": 2500}
    score = score_path(steps, RelationPolicy(), counts)
    assert score.hub_penalty == pytest.approx(0.75)
    assert any(flag.startswith("hub intermediate: HUB") for flag in score.flags)


def test_hub_endpoint_is_not_penalised():
    steps = [step(PREFERRED, "HUB", "P"), step(PREF_GENE_DISEASE, "P", "D")]
    counts = {"HUB": 2500, "D": 2500}
    score = score_path(steps, RelationPolicy(), counts)
    assert score.hub_penalty == pytest.approx(0.0)


# Length and ordering

def test_shorter_path_scores_higher_when_relations_are_equal():
    short = [step(PREFERRED, "A", "P"), step(PREF_GENE_DISEASE, "P", "D")]
    long = [step(PREFERRED, "A", "P"), step(PREFERRED, "P", "Q"), step(PREF_GENE_DISEASE, "Q", "D")]
    policy = RelationPolicy()
    assert score_path(short, policy, {}).total_score > score_path(long, policy, {}).total_score


def test_rank_paths_is_deterministic_and_drops_invalid_paths():
    valid_a = [step(PREFERRED, "A", "P1"), step(PREF_GENE_DISEASE, "P1", "D")]
    valid_b = [step(PREFERRED, "A", "P2"), step(PREF_GENE_DISEASE, "P2", "D")]
    invalid = [step(EXCLUDED, "A", "X"), step(PREF_GENE_DISEASE, "X", "D")]
    paths = [invalid, valid_b, valid_a]
    policy = RelationPolicy()

    first = rank_paths(paths, policy, {})
    second = rank_paths(paths, policy, {})

    assert [r.raw_index for r in first] == [2, 1]
    assert [r.raw_index for r in first] == [r.raw_index for r in second]
    assert all(r.score.valid for r in first)


def test_ranked_path_keeps_its_raw_position():
    paths = [one_hop(CAUTION), one_hop(PREFERRED)]
    ranked = rank_paths(paths, RelationPolicy(), {})
    assert ranked[0].raw_index == 1
    assert ranked[1].raw_index == 0


def test_explain_ranking_names_the_differing_component():
    short = [step(PREFERRED, "A", "P"), step(PREF_GENE_DISEASE, "P", "D")]
    long = [step(PREFERRED, "A", "P"), step(PREFERRED, "P", "Q"), step(PREF_GENE_DISEASE, "Q", "D")]
    ranked = rank_paths([long, short], RelationPolicy(), {})
    lines = explain_ranking(ranked)
    assert len(lines) == 1
    assert lines[0].startswith("#1 ranks above #2:")
    assert "2 hop(s) vs 3 hop(s)" in lines[0]


# Structured breakdown

def test_score_breakdown_has_the_documented_keys():
    score = score_path(one_hop(PREFERRED), RelationPolicy(), {})
    breakdown = score.to_dict()
    assert set(breakdown) == {
        "valid", "total_score", "length_score", "relation_score", "hub_penalty", "hop_categories", "flags",
    }
    assert breakdown["hop_categories"] == ["preferred_mechanistic"]


# Synthetic compatibility: any relation names can be used with a custom policy

def test_custom_policy_scores_synthetic_relation_names():
    policy = RelationPolicy({("drug_targets_protein", ""): RelationCategory.PREFERRED_MECHANISTIC})
    score = score_path([step(("drug_targets_protein", ""), "A", "P")], policy, {})
    assert score.valid
    assert score.total_score == pytest.approx(2.0)
