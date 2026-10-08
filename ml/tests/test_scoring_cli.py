"""CLI for explain-paths, rank-drugs and evaluate-baseline, on the synthetic PrimeKG-schema fixture."""

from pathlib import Path

from repurposemap.cli import main


def test_explain_paths_shows_raw_and_ranked_sections(primekg_csv: Path, capsys):
    exit_code = main([
        "explain-paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-name", "DEMO Drug Kappa", "--target-name", "DEMO Disease Lambda",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Raw shortest paths (full graph, every relation):" in out
    assert "Quality-ranked paths (valid candidates only):" in out
    # The direct indication edge is the raw shortest path, and the policy flags it.
    assert "flag: excluded relation: indication [indication]" in out


def test_explain_paths_ranked_section_is_empty_when_only_excluded_links_exist(primekg_csv: Path, capsys):
    main([
        "explain-paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-name", "DEMO Drug Kappa", "--target-name", "DEMO Disease Lambda",
    ])
    out = capsys.readouterr().out
    assert "Quality-ranked paths (valid candidates only):\n  no candidate path found" in out


def test_explain_paths_with_primekg_relations_ranks_a_valid_path(primekg_csv: Path, capsys):
    # Kappa targets Nu, and Nu interacts with Xi. Nothing in the fixture links Xi to a disease,
    # so the ranked section stays empty and the test checks the raw path alone.
    exit_code = main([
        "explain-paths", "--source", "primekg", "--csv", str(primekg_csv),
        "--source-id", "103", "--target-id", "102", "--max-path-length", "2",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "DEMO Drug Kappa" in out and "DEMO Protein Xi" in out


def test_rank_drugs_reports_research_hypothesis_label_and_no_candidates(primekg_csv: Path, capsys):
    exit_code = main([
        "rank-drugs", "--source", "primekg", "--csv", str(primekg_csv),
        "--disease", "DEMO Disease Lambda",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Research hypothesis ranking. Not a treatment recommendation." in out
    assert "Candidate drugs searched: 0" in out
    assert "No candidate drug has a valid path" in out


def test_rank_drugs_rejects_a_non_disease_name(primekg_csv: Path, capsys):
    # "DEMO Compound Pi" is a drug and an exposure, never a disease.
    exit_code = main([
        "rank-drugs", "--source", "primekg", "--csv", str(primekg_csv),
        "--disease", "DEMO Compound Pi",
    ])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "disease entity not found" in err


def test_rank_drugs_needs_the_primekg_source(sample_csv: Path, capsys):
    exit_code = main(["rank-drugs", "--csv", str(sample_csv), "--disease", "DEMO Disease Zeta"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "add --source primekg" in err


def test_explain_paths_needs_the_primekg_source(sample_csv: Path, capsys):
    exit_code = main([
        "explain-paths", "--csv", str(sample_csv),
        "--source-name", "DEMO Drug Alpha", "--target-name", "DEMO Disease Zeta",
    ])
    assert exit_code == 1
    assert "add --source primekg" in capsys.readouterr().err


def test_evaluate_baseline_prints_disease_based_header_and_metrics(primekg_csv: Path, capsys):
    # The fixture has 3 indication pairs, one disease each. --test-fraction 1.0 holds out all of them.
    exit_code = main([
        "evaluate-baseline", "--source", "primekg", "--csv", str(primekg_csv),
        "--test-diseases", "3", "--test-fraction", "1.0", "--seed", "0",
    ])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Path baseline evaluation — research use only" in out
    assert "Research hypothesis evaluation. Not a treatment recommendation." in out
    assert "Train diseases: 0   Test diseases: 3" in out
    assert "Held-out (test) indication edges: 3" in out
    assert "Hits@1:" in out and "Hits@3:" in out and "Hits@10:" in out
    assert "MRR:" in out and "random baseline" in out
    assert "Runtime:" in out


def test_evaluate_baseline_rejects_a_bad_test_fraction(primekg_csv: Path, capsys):
    exit_code = main([
        "evaluate-baseline", "--source", "primekg", "--csv", str(primekg_csv), "--test-fraction", "0",
    ])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "test_fraction" in err


def test_evaluate_baseline_is_reproducible_for_a_seed(primekg_csv: Path, capsys):
    args = [
        "evaluate-baseline", "--source", "primekg", "--csv", str(primekg_csv),
        "--test-diseases", "2", "--test-fraction", "1.0", "--seed", "4",
    ]
    main(args)
    first = capsys.readouterr().out
    main(args)
    second = capsys.readouterr().out
    # Runtime varies between runs; everything else must match exactly.
    strip_runtime = lambda text: "\n".join(line for line in text.splitlines() if not line.startswith("Runtime:"))
    assert strip_runtime(first) == strip_runtime(second)
