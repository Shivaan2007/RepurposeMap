"""Resumable, checkpointed disease-based evaluation.

Uses the same tiny synthetic graph as test_disease_evaluation.py, imported
from there rather than duplicated, so a change to that fixture cannot make
the two test files disagree about what graph they are both checking against.
"""

from __future__ import annotations

import json

import pytest
from test_disease_evaluation import eval_graph

import repurposemap.evaluation.checkpoint as checkpoint_module
from repurposemap.evaluation import EvaluationConfig, evaluate_disease_split
from repurposemap.evaluation.checkpoint import read_checkpoint, run_with_checkpoint


def _config() -> EvaluationConfig:
    return EvaluationConfig(seed=0, test_fraction=1.0, max_test_diseases=4, max_candidates=50)


def _outcome_signature(outcomes):
    return sorted((o.disease_id, o.drug_id, o.path_rank, o.random_rank) for o in outcomes)


def test_run_with_checkpoint_matches_a_plain_run(tmp_path):
    graph = eval_graph()
    config = _config()
    plain = evaluate_disease_split(graph, config)
    checkpointed = run_with_checkpoint(graph, config, tmp_path / "ckpt.jsonl")
    assert _outcome_signature(checkpointed.outcomes) == _outcome_signature(plain.outcomes)


def test_checkpoint_file_is_valid_json_lines_one_per_disease(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    result = run_with_checkpoint(eval_graph(), _config(), path)

    lines = path.read_text().splitlines()
    assert len(lines) == result.n_test_diseases
    seen_diseases = set()
    for line in lines:
        record = json.loads(line)
        assert set(record) == {"disease_id", "outcomes"}
        seen_diseases.add(record["disease_id"])
        for outcome in record["outcomes"]:
            assert outcome["disease_id"] == record["disease_id"]
    assert seen_diseases == set(result.split.test_disease_ids)


def test_read_checkpoint_is_empty_for_a_missing_file(tmp_path):
    assert read_checkpoint(tmp_path / "does-not-exist.jsonl") == {}


def test_resuming_does_not_re_rank_diseases_already_checkpointed(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.jsonl"
    graph, config = eval_graph(), _config()
    run_with_checkpoint(graph, config, path)  # first run: every disease is new

    calls = []
    real_rank_one_disease = checkpoint_module.rank_one_disease

    def spy(*args, **kwargs):
        calls.append(args[1])  # disease_id is the second positional argument
        return real_rank_one_disease(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "rank_one_disease", spy)
    run_with_checkpoint(graph, config, path)  # second run: everything is already checkpointed

    assert calls == []


def test_resuming_from_a_truncated_checkpoint_only_ranks_the_missing_diseases(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.jsonl"
    graph, config = eval_graph(), _config()
    full = run_with_checkpoint(graph, config, path)

    # Simulate an interruption: keep only the first disease's line, as if the run had
    # stopped right after it and before any other disease was recorded.
    lines = path.read_text().splitlines()
    assert len(lines) > 1, "test needs more than one test disease to be meaningful"
    path.write_text(lines[0] + "\n")

    calls = []
    real_rank_one_disease = checkpoint_module.rank_one_disease

    def spy(*args, **kwargs):
        calls.append(args[1])
        return real_rank_one_disease(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "rank_one_disease", spy)
    resumed = run_with_checkpoint(graph, config, path)

    first_disease_id = json.loads(lines[0])["disease_id"]
    assert first_disease_id not in calls
    assert len(calls) == full.n_test_diseases - 1
    assert _outcome_signature(resumed.outcomes) == _outcome_signature(full.outcomes)


def test_on_disease_done_is_called_only_for_diseases_actually_ranked(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    graph, config = eval_graph(), _config()

    first_calls = []
    run_with_checkpoint(graph, config, path, on_disease_done=lambda disease_id, outcomes: first_calls.append(disease_id))
    assert len(first_calls) == len(set(first_calls)) > 0

    second_calls = []
    run_with_checkpoint(graph, config, path, on_disease_done=lambda disease_id, outcomes: second_calls.append(disease_id))
    assert second_calls == []


def test_run_with_checkpoint_rejects_a_graph_without_indications():
    from graph_builders import primekg_style_graph

    graph = primekg_style_graph({"A": "drug", "D": "disease"}, [("drug_protein", "target", "A", "D")])
    with pytest.raises(ValueError):
        run_with_checkpoint(graph, EvaluationConfig(), "unused-path.jsonl")
