import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collect = _load("time_travel_collect")
report = _load("time_travel_report")
rankings_mod = _load("time_travel_rankings")


def _rankings():
    entry = lambda d: {"drug_id": d, "drug_name": f"drug{d}"}
    return {"diseases": [{"disease_id": "9", "disease_name": "dis", "rgcn": [entry("1"), entry("2")], "popularity": [entry("2"), entry("3")], "random": [entry("4")]}]}


def test_unique_pairs_dedupes_across_rankers():
    assert [p["key"] for p in collect.unique_pairs(_rankings())] == ["1|9", "2|9", "3|9", "4|9"]


def test_collect_resumes_and_retries_errors(tmp_path):
    path, pairs, calls, state = tmp_path / "c.jsonl", collect.unique_pairs(_rankings()), [], {"fail": True}

    def query(pair):
        calls.append(pair["key"])
        if pair["key"] == "2|9" and state["fail"]:
            raise collect.PubMedError("boom")
        return {"prior_papers": 0, "later_papers": 1}

    first = collect.collect(pairs, path, "2026-11-28", query, sleep=lambda s: None, log=lambda m: None)
    assert first["2|9"]["error"] and first["1|9"]["later_papers"] == 1
    calls.clear()
    state["fail"] = False
    second = collect.collect(pairs, path, "2026-11-28", query, sleep=lambda s: None, log=lambda m: None)
    assert calls == ["2|9"] and "error" not in second["2|9"]


def test_collect_refuses_other_as_of(tmp_path):
    path = tmp_path / "c.jsonl"
    collect.collect([], path, "2026-11-28", lambda p: {}, log=lambda m: None)
    with pytest.raises(SystemExit):
        collect.collect([], path, "2027-01-01", lambda p: {}, log=lambda m: None)


def test_report_classifies_and_verdict():
    r = _rankings()
    pubmed = {f"{d}|9": {"key": f"{d}|9", "prior_papers": 0, "later_papers": 1 if d in "12" else 0} for d in "1234"}
    pubmed["3|9"]["prior_papers"] = 5
    trials = {f"{d}|9": {"key": f"{d}|9", "prior_trials": 0, "later_trials": 0} for d in "1234"}
    s = report.analyse(r, pubmed, trials)
    assert s["methods"]["rgcn"]["new"] == 2 and s["methods"]["rgcn"]["hits"] == 2
    assert s["methods"]["popularity"]["prior"] == 1 and s["methods"]["popularity"]["hits"] == 1
    assert "No evidence" in report.verdict("rgcn", "popularity", s["comparisons"]["rgcn_minus_popularity"])
    assert "R-GCN" in report.render(s, "2026-11-28")


def test_missing_collection_is_error():
    assert report.pair_evidence({}, {}, "x").error


def test_popularity_degrees_counts_head_or_tail():
    class Idx:
        id_to_index = {"a": 0, "b": 1}

    out = rankings_mod.popularity_degrees([(0, 0, 5), (5, 0, 0), (1, 0, 5)], Idx, ["a", "b"])
    assert out == {"a": 2, "b": 1}
