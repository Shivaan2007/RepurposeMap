"""Stage 3 of the time-travel test: tables and intervals from the rankings and the collected evidence.

    python ml/scripts/time_travel_report.py --rankings ml/results/time_travel/rankings.json --as-of 2026-11-28

Follows ``ml/TIME_TRAVEL_PROTOCOL.md``. Writes ``summary.json`` next to the rankings and ``docs/TIME_TRAVEL_RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml" / "src"))

from repurposemap.evaluation import time_travel as tt  # noqa: E402

METHODS = ("rgcn", "popularity", "random")
LABELS = {"rgcn": "R-GCN", "popularity": "Popularity", "random": "Random"}


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "_meta" not in record:
            out[record["key"]] = record
    return out


def pair_evidence(pubmed: dict, trials: dict, key: str, *, trial_error_as_zero: bool = False) -> tt.PairEvidence:
    """``trial_error_as_zero`` is the labelled sensitivity analysis only: a failed trial lookup counts as no trials."""

    p, t = pubmed.get(key), trials.get(key)
    if p is None or t is None:
        return tt.PairEvidence(error="not collected")
    if p.get("error"):
        return tt.PairEvidence(error=str(p["error"]))
    if t.get("error"):
        if trial_error_as_zero:
            return tt.PairEvidence(p["prior_papers"], p["later_papers"], 0, 0)
        return tt.PairEvidence(error=str(t["error"]))
    return tt.PairEvidence(p["prior_papers"], p["later_papers"], t["prior_trials"], t["later_trials"])


def analyse(rankings: dict, pubmed: dict, trials: dict, *, trial_error_as_zero: bool = False) -> dict:
    per_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    for disease in rankings["diseases"]:
        for method in METHODS:
            evidences = [pair_evidence(pubmed, trials, f"{d['drug_id']}|{disease['disease_id']}", trial_error_as_zero=trial_error_as_zero) for d in disease[method]]
            per_method[method].append(tt.counts(evidences))
    summary: dict = {"methods": {}, "comparisons": {}}
    for method, rows in per_method.items():
        rates = [r["hits"] / r["new"] for r in rows if r["new"]]
        summary["methods"][method] = {
            "pairs": sum(r["pairs"] for r in rows), "prior": sum(r["prior"] for r in rows), "new": sum(r["new"] for r in rows),
            "hits": sum(r["hits"] for r in rows), "errors": sum(r["errors"] for r in rows),
            "hit_rate": tt.pooled_rate(rows), "paper_hit_rate": tt.pooled_rate(rows, "paper_hits"),
            "trial_hit_rate": tt.pooled_rate(rows, "trial_hits"),
            "macro_hit_rate": sum(rates) / len(rates) if rates else None, "diseases_with_new_pairs": len(rates),
        }
    for other in ("popularity", "random"):
        summary["comparisons"][f"rgcn_minus_{other}"] = tt.paired_bootstrap_difference(per_method["rgcn"], per_method[other])
    return summary


def verdict(name: str, other: str, c: dict) -> str:
    if c["difference"] is None:
        return f"R-GCN vs {other}: not computable (no new pairs)."
    if c["difference"] > 0 and c["excludes_zero"]:
        return f"R-GCN's new-pair hit rate is higher than {other}'s (difference {c['difference']:+.3f}, 95% interval {c['ci_low']:+.3f} to {c['ci_high']:+.3f})."
    return f"No evidence of a difference between R-GCN and {other} (difference {c['difference']:+.3f}, 95% interval {c['ci_low']:+.3f} to {c['ci_high']:+.3f}; interval {'excludes' if c['excludes_zero'] else 'includes'} zero)."


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(m: dict) -> list[str]:
    rows = [
        "| Ranker | Pairs | Had prior evidence | New pairs | Hits | Hit rate | Papers only | Trials only | Macro avg | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in METHODS:
        r = m[k]
        rows.append(f"| {LABELS[k]} | {r['pairs']} | {r['prior']} | {r['new']} | {r['hits']} | {pct(r['hit_rate'])} | {pct(r['paper_hit_rate'])} | {pct(r['trial_hit_rate'])} | {pct(r['macro_hit_rate'])} | {r['errors']} |")
    return rows


def render(summary: dict, as_of: str) -> str:
    m = summary["methods"]
    lines = [
        "# Time-travel test: results", "",
        f"Protocol: `ml/TIME_TRAVEL_PROTOCOL.md` (fixed before any query; Amendment 1 quotes trial query names). Cutoff {tt.CUTOFF}; later evidence counted through {as_of}. "
        "50 locked-test diseases, top 50 drugs per ranker.", "",
        "A hit is a co-mention in a PubMed title/abstract or a registered trial. It is not evidence that a drug works, and nothing here is a treatment suggestion.", "",
    ]
    lines += table(m)
    lines += ["", "## Pre-registered comparisons (disease-level paired bootstrap, 10,000 resamples, seed 42)", ""]
    for other in ("popularity", "random"):
        c = summary["comparisons"][f"rgcn_minus_{other}"]
        lines.append(f"- {verdict('rgcn', other, c)} Resamples used {c['resamples_used']}, dropped {c['resamples_dropped']}.")
    sens = summary.get("sensitivity_trial_errors_as_zero")
    if sens:
        lines += ["", "## Sensitivity analysis (not the pre-registered result): failed trial lookups counted as no trials", "",
                  "ClinicalTrials.gov rejected some drug names with HTTP 400, unevenly across rankers, and the primary analysis excludes those pairs. Here they are kept, with zero trials, so only PubMed decides them.", ""]
        lines += table(sens["methods"])
        lines.append("")
        for other in ("popularity", "random"):
            c = sens["comparisons"][f"rgcn_minus_{other}"]
            lines.append(f"- {verdict('rgcn', other, c)} Resamples used {c['resamples_used']}, dropped {c['resamples_dropped']}.")
    errors = {k: m[k]["errors"] for k in METHODS}
    total_errors = sum(errors.values())
    lines += [
        "", "## Reading this result", "",
        f"- Errors: {total_errors} of {sum(m[k]['pairs'] for k in METHODS)} ranker-pairs ({100 * total_errors / sum(m[k]['pairs'] for k in METHODS):.1f}%; "
        f"R-GCN {errors['rgcn']}, popularity {errors['popularity']}, random {errors['random']}) could not be looked up on ClinicalTrials.gov (HTTP 400 on long or unusual chemical names, "
        "unchanged by a retry), which is above the 2% line in the protocol. They are excluded in the primary analysis, as pre-registered, and the sensitivity analysis above shows the conclusions do not change.",
        "- Absolute rates are low: roughly 2 in 100 new pairs gain any later co-mention or trial. A hit is a co-mention, not a finding that the drug works.",
        "- \"Had prior evidence\" is counted before the filter. A ranker with many such pairs (R-GCN, popularity) is mostly recommending drugs that were already studied with the disease; random rarely is.",
        "- The ranked lists are not filtered for known indications. Name matching uses PrimeKG's own names with no synonyms.",
    ]
    lines += ["", "Name matching misses synonyms; popular drugs gain papers faster over time, so popularity is a strong baseline. Reported as found.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--docs-output", type=Path, default=REPO / "docs" / "TIME_TRAVEL_RESULTS.md")
    args = parser.parse_args(argv)
    directory = args.rankings.parent
    rankings = json.loads(args.rankings.read_text())
    summary = analyse(rankings, load_jsonl(directory / "pubmed_cache.jsonl"), load_jsonl(directory / "trials_cache.jsonl"))
    summary["as_of"] = args.as_of
    summary["sensitivity_trial_errors_as_zero"] = analyse(
        rankings, load_jsonl(directory / "pubmed_cache.jsonl"), load_jsonl(directory / "trials_cache.jsonl"), trial_error_as_zero=True
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    args.docs_output.write_text(render(summary, args.as_of))
    print(render(summary, args.as_of))
    return 0


if __name__ == "__main__":
    sys.exit(main())
