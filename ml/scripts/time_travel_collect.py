"""Stage 2 of the time-travel test: query PubMed and ClinicalTrials.gov for every unique drug-disease pair.

    python ml/scripts/time_travel_collect.py --rankings ml/results/time_travel/rankings.json --as-of 2026-11-28

Follows ``ml/TIME_TRAVEL_PROTOCOL.md``. Each unique pair is queried once per source and every answer is appended to a
cache file (``pubmed_cache.jsonl`` and ``trials_cache.jsonl`` next to the rankings), so an interrupted run resumes where it
stopped. The as-of date is stored in each cache and a resume under a different date is refused: the window is never
changed to improve a number. A pair whose query still fails after 3 retries is cached as an error and retried by the next
run. PubMed and ClinicalTrials.gov are queried by two threads at once, each with its own rate limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml" / "src"))

from repurposemap.evaluation import time_travel as tt  # noqa: E402
from repurposemap.evidence.clinicaltrials import ClinicalTrialsAdapter, ClinicalTrialsError, ClinicalTrialsQuery  # noqa: E402
from repurposemap.evidence.pubmed import PubMedError, PubMedEUtilsAdapter, PubMedQuery  # noqa: E402

PRIOR_RANGE = ("1900/01/01", "2022/04/25")
LATER_START = "2022/04/26"
RETRIES = 3


def unique_pairs(rankings: dict) -> list[dict]:
    """One entry per unique (drug, disease) across the three rankers, in a stable order."""

    seen: dict[str, dict] = {}
    for disease in rankings["diseases"]:
        for method in ("rgcn", "popularity", "random"):
            for drug in disease[method]:
                key = pair_key(drug["drug_id"], disease["disease_id"])
                seen.setdefault(key, {"key": key, "drug": drug["drug_name"], "disease": disease["disease_name"]})
    return list(seen.values())


def pair_key(drug_id: str, disease_id: str) -> str:
    return f"{drug_id}|{disease_id}"


def load_cache(path: Path, as_of: str) -> dict[str, dict]:
    """Cached answers keyed by pair. Refuses a cache written under a different as-of date."""

    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "_meta" in record:
            if record["_meta"]["as_of"] != as_of:
                raise SystemExit(f"{path.name} was collected as of {record['_meta']['as_of']}, not {as_of}: the window is fixed, so use that date")
            continue
        cache[record["key"]] = record
    return cache


def _with_retries(call: Callable[[], dict], sleep: Callable[[float], None]) -> dict:
    last = "unknown"
    for attempt in range(RETRIES + 1):
        try:
            return call()
        except (PubMedError, ClinicalTrialsError) as error:
            last = f"{type(error).__name__}: {error}"
            retry_after = getattr(error, "retry_after", None)
            if attempt < RETRIES:
                sleep(float(retry_after) if retry_after and str(retry_after).isdigit() else 2.0 * 2**attempt)
    return {"error": last}


def collect(
    pairs: Iterable[dict], cache_path: Path, as_of: str, query: Callable[[dict], dict], *,
    sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print, label: str = "",
) -> dict[str, dict]:
    """Query each pair not already answered without error, appending every answer to the cache as it arrives."""

    cache = load_cache(cache_path, as_of)
    todo = [p for p in pairs if p["key"] not in cache or cache[p["key"]].get("error")]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.write_text(json.dumps({"_meta": {"as_of": as_of}}) + "\n")
    log(f"[{label}] {len(todo)} to query, {len(cache) - sum(1 for p in todo if p['key'] in cache)} already cached")
    with cache_path.open("a") as handle:
        for index, pair in enumerate(todo, start=1):
            record = {"key": pair["key"], "drug": pair["drug"], "disease": pair["disease"], **_with_retries(lambda: query(pair), sleep)}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            cache[pair["key"]] = record
            if index % 100 == 0:
                log(f"[{label}] {index}/{len(todo)}")
    return cache


def make_pubmed_query(adapter: PubMedEUtilsAdapter, as_of: str) -> Callable[[dict], dict]:
    later_end = as_of.replace("-", "/")

    def query(pair: dict) -> dict:
        q = PubMedQuery(drug_name=pair["drug"], disease_name=pair["disease"])
        return {
            "prior_papers": adapter.count_in_range(q, mindate=PRIOR_RANGE[0], maxdate=PRIOR_RANGE[1]),
            "later_papers": adapter.count_in_range(q, mindate=LATER_START, maxdate=later_end),
        }

    return query


def quoted(name: str) -> str:
    """Send a name to ClinicalTrials.gov as one quoted phrase. Unquoted, brackets and commas in chemical names are
    parsed as query syntax and the API answers HTTP 400 (see the 2026-11-28 amendment in the protocol)."""

    return '"' + name.replace('"', " ") + '"'


def make_trials_query(adapter: ClinicalTrialsAdapter) -> Callable[[dict], dict]:
    def query(pair: dict) -> dict:
        dates = adapter.start_dates(ClinicalTrialsQuery(drug_name=quoted(pair["drug"]), disease_name=quoted(pair["disease"])))
        prior = sum(1 for d in dates if tt.trial_is_prior(d))
        return {"prior_trials": prior, "later_trials": len(dates) - prior}

    return query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD; the window end, fixed for the whole run")
    parser.add_argument("--email", default=None, help="sent to NCBI with each request, as they ask")
    args = parser.parse_args(argv)

    rankings = json.loads(args.rankings.read_text())
    pairs = unique_pairs(rankings)
    print(f"{len(pairs)} unique pairs from {len(rankings['diseases'])} diseases; cutoff {tt.CUTOFF}, as of {args.as_of}")
    directory = args.rankings.parent
    pubmed = PubMedEUtilsAdapter(email=args.email, timeout_s=30.0)
    trials = ClinicalTrialsAdapter(timeout_s=60.0)
    pubmed_query, trials_query = make_pubmed_query(pubmed, args.as_of), make_trials_query(trials)

    def slow_trials_query(pair: dict) -> dict:
        time.sleep(0.4)  # one request per 0.4 s at most; ClinicalTrials.gov asks users to be gentle
        return trials_query(pair)

    threads = [
        threading.Thread(target=collect, args=(pairs, directory / "pubmed_cache.jsonl", args.as_of, pubmed_query), kwargs={"label": "pubmed"}),
        threading.Thread(target=collect, args=(pairs, directory / "trials_cache.jsonl", args.as_of, slow_trials_query), kwargs={"label": "trials"}),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
