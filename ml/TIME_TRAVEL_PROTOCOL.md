# Time-travel test: pre-registered protocol

**Status: fixed on 2026-11-28, before any drug-disease pair was queried and before any ranking for this test was
generated.** The rules below are not changed after results exist. A change needs a new dated section with a reason, and
the original result is still reported.

## The question

PrimeKG has no per-edge dates, so the graph cannot be rebuilt as it was in an earlier year. The honest version is to use
the snapshot's release date as the cutoff and look forward from it:

> Among drug-disease pairs that the final R-GCN ranks in a disease's top 50 and that had **no prior evidence** at the
> PrimeKG release date, how often does *later* evidence appear, compared with a popularity baseline and a random baseline?

Beating random only shows the model finds well-studied drugs. Beating the popularity baseline is the real test.

## Fixed inputs

- **Cutoff: 2022-04-25.** The release time of the first PrimeKG version containing the `kg.csv` this project uses
  (`ml/provenance/primekg_provenance.json`). The graph's true content may be older; pairs studied between its build and
  this date count as "prior", which is conservative.
- **Window end ("as-of" date):** the date the collection run starts, recorded in the output. An interrupted run is
  resumed with the same date. The window is never extended to improve a number.
- **Model:** the final R-GCN checkpoint evaluated once on the locked test (record
  `ml/experiments/rgcn_final_locked_test_ca9313b1.json`). No retraining and no change to it.
- **Diseases:** 50 of the 100 locked-test diseases, drawn uniformly without replacement with
  `numpy.random.default_rng(42)` from the sorted list of `test_disease_ids` in the export's `metadata.json`.
  They are used because the model never saw their treatments. This analysis uses only disease IDs and names. It reads no
  locked-test label and no manifest, so it is not a second locked-test evaluation and is not used to tune anything.
- **Names:** PrimeKG's own `x_name`/`y_name` for the drug and disease, with no synonym expansion.

## The three rankers (top 50 per disease, drugs from the 7,957-drug universe)

1. **R-GCN:** all drugs scored for the disease by the same scoring code as `rgcn-evaluate`; the 50 highest.
2. **Popularity:** the 50 drugs with the highest total degree in the final training triples (head or tail occurrences),
   ties broken by drug ID. Same 50 drugs for every disease.
3. **Random:** 50 drugs drawn uniformly without replacement, with `numpy.random.default_rng([42, int(disease_id)])`.

Known indications are **not** removed from any list. Pairs with prior literature are removed by the filter below, for all
three rankers alike.

## Evidence queries (per pair)

- **PubMed** (E-utilities `esearch`, count only): term `"<drug>"[Title/Abstract] AND "<disease>"[Title/Abstract]`,
  `datetype=pdat`. *Prior* count: 1900/01/01 to 2022/04/25. *Later* count: 2022/04/26 to the as-of date.
- **ClinicalTrials.gov** (API v2): `query.intr=<drug>`, `query.cond=<disease>`, fields `NCTId` and `StartDate`, all pages.
  A study is *prior* if its start date is on or before 2022-04-25. Dates with month precision only are compared at month
  level and **April 2022 counts as prior**. A study with **no start date counts as prior** (unknown, so conservative).
  A study is *later* if its start date is after the cutoff by that rule.

## Definitions

- **New pair:** zero prior PubMed records **and** zero prior studies.
- **Hit:** a new pair with at least one *later* PubMed record **or** at least one *later* study.
- **Error:** a pair whose queries still fail after 3 retries. It is excluded for that pair only, from every ranker's
  counts, and the number of errors is reported. If more than 2% of pairs error, the failed pairs are retried once more.
  No disease is dropped.

## Outcomes and statistics

- **Primary:** pooled hit rate = hits / new pairs, over all 50 diseases, for each ranker. The number of new pairs per
  ranker is reported next to it, because the denominators differ.
- **Two pre-registered comparisons:** R-GCN minus popularity, and R-GCN minus random, on the primary outcome.
  Interval: disease-level paired bootstrap, resampling the 50 diseases with replacement (the three rankers see the same
  resampled diseases), 10,000 resamples, seed 42, 95% percentile interval. A resample where either ranker has zero new
  pairs is dropped and the number dropped is reported. State plainly whether each interval excludes zero. No
  multiple-comparison correction, since there are exactly two comparisons.
- **Secondary, reported but not used for the claim:** papers-only hit rate, trials-only hit rate, and the macro average of
  per-disease hit rates.

## How the result may be worded

- "R-GCN's new-pair hit rate is higher than popularity's" only if the difference is positive **and** its interval excludes
  zero. The same rule applies to random. Otherwise: "no evidence of a difference."
- A hit is a co-mention in a title or abstract, or a registered trial. It is **not** evidence that the drug works: the
  paper may be negative or incidental. No result of this test is described as a discovery or as a treatment suggestion.
- Name matching misses synonyms and misreads some names. That affects all three rankers alike but is not corrected.
- Literature volume grows over time, and popular drugs gain papers faster, so popularity is expected to be a strong
  baseline. The test reports whatever happens.

## Files

- `ml/src/repurposemap/evaluation/time_travel.py`: the classification and statistics, with tests.
- `ml/scripts/time_travel_rankings.py`: stage 1, needs the checkpoint and the export (run locally); writes the three
  top-50 lists for the 50 diseases.
- `ml/scripts/time_travel_collect.py`: stage 2, needs network; queries each unique pair once, resumable.
- `ml/scripts/time_travel_report.py`: stage 3, computes the tables and intervals.
- Results: `docs/TIME_TRAVEL_RESULTS.md` and `ml/results/time_travel/`.

## Amendment 1 (2026-11-28, before any result was analysed)

**Reason:** the first trial lookups returned HTTP 400 for drugs whose PrimeKG names contain brackets or commas (for
example `...-INDENO[1,2-.C.] PYRAZOLE`), because ClinicalTrials.gov parses them as query syntax. All four failures seen
were random-baseline drugs, so leaving them would have dropped pairs unevenly across the rankers.

**Change:** `query.intr` and `query.cond` are sent as quoted phrases (`"<drug>"`, `"<disease>"`) for **every** pair of
every ranker. PubMed already used quoted phrases. Nothing else changes. The trial cache was discarded and the trial
lookups restarted under the quoted form. No result had been looked at.
