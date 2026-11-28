# Time-travel test: results

Protocol: `ml/TIME_TRAVEL_PROTOCOL.md` (fixed before any query; Amendment 1 quotes trial query names). Cutoff 2022-04-25; later evidence counted through 2026-11-28. 50 locked-test diseases, top 50 drugs per ranker.

A hit is a co-mention in a PubMed title/abstract or a registered trial. It is not evidence that a drug works, and nothing here is a treatment suggestion.

| Ranker | Pairs | Had prior evidence | New pairs | Hits | Hit rate | Papers only | Trials only | Macro avg | Errors |
|---|---|---|---|---|---|---|---|---|---|
| R-GCN | 2500 | 437 | 2003 | 38 | 1.9% | 1.6% | 0.3% | 3.1% | 60 |
| Popularity | 2500 | 351 | 2099 | 27 | 1.3% | 1.1% | 0.3% | 1.7% | 50 |
| Random | 2500 | 70 | 2249 | 12 | 0.5% | 0.3% | 0.2% | 0.6% | 181 |

## Pre-registered comparisons (disease-level paired bootstrap, 10,000 resamples, seed 42)

- No evidence of a difference between R-GCN and popularity (difference +0.006, 95% interval -0.003 to +0.016; interval includes zero). Resamples used 10000, dropped 0.
- R-GCN's new-pair hit rate is higher than random's (difference +0.014, 95% interval +0.005 to +0.024). Resamples used 10000, dropped 0.

## Sensitivity analysis (not the pre-registered result): failed trial lookups counted as no trials

ClinicalTrials.gov rejected some drug names with HTTP 400, unevenly across rankers, and the primary analysis excludes those pairs. Here they are kept, with zero trials, so only PubMed decides them.

| Ranker | Pairs | Had prior evidence | New pairs | Hits | Hit rate | Papers only | Trials only | Macro avg | Errors |
|---|---|---|---|---|---|---|---|---|---|
| R-GCN | 2500 | 437 | 2063 | 38 | 1.8% | 1.6% | 0.3% | 3.0% | 0 |
| Popularity | 2500 | 351 | 2149 | 27 | 1.3% | 1.1% | 0.3% | 1.6% | 0 |
| Random | 2500 | 70 | 2430 | 12 | 0.5% | 0.3% | 0.2% | 0.5% | 0 |

- No evidence of a difference between R-GCN and popularity (difference +0.006, 95% interval -0.003 to +0.015; interval includes zero). Resamples used 10000, dropped 0.
- R-GCN's new-pair hit rate is higher than random's (difference +0.013, 95% interval +0.005 to +0.023). Resamples used 10000, dropped 0.

## Reading this result

- Errors: 291 of 7500 ranker-pairs (3.9%; R-GCN 60, popularity 50, random 181) could not be looked up on ClinicalTrials.gov (HTTP 400 on long or unusual chemical names, unchanged by a retry), which is above the 2% line in the protocol. They are excluded in the primary analysis, as pre-registered, and the sensitivity analysis above shows the conclusions do not change.
- Absolute rates are low: roughly 2 in 100 new pairs gain any later co-mention or trial. A hit is a co-mention, not a finding that the drug works.
- "Had prior evidence" is counted before the filter. A ranker with many such pairs (R-GCN, popularity) is mostly recommending drugs that were already studied with the disease; random rarely is.
- The ranked lists are not filtered for known indications. Name matching uses PrimeKG's own names with no synonyms.

Name matching misses synonyms; popular drugs gain papers faster over time, so popularity is a strong baseline. Reported as found.
