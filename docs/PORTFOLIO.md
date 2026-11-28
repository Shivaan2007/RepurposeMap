# Portfolio summary

A concise, honest summary for a resume/portfolio context, as of 2026-11-28. Every number comes from
[`docs/FINAL_RESULTS.md`](FINAL_RESULTS.md) and `README.md`; if either changes, update this rather than letting it drift.
Live demo: https://repurposemap.vercel.app

## Project summary

RepurposeMap is an explainable drug-repurposing research tool: it
explores a 4M+ edge biomedical knowledge graph (PrimeKG) to surface and
explain candidate drugs for diseases with no approved treatment, with
every output framed as a research hypothesis for expert review, never a
clinical recommendation. The project's emphasis is on scientific
honesty under evaluation -- leakage-safe disease-held-out splits, a
single locked test set never touched during model development, and an
API/UI that report "not available" rather than fabricate a result when
no real ranking exists.

## Resume bullets

- Built an explainable drug-repurposing research pipeline over PrimeKG (8.1M edges): a leakage-safe,
  disease-held-out evaluation harness, path-quality and popularity baselines, TransE, and a sampled relational
  GNN (R-GCN) in PyTorch. Under a pre-registered protocol with one locked-test evaluation (100 held-out diseases,
  498 pairs), the R-GCN reached Hits@10 0.062 and MRR 0.033 against 0.016 and 0.010 for TransE.
- Designed a typed FastAPI + Pydantic service with an OpenAPI-to-TypeScript client pipeline, a precomputed-ranking
  serving contract, and a React research workspace over live PubMed, ClinicalTrials.gov, openFDA, DailyMed and
  Open Targets lookups; added CI and deployed it on Render and Vercel.
- Enforced the experiment protocol in code, not just in documentation: frozen model-selection rule, an
  exactly-once locked-test guard, checkpoint verification against the frozen spec, and provenance records;
  found and fixed integration bugs (a browser fetch error that made every request fail, and a
  ClinicalTrials.gov lookup that always failed) by testing the deployed system in a real browser.
- Ran 3 masked case studies (a known indication removed from the message-passing graph at inference; known drugs
  ranked 20, 151 and 190 of 7,957), reported as anecdotes, not validation.

## What this project does not show

- No claim that any drug works: outputs are research hypotheses, and the absolute accuracy is low (the true drug is
  in the top 10 for about 6 in 100 held-out pairs).
- No significance interval for R-GCN vs TransE (TransE's per-disease results were not saved), and the path
  baseline ranks a smaller candidate set, so no ranking against it is claimed.
- Rankings are precomputed for 5 diseases, not live, and carry scores but no explanation paths.
- Not built: the time-travel test against later literature, the TxGNN comparison, expert review, a triage shortlist.

## Tech stack

- **ML/graph**: Python, PyTorch, NetworkX, pandas -- PrimeKG adapter,
  path-finding, TransE/R-GCN embedding models, leakage-safe evaluation
  harness, case-study masking infrastructure.
- **API**: FastAPI, Pydantic, OpenAPI, deterministic schema export.
- **Frontend**: React, TypeScript, Vite, a generated typed API client
  (`openapi-typescript`).
- **External integrations**: PubMed E-utilities, ClinicalTrials.gov
  API, openFDA, DailyMed, Open Targets GraphQL API.
- **Deployment**: Render (backend, free tier) and Vercel (frontend), live; GitHub Actions CI (pytest and the
  frontend build).

## Research contribution

- A leakage-safe, disease-held-out train/validation/locked-test
  protocol with a documented candidate-universe caveat (path baseline's
  bounded candidate search vs. popularity/TransE/R-GCN's full
  7,957-drug universe), explicit about what "N times better" claims
  can and cannot mean across methods that rank different-sized
  candidate pools.
- A frozen, pre-registered R-GCN validation-search protocol (3 configs,
  fixed fanout/epochs/seed/candidate universe) run on Kaggle, with a
  documented, mechanical tie-break rule for selecting a winner --
  designed specifically to prevent post-hoc retuning after seeing
  results.
- A masked case-study harness (remove a known indication edge from the message-passing graph, confirm removal,
  re-encode, then re-rank), used for 3 case studies. Limitation stated everywhere it is cited: the weights were
  trained with that edge and not retrained, so this is weaker than a retrain-and-replay.

## Engineering contribution

- A provenance layer that answers "is a usable model actually
  available" from read-only experiment JSON records alone, requiring
  both a record's own self-reported eligibility AND an explicitly
  configured checkpoint path -- never inferred automatically from a
  record's own claims, and covered by tests asserting it reads no
  locked-test label and loads no graph/checkpoint.
- An exception-backstop pattern across every external-API-calling route
  (PubMed/ClinicalTrials.gov/openFDA/DailyMed/Open Targets): each typed
  exception is handled first, then a generic backstop guarantees no
  unhandled 500 ever reaches a caller.
- A precomputed-ranking serving contract (`repurposemap.serving`)
  separating the offline exporter from the online reader as two
  genuinely distinct code paths, so that wiring in a real trained
  model's output is a data-export step, not a code change.

## Results

The one locked-test result for the frozen R-GCN, next to the path, popularity and TransE baselines, is in [`docs/FINAL_RESULTS.md`](FINAL_RESULTS.md). Headline: the frozen sampled R-GCN (`rgcn-wide`), evaluated once on the 100-disease / 498-pair locked test, beat the final TransE baseline on both MRR (0.0334 vs. 0.0100) and Hits@10 (0.0622 vs. 0.0161) under the pre-registered decision rule, though absolute performance on the full 7,957-drug universe remains weak. Caveats stated there: a small test set (100 diseases, 498 pairs), a path baseline that ranks a different candidate set, and no claim of clinical efficacy.
