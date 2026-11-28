# Portfolio summary (draft)

A concise, honest summary for a resume/portfolio context. Every claim
here is sourced from `README.md` / `V1_RELEASE_CHECKLIST.md` as of this
draft -- update it if either changes, rather than letting it drift.

**Do not use this before updating it** once R-GCN's validation search
finishes and/or the project is actually deployed -- several lines below
are explicitly scoped to "as of now" and would become false otherwise.

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

## Resume bullets (draft -- verify current state before using)

- Built an explainable drug-repurposing research pipeline over a
  4M-edge biomedical knowledge graph (PrimeKG), including a
  leakage-safe disease-held-out evaluation harness, a path-quality
  heuristic baseline, and a trained TransE knowledge-graph-embedding
  baseline evaluated once against a single locked test set.
- Designed and implemented a typed FastAPI + Pydantic service with a
  deterministic OpenAPI-schema-to-TypeScript-client pipeline, a
  precomputed-ranking serving contract decoupled from live model
  inference, and a React research workspace integrating four live
  external evidence/safety APIs (PubMed, ClinicalTrials.gov, openFDA,
  DailyMed) plus Open Targets.
- Enforced a strict experiment-provenance protocol across a
  multi-week model search (TransE validation search -> locked test;
  R-GCN validation search in progress on Kaggle), with an
  honest-by-construction API layer that reports model availability
  from read-only experiment records and refuses to report a model as
  usable without both an eligible record and an explicitly configured
  checkpoint.

## Tech stack

- **ML/graph**: Python, PyTorch, NetworkX, pandas -- PrimeKG adapter,
  path-finding, TransE/R-GCN embedding models, leakage-safe evaluation
  harness, case-study masking infrastructure.
- **API**: FastAPI, Pydantic, OpenAPI, deterministic schema export.
- **Frontend**: React, TypeScript, Vite, a generated typed API client
  (`openapi-typescript`).
- **External integrations**: PubMed E-utilities, ClinicalTrials.gov
  API, openFDA, DailyMed, Open Targets GraphQL API.
- **Deployment**: Render (backend), Vercel (frontend) -- config
  committed, not yet exercised against a live account as of this draft.

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
- A masked case-study harness (remove a known indication edge in both
  directions, confirm removal, then re-rank) for later qualitative
  validation of a trained model, built before any model exists to run
  it against.

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
