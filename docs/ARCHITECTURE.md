# Architecture

```
Python research engine (ml/)  ->  FastAPI + Pydantic (ml/src/repurposemap/api/)
    ->  OpenAPI schema (openapi.json)
    ->  generated TypeScript client (web/client)
    ->  React research workspace (web/app)
```

See `API.md` for the endpoint-level contract. This document covers the
pieces around it: what loads when, what's on disk vs. in a live request,
and where each layer's boundary is.

## Layers

**Graph + research engine** (`ml/src/repurposemap/{graph,embeddings,evaluation,
scoring,ranking}`). Loads PrimeKG into memory (`graph/loader.py`), computes
path/TransE/R-GCN scores, runs the offline evaluation harnesses (disease-held-out
splits, leakage checks, case-study replay). This code runs offline -- on a
laptop or a Kaggle/Colab GPU session -- and produces artifacts under
`ml/experiments/*.json`, never inside a live API request.

**Provenance** (`ml/src/repurposemap/provenance`). Reads `ml/experiments/*.json`
to answer "is a real model available right now?" honestly. This is the only
thing that connects the offline research engine to the live API today --
`GET /api/v1/system/info` calls it to report `model.available` /
`inference_eligible`, and it currently always resolves to `False` because no
final-training experiment record + configured checkpoint exists yet.

**Live API** (`ml/src/repurposemap/api`, served by FastAPI/uvicorn). Five real
routes (`/health`, `/api/v1/system/info`, `/api/v1/evidence`,
`/api/v1/disease-biology`, `/api/v1/target-scorecard`) plus a stubbed
`/api/v1/rank`. **None of the five real routes ever loads PrimeKG or computes
its file fingerprint** (`ml/tests/test_api_no_graph_load.py` asserts this for
health/system-info directly; evidence/biology/scorecard call external APIs,
not the graph). This matters for deployment: the live service today has no
4M-edge graph to hold in memory and no multi-GB data file to ship with the
container -- see `docs/DEPLOYMENT.md`.

**Evidence/safety domain** (`ml/src/repurposemap/evidence`). Four live
adapters (PubMed, ClinicalTrials.gov, openFDA, DailyMed), each raising its own
typed error rather than returning an empty/fabricated result. `/api/v1/evidence`
queries all four synchronously, on the request thread -- see API.md for why,
and the plan to move this to a background job once that infrastructure exists.

**Disease biology domain** (`ml/src/repurposemap/biology`). One live adapter
(Open Targets), backing researcher-workflow steps 1 and 2 (`/api/v1/disease-biology`,
`/api/v1/target-scorecard`). Also synchronous, same reason.

**Frontend** (`web/app`). A Vite/React app, one panel per workflow step, each
a thin synchronous caller of its own endpoint via the generated
`RepurposeMapClient` (`web/client`). Panels share the disease name typed into
the main query (see `App.tsx`'s `sharedDiseaseName`) so a researcher doesn't
retype it per panel, but each panel's field stays independently editable --
there is deliberately no forced single source of truth across panels, since a
researcher may want to look up evidence for a different disease than the one
they ranked.

## What "ranking" means today vs. later

`POST /api/v1/rank` is real wire-format (the full `RankingResponse` schema --
candidates, scores, explanation paths, safety flags -- is part of the
committed OpenAPI schema), but its body always returns `model_status:
"not_available"` with zero candidates. This is intentional: PLAN.md's
Hypothesis-generation step needs a trained R-GCN (or TransE) checkpoint, which
does not exist yet (validation search in progress -- see `ml/RGCN_RUNBOOK.md`
and `V1_RELEASE_CHECKLIST.md`). Wiring a real model in means replacing this
one function's body; the contract, frontend rendering (`CandidateList.tsx`),
and `RepurposingHypothesis` assembly point are already shaped for it.

**Offline research/evaluation vs. online serving/demo:** these are the same
thing today, because there is nothing yet to separate -- every committed
experiment artifact (`ml/experiments/*.json`) holds aggregate metrics only,
never a per-disease ranked candidate list, so there is no real ranking data
to serve either live or precomputed. Once a model is trained and
locked-test-evaluated, `docs/DEPLOYMENT.md`'s "Serving strategy" section lays
out the two options (a precomputed per-disease bundle, or live inference) and
why the precomputed bundle is the right default for a cheap V1 host -- but
until then, this split is a documented plan, not two actually-different
code paths.

## Directory map

```
ml/src/repurposemap/
├── adapters/        # PrimeKG CSV -> graph loader, file fingerprinting
├── api/              # FastAPI app, routers, Pydantic schemas (the live service)
├── biology/          # Open Targets domain (disease biology, target scorecard)
├── cli.py            # `repurposemap` command-line entry point
├── embeddings/        # TransE / R-GCN model + training code (offline, GPU)
├── evaluation/        # disease-held-out splits, metrics, baselines, case studies
├── evidence/          # PubMed / ClinicalTrials.gov / openFDA / DailyMed adapters
├── graph/             # in-memory graph, search, path-finding, stats
├── hypothesis.py      # future researcher-facing assembly of score+evidence+safety
├── provenance/        # experiment-record-backed "is a model really available" service
├── ranking/           # path-count drug ranking (used by the path baseline, not the API)
└── scoring/           # path-quality / hub / promiscuity heuristics

web/
├── app/               # React research workspace (what a researcher opens)
└── client/            # generated TypeScript client from openapi.json
```
