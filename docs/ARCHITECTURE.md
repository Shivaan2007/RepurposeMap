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

**Live API** (`ml/src/repurposemap/api`, served by FastAPI/uvicorn). Seven
real routes: `/health`, `/api/v1/system/info`, `/api/v1/evidence`,
`/api/v1/disease-biology`, `/api/v1/target-scorecard`, `/api/v1/rank`, and
`/api/v1/rank/diseases`. **None of these ever loads PrimeKG or computes its
file fingerprint** (`ml/tests/test_api_no_graph_load.py` asserts this for
health/system-info directly; evidence/biology/scorecard call external APIs,
not the graph; rank/rank-diseases read only `data/serving/*.json`). This
matters for deployment: the live service today has no 4M-edge graph to hold
in memory and no multi-GB data file to ship with the container -- see
`docs/DEPLOYMENT.md`.

**Serving** (`ml/src/repurposemap/serving`). The mechanism that lets
`/api/v1/rank` answer with a real precomputed ranking instead of a
hardcoded stub, once one exists. `schemas.py` defines
`CandidateRankingBundle` (the on-disk contract); `loader.py` reads and
validates bundles from `data/serving/*.json`; `export_rankings.py` is the
offline CLI that builds one from an already-ranked, already-approved
candidate list. This package never computes or fabricates a ranking --
`data/serving/` is empty today (only a `.gitkeep`), because no committed
experiment artifact anywhere holds a real per-disease ranked candidate
list yet (see "What 'ranking' means today vs. later" below).

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

`POST /api/v1/rank` is real wire-format end-to-end (the full
`RankingResponse` schema -- candidates, scores, explanation paths, safety
flags -- is part of the committed OpenAPI schema) and is now backed by a
real serving mechanism (`repurposemap.serving`, see above), not a hardcoded
stub. But its body still always returns `model_status: "not_available"`
with zero candidates in practice, because `data/serving/` is empty: no
committed experiment artifact anywhere holds a per-disease ranked candidate
list to export from yet. This is intentional: PLAN.md's Hypothesis-generation
step needs a trained R-GCN (or TransE) checkpoint, which does not exist yet
(validation search in progress -- see `ml/RGCN_RUNBOOK.md` and
`V1_RELEASE_CHECKLIST.md`). Wiring a real model in from here is a *data*
step, not a *code* step: run `repurposemap.serving.export_rankings` against
the model's approved ranking output for each deployed demo disease. The
frontend rendering (`CandidateList.tsx`, `QueryForm.tsx`) and
`RepurposingHypothesis` assembly point are already shaped to consume a real
bundle once one lands.

**Offline research/evaluation vs. online serving/demo:** the serving
*mechanism* now exists as two genuinely separate code paths -- the offline
exporter (`export_rankings.py`) and the online reader (`loader.py` +
`api/ranking.py`) -- but there is still nothing real flowing through either
path, because every committed experiment artifact (`ml/experiments/*.json`)
holds aggregate metrics only, never a per-disease ranked candidate list.
Once a model is trained and locked-test-evaluated, the exporter is the tool
that turns its approved output into something `/api/v1/rank` can actually
serve for a chosen set of demo diseases -- see `docs/DEPLOYMENT.md`'s
"Serving strategy" section for why a precomputed bundle is the right default
for a cheap V1 host over live inference.

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
├── scoring/           # path-quality / hub / promiscuity heuristics
└── serving/           # CandidateRankingBundle schema, loader, offline exporter -- see above

data/
├── raw/               # large raw datasets (PrimeKG), gitignored, never committed
├── sample/            # small committed samples for tests/dev
└── serving/           # precomputed CandidateRankingBundle files, one per disease (empty today)

web/
├── app/               # React research workspace (what a researcher opens)
└── client/            # generated TypeScript client from openapi.json
```
