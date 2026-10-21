# RepurposeMap API

The boundary between the Python research engine and the future TypeScript
research workspace:

```
Python research engine -> FastAPI + Pydantic -> OpenAPI
    -> generated TypeScript client -> future React research workspace
```

Two real endpoints exist today (`/health`, `/api/v1/system/info`), plus a
stubbed `POST /api/v1/rank` that always reports `model_status:
"not_available"`. There is no trained model wired in yet -- see "Plugging
in a real model" below.

## Running the API locally

```bash
pip install -e ".[api]"
uvicorn repurposemap.api.app:app --reload
```

Then `curl localhost:8000/health` or open `localhost:8000/docs` for the
interactive Swagger UI FastAPI builds automatically from the same schema.

## Exporting the OpenAPI schema

```bash
PYTHONPATH=ml/src python ml/scripts/export_openapi.py
```

Writes `openapi.json` at the repo root. This never starts a server or
loads the PrimeKG graph -- `app.openapi()` builds the schema purely from
the route and Pydantic model definitions in `ml/src/repurposemap/api/`.

## Regenerating the TypeScript client

```bash
cd web/client
npm install
npm run generate   # openapi-typescript ../../openapi.json -> src/types.ts
npm run typecheck
```

`src/client.ts` and `src/index.ts` are hand-written and don't need
regenerating; only `src/types.ts` is meant to be replaced by the generator.
Re-export `openapi.json` first if you've changed anything in
`ml/src/repurposemap/api/schemas.py`.

## Plugging in a real model

`GET /api/v1/system/info` and `POST /api/v1/rank` both report model
availability honestly rather than guessing. Wiring in a real model means:

1. Implementing a scoring function with the same shape the disease-based
   evaluation harness already uses (`repurposemap.evaluation.disease_eval`
   for the path baseline, `repurposemap.evaluation.transe_baseline` for a
   trained TransE checkpoint).
2. Updating `ModelStatus` in `system_info.py` to reflect the real model
   that's loaded.
3. Replacing `rank_candidates`'s body in `ranking.py` with an actual call
   into that scoring function, filling in `CandidateDrug`/`CandidateScore`/
   `ExplanationPath` from its real output -- the contract in `schemas.py`
   is already shaped for either baseline.

Until then, `rank_candidates` stays a stub. It must keep returning
`model_status: "not_available"` with an empty candidate list rather than a
mock or random ranking.

Model availability is reported honestly end-to-end: `GET
/api/v1/system/info` calls `repurposemap.provenance.load_experiments()`
to read `ml/experiments/*.json` (not PrimeKG, not a checkpoint) and
reports `inference_eligible`/`model.available` as `False` until both an
eligible final-training experiment record exists and a checkpoint path
is explicitly configured in `repurposemap.provenance.service` -- see
that module's own docstring.

## Running the research workspace (web/app)

```bash
cd web/app
npm install
npm run dev     # proxies /api and /health to localhost:8000 (see vite.config.ts)
```

It imports `RepurposeMapClient` from the generated client in `web/client`
(a `file:../client` dependency) rather than hand-writing fetch calls or
API types. Regenerate `web/client/src/types.ts` first if you've changed
`schemas.py` -- the workspace picks up the new types on its next
`npm install`/build without any changes of its own.

## Evidence/safety domain foundation (not wired into the API yet)

`repurposemap.evidence` defines typed contracts (`EvidenceSource`,
`EvidenceRecord`, `PubMedEvidence`, `ClinicalTrialEvidence`,
`SafetyEvidence`, `EvidenceBundle`) and `Protocol`-based adapter
interfaces for PubMed, ClinicalTrials.gov, DailyMed and openFDA. No
adapter performs a live query yet -- `NotImplementedAdapter.search()`
raises rather than returning an empty or fabricated result.
`repurposemap.hypothesis.RepurposingHypothesis` shows how a model score,
explanation paths, an evidence bundle, safety flags and provenance will
assemble into one researcher-facing object once both exist.
