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
