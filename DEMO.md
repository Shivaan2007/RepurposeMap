# Running a local RepurposeMap demo

A reproducible way to run the current V1 backend and frontend together,
plus an honest audit of what would block a real deployment. This
describes the system as it exists today: no trained ranking model, four
live evidence/biology sources, and local-only researcher-workflow state.

## Prerequisites

- Python 3.12+ and `pip install -e ".[api]"` from `ml/` (installs FastAPI,
  Pydantic, and this package).
- Node.js 20+ and `npm install` in both `web/client` and `web/app`.
- No API keys or secrets are required to run the demo -- see "Secrets and
  environment variables" below.
- No PrimeKG download, no GPU, no Kaggle account. The ranking endpoint is
  a stub (see "Known limitations"), so none of the real graph/model data
  this repo also contains is needed to run the demo.

## Backend

```bash
cd ml
pip install -e ".[api]"
uvicorn repurposemap.api.app:app --reload
```

Runs on **port 8000**. Check it with `curl localhost:8000/health` or open
`localhost:8000/docs` for the interactive Swagger UI.

## Frontend

In a second terminal:

```bash
cd web/client && npm install && npm run build
cd ../app && npm install
npm run dev
```

Runs on **port 5173** (Vite's default). `web/app/vite.config.ts` proxies
`/api` and `/health` to `http://localhost:8000`, so the backend must
already be running on port 8000 for the frontend to work.

## What you can actually do in the demo

- **Evidence lookup**: enter a real drug and disease/condition name --
  this makes live calls to PubMed, ClinicalTrials.gov, openFDA and
  DailyMed and shows real literature, trial and safety records (or an
  honest per-source failure/empty state).
- **Disease biology** / **Target scorecard**: enter a real disease name or
  gene symbol -- live calls to the Open Targets Platform API.
- **Candidate comparison**: compare 2-4 real drug names for one disease
  using the same live evidence lookups, side by side.
- **Hypothesis notebook**: track drug/disease ideas locally (this
  browser's `localStorage` only -- never sent to the server, lost if site
  data is cleared, not shared across devices).
- **Ranking** (the top query form): always returns `model_status:
  "not_available"` with an empty candidate list. This is correct, current
  behavior -- see "Known limitations".

## Known limitations

- **No model is available yet.** `POST /api/v1/rank` is an intentional
  stub until a real R-GCN (or TransE) model is wired in -- see API.md's
  "Plugging in a real model" section. It never fabricates a ranking.
- **The evidence/biology endpoints are synchronous.** Each request calls
  up to four external APIs in-process on the request thread, so a slow or
  rate-limited source adds directly to response latency. PLAN.md calls
  for moving this to a background job (Redis + RQ) once that
  infrastructure exists in this repo -- it does not yet.
- **The hypothesis notebook has no server-side persistence.** It is a
  deliberate, documented limitation (see `HypothesisNotebookPanel.tsx`),
  not an oversight.
- **R-GCN training is in progress, not complete.** The frozen 3-config
  validation search runs on Kaggle; no final R-GCN model exists yet, so
  the ranking stub's `model_status` will not change until that finishes
  and a model is explicitly wired in.

## Deployment readiness audit (audit only -- nothing below was deployed)

**Deploy-ready now:**
- The FastAPI service has no required secrets and starts cleanly with
  zero environment configuration.
- The frontend is a static Vite build with no server-side rendering,
  deployable to any static host once it knows the API's real URL.
- `ml/scripts/export_openapi.py` -> `web/client`'s generated types gives a
  deterministic, reproducible contract between the two, which a CI step
  could enforce before any deploy.

**What would block a real split deployment (e.g. frontend on Vercel, API
on Render):**
- **No CORS configuration exists.** `ml/src/repurposemap/api/app.py`
  never adds `CORSMiddleware`. Today this works only because the Vite
  dev server proxies same-origin; a frontend served from a different
  origin than the API would have every request blocked by the browser
  until CORS is added.
- **The frontend's API base URL is hardcoded to same-origin.**
  `web/app/src/api.ts` constructs `RepurposeMapClient({ baseUrl: "" })`
  unconditionally -- there is no environment variable (e.g.
  `VITE_API_BASE_URL`) to point a deployed frontend at a separately
  deployed API. Both of the above need to change together.
- **No production reverse-proxy/static-file config exists.** The dev
  setup assumes Vite's dev-server proxy; a production deploy needs either
  a reverse proxy mapping `/api` and `/health` to the FastAPI service, or
  the base-URL fix above plus CORS.

**Environment variables that would be needed once split-deployed:**
- `VITE_API_BASE_URL` (frontend) -- does not exist yet, would need to be
  added alongside the CORS fix.
- Optionally, `PUBMED_API_KEY` / `OPENFDA_API_KEY` -- both adapters
  already accept an optional API key in their constructors
  (`PubMedEUtilsAdapter(api_key=...)`, `OpenFDALabelAdapter(api_key=...)`)
  to raise their public rate limits, but nothing currently reads these
  from the environment; the API routers always construct the adapters
  with no key. Not required to function, only to raise throughput.

**Secrets currently assumed locally:** none. No `.env` file, no hardcoded
credential, no required API key for any of the four evidence/safety
sources or Open Targets.

**CORS/API base URL summary:** not deployment-safe as configured today --
same-origin only. Fixing both together (CORS middleware + a real API base
URL env var) is a small, well-scoped follow-up, not attempted here per
this task's own instruction not to spend time deploying yet.
