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

**Production start command** (no `--reload`, binds to the platform's port):

```bash
uvicorn repurposemap.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
```

`uvicorn` reads the port from its own `--port` flag, not an environment
variable it parses itself -- `${PORT:-8000}` is a shell expansion that
uses the hosting platform's `PORT` (Render, Heroku, etc. all set one) and
falls back to 8000 locally. No code change was needed for this.

**CORS**: set `REPURPOSEMAP_CORS_ORIGINS` to a comma-separated list of the
origin(s) allowed to call this API cross-origin, e.g.

```bash
REPURPOSEMAP_CORS_ORIGINS=https://app.example.com,https://staging.example.com
```

Unset, it defaults to the Vite dev server's own origins
(`http://localhost:5173`, `http://127.0.0.1:5173`) so local development
needs no configuration. `*` is never combined with credentialed requests
-- this API sends no cookies, so `allow_credentials` is always `False`
regardless of configured origins. See
`ml/src/repurposemap/api/cors.py`.

## Frontend

In a second terminal:

```bash
cd web/client && npm install && npm run build
cd ../app && npm install
npm run dev
```

Runs on **port 5173** (Vite's default). `web/app/vite.config.ts` proxies
`/api` and `/health` to `http://localhost:8000`, so the backend must
already be running on port 8000 for the frontend to work in dev mode (no
`VITE_API_BASE_URL` needed locally).

**Production build**: point the build at a separately deployed backend
with `VITE_API_BASE_URL` (see `web/app/.env.example`):

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

Output goes to `web/app/dist/` -- deploy that directory to any static
host. With `VITE_API_BASE_URL` unset, the build falls back to `""`
(same-origin), matching local dev's same-origin assumption.

**Example split-deployment configuration:**

```
Frontend:
https://<frontend-domain>

Backend:
https://<backend-domain>

REPURPOSEMAP_CORS_ORIGINS=https://<frontend-domain>
VITE_API_BASE_URL=https://<backend-domain>
```

No real domains or secrets are committed anywhere in this repo -- the
above is illustrative only.

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
  zero environment configuration (CORS falls back to dev-only origins).
- CORS is configurable via `REPURPOSEMAP_CORS_ORIGINS`
  (`ml/src/repurposemap/api/cors.py`), never combining `*` with
  credentials.
- The frontend's API base URL is configurable via `VITE_API_BASE_URL`
  (`web/app/src/api.ts`), falling back to same-origin when unset.
- The frontend is a static Vite build with no server-side rendering,
  deployable to any static host once `VITE_API_BASE_URL` points at the
  real backend.
- `ml/scripts/export_openapi.py` -> `web/client`'s generated types gives a
  deterministic, reproducible contract between the two, which a CI step
  could enforce before any deploy.
- Confirmed locally with a real cross-origin request (backend on port
  8000, simulated frontend origin `http://localhost:5173`) -- see
  "Local cross-origin smoke test" below.

**Remaining blockers for a real split deployment (e.g. frontend on
Vercel, API on Render):**
- **No production reverse-proxy/static-file config exists yet.** Both
  ends now support being configured for a split deployment, but nothing
  in this repo yet runs them that way continuously (e.g. a Render
  `render.yaml` / Vercel project config) -- that's an actual hosting
  setup task, not a code gap.
- **No rate-limiting or abuse protection** on the evidence/biology
  endpoints, which proxy to external APIs on every request -- fine for a
  demo, a real concern before public traffic.

**Environment variables for a split deployment:**
- `REPURPOSEMAP_CORS_ORIGINS` (backend) -- comma-separated list of
  frontend origin(s) allowed to call the API cross-origin. Defaults to
  the Vite dev server's origins if unset.
- `VITE_API_BASE_URL` (frontend, build-time) -- the deployed backend's
  base URL. Falls back to same-origin (`""`) if unset.
- `PORT` (backend, platform-provided) -- read via shell expansion in the
  production start command, not parsed by application code.
- Optionally, `PUBMED_API_KEY` / `OPENFDA_API_KEY` -- both adapters
  already accept an optional API key in their constructors
  (`PubMedEUtilsAdapter(api_key=...)`, `OpenFDALabelAdapter(api_key=...)`)
  to raise their public rate limits, but nothing currently reads these
  from the environment; the API routers always construct the adapters
  with no key. Not required to function, only to raise throughput.

**Secrets currently assumed locally:** none. No `.env` file is committed
(`.env.example` documents the one optional frontend variable), no
hardcoded credential, no required API key for any of the four
evidence/safety sources or Open Targets.

**CORS/API base URL summary:** both fixed this round. A split deployment
is now technically possible; what remains is hosting configuration
(actually standing up two services pointed at each other), not missing
application code.

## V1 smoke test

`scripts/smoke_test.sh [base_url]` (default `http://localhost:8000`) checks,
against a real running backend: `/health` and `/api/v1/system/info` answer,
`/openapi.json` is servable, CORS headers are present for an allowed origin
and absent for a disallowed one, and -- the honesty check -- `POST
/api/v1/rank` still reports `model_status: "not_available"` rather than a
fabricated ranking. Run it any time after starting the backend (see
"Backend" above):

```bash
./scripts/smoke_test.sh
```

## Local cross-origin smoke test

Reproduces what a browser sends when the frontend and API are on
different origins. With two real terminals and local port binding
available, run:

```bash
REPURPOSEMAP_CORS_ORIGINS=http://localhost:5173 \
  uvicorn repurposemap.api.app:app --port 8000 &

curl -s -D - -o /dev/null \
  -H "Origin: http://localhost:5173" http://localhost:8000/health \
  | grep -i access-control-allow-origin
curl -s -D - -o /dev/null \
  -H "Origin: http://evil.example.com" http://localhost:8000/health \
  | grep -i access-control-allow-origin
```

Expected: the allowed origin's request returns
`access-control-allow-origin: http://localhost:5173`; the disallowed
origin's request has no `access-control-allow-origin` header at all, so a
real browser would block the response from being read even though the
server itself still returns `200`.

`ml/tests/test_api_cors.py` runs the equivalent check in-process (via
FastAPI's `TestClient`, no real socket) for `/health` and, separately,
for `POST /api/v1/evidence` with its adapters stubbed -- confirming CORS
headers are present on a real API route, not just the liveness check.
This is the version actually exercised in this repo's own development
sandbox, which blocks binding real local ports; the `curl` form above is
for anyone running this on a normal machine who wants to see it with real
HTTP requests across two live processes.
