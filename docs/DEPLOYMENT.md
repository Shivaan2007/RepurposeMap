# Deployment

This is the deployment-specific companion to `DEMO.md` (which covers running
both services locally). It answers: what does a real split deployment need,
what's already handled, and what's the honest serving strategy once a
trained model exists.

## What's already handled

- **CORS**: `REPURPOSEMAP_CORS_ORIGINS` (backend, comma-separated origin
  list). Never combines `*` with credentials -- this API sends no cookies.
  See `ml/src/repurposemap/api/cors.py`.
- **API base URL**: `VITE_API_BASE_URL` (frontend, build-time). Falls back
  to same-origin (`""`) when unset.
- **Port**: the production start command reads the platform's `PORT` via
  shell expansion (`"${PORT:-8000}"`), not application code.
- **Health / status**: `GET /health` and `GET /api/v1/system/info` are both
  designed to answer instantly -- see "Startup and memory" below.
- **No secrets required**: the four evidence/safety adapters and the Open
  Targets adapter all work with no API key. `PUBMED_API_KEY` /
  `OPENFDA_API_KEY` are optional, raise-throughput-only.

## Startup and memory

`ml/tests/test_api_no_graph_load.py` enforces this directly: **the live API's
five real routes never load PrimeKG into memory or hash its CSV.** Only a
future wired-in `/api/v1/rank` would need that. This means today's deployable
service:

- Starts in well under a second (no 4M-edge graph parse, no model checkpoint
  load). Measured: ~0.3s to import `repurposemap.api.app:app`.
- Has a small, roughly constant memory footprint regardless of PrimeKG's
  size on disk. Measured: ~46 MB peak RSS at import time.
- Needs no data file shipped with the container at all -- `data/raw/` is
  gitignored and irrelevant to the live service as it stands.

This is a direct consequence of `/api/v1/rank` being an honest stub (see
`docs/ARCHITECTURE.md`), not a deliberate optimization -- it's worth stating
plainly so it doesn't look accidental: **the current V1 deployment is
light precisely because no model is wired in yet.**

## Outbound network needs

The live service makes outbound HTTPS calls to four evidence/safety sources
plus Open Targets, all on the request thread (synchronous, per `API.md`):
PubMed E-utilities, ClinicalTrials.gov, openFDA, DailyMed, and the Open
Targets GraphQL API. A production host needs outbound HTTPS to all five; none
require inbound webhooks or special firewall allowances beyond that.

## Serving strategy once a model exists (R-GCN or TransE)

No model is wired in yet, so this section is a plan, not a shipped feature --
labeled that way deliberately rather than describing it as done.

Full live inference means loading PrimeKG (4M+ edges) and a trained
checkpoint into the API process's memory, which is a materially heavier
deploy than today's (likely >1GB+ depending on final embedding
dimensions, vs. today's near-zero graph footprint). Two honest options once
a model is selected, in order of preference for a free/cheap V1 host:

1. **Precomputed candidate bundle (recommended for V1).** Offline, after the
   final model is trained and locked-test-evaluated, run inference once for
   every disease the demo supports and write the ranked candidates +
   explanation paths to a static file (JSON keyed by disease ID). The live
   API serves lookups from that file -- no graph, no model weights, no
   GPU, in the request path. This must be clearly labeled in the UI and
   docs as **precomputed, not live inference** (e.g. "ranked offline on
   <date>, not recomputed per request") so it is never mistaken for a live
   model serving arbitrary new diseases. This is the "transparent
   precomputed demo bundle, clearly separated from offline research
   evaluation" approach -- honest about being a snapshot, not a simulation
   of live inference.
2. **Live inference.** Only once the precomputed bundle is insufficient
   (e.g. a reviewer wants to query a disease outside the precomputed set).
   Needs a host with enough memory for the graph + checkpoint, which likely
   means moving off a free tier. Not required for a V1 demo of the core
   workflow.

Neither is implemented, because no final model exists yet to build either
from. This section exists so the decision is made once, deliberately, when
that model lands -- not improvised under deadline pressure.

**Why option 1 cannot be built early, even as a placeholder:** every
committed experiment artifact in `ml/experiments/` (TransE's final
locked-test result included) holds only aggregate metrics (Hits@k, MRR,
AP) -- none holds a per-disease ranked candidate list. There is no real
per-disease ranking data anywhere in this repo today to put into a
"precomputed bundle," demo-disease selection (picking 2-3 diseases to
showcase) included. Writing one now would mean fabricating candidate
rankings, which this project does not do. This is also why choosing demo
diseases ahead of a trained model is not a task that can be completed
honestly yet -- it is blocked on the same thing live inference is.

## Suggested split-deployment layout

See `DEMO.md`'s "Example split-deployment configuration" for the exact
environment variables. In outline: frontend on any static host (Vercel,
Netlify, GitHub Pages), backend on any Python host that can run a long-lived
uvicorn process (Render, Fly.io, Railway). No database, no Redis, no
background-job queue is required for anything currently implemented --
PLAN.md calls for Postgres/RQ/Redis for later features (hypothesis notebook
sync, background evidence jobs), none of which exist yet.
