# V1 release checklist

Status against PLAN.md's Core (Version 1) feature table and researcher
workflow, as of the current repo state. Labels are honest, not aspirational:
**Done**, **Partial** (something real exists but doesn't fully match the
PLAN.md description), **Stubbed** (contract exists, deliberately returns "not
available" rather than a fake result), or **Missing**.

## Core (Version 1) features

| Feature | Status | Notes |
|---|---|---|
| Disease search | Partial | `repurposemap.graph.search` does exact/fuzzy entity matching over the loaded graph; Open Targets name resolution (with `resolved_from` alternates) backs `/api/v1/disease-biology` and `/api/v1/target-scorecard`. No dedicated rare-disease-synonym table beyond what PrimeKG/Open Targets already encode. |
| Ranked candidates | **Mechanism done, no data yet** | `POST /api/v1/rank` is now backed by a real precomputed-serving mechanism (`repurposemap.serving`, `GET /api/v1/rank/diseases` for discovery) rather than a hardcoded stub, but `data/serving/` is empty -- every request still honestly returns `model_status: "not_available"`. R-GCN's validation search is now complete (`rgcn-wide` selected, see "Known blockers" below), but final training and the locked-test evaluation -- the step that would produce a real approved ranking to export -- have not started and require explicit owner approval. Not a code gap. |
| Reasoning paths | Partial | `graph/paths.py` + `scoring/path_quality.py` implement path-finding and a heuristic path-quality score; `RankingResponse.ExplanationPath` schema is wired end-to-end in the frontend (`CandidateList.tsx`) and in the serving bundle schema (`ExplanationPathRecord`). Not independently exercisable without a real ranking result, since `/rank` has nothing in `data/serving/` to serve yet. |
| Evidence check | Partial | Real PubMed + ClinicalTrials.gov adapters exist and are labeled by a deterministic, source-controlled-vocabulary `EvidenceStrength` (`unknown` / `case_report` / `observational` / `controlled_trial` / `systematic_review`). This is a different, finer taxonomy than PLAN.md's three-way "Novel / Some early evidence / Already tested" -- the three-way label is not implemented. |
| Safety check | Done | Real openFDA + DailyMed adapters; `boxed_warning_present` and `contraindication_text` are surfaced in both the API response and `EvidencePanel`. |
| Triage shortlist | Missing | No code combines model score + evidence + safety into one ranked list. `repurposemap.hypothesis.RepurposingHypothesis` is a typed placeholder for this, documented as not yet wired to the ranking path (`API.md`). Blocked on ranked candidates existing at all. |
| Evidence brief | Partial (just added) | `EvidencePanel` now has a client-side "Download brief (Markdown)" button producing a Markdown research brief from the already-fetched evidence bundle. No PDF, BibTeX/RIS citation export, or CSV table export -- deliberately not built per current scope guidance (one lightweight format only). |
| Results page | Missing | No page publishing Hits@10/MRR/AUPRC/time-travel results exists. Correctly blocked: this is a Phase 7 deliverable and the R-GCN results it would report don't exist yet. |

## Researcher workflow (steps 1, 2, 5, 7 are "build first" per PLAN.md)

| Step | Status | Notes |
|---|---|---|
| 1. Disease biology | Done | `POST /api/v1/disease-biology`, `DiseaseBiologyPanel.tsx`. |
| 2. Target prioritization | Done | `POST /api/v1/target-scorecard`, `TargetScorecardPanel.tsx`. |
| 3. Hypothesis generation | Stubbed | See "Ranked candidates" above. |
| 5. Evidence review | Partial | See "Evidence check" above. |
| 7. Safety profile | Done | See "Safety check" above (label warnings + contraindications; no hERG/Tox21 predictions, which PLAN.md marks "Later" anyway). |

Scientist-style tools: hypothesis notebook (done, localStorage-only by
design), side-by-side scorecard (done, as `CandidateComparisonPanel`, limited
to evidence-lookup counts since no model score exists yet), provenance on
every claim (done for experiments/provenance service; evidence/safety
responses link their own source), export (partial -- see Evidence brief
above), case studies (masked case-study harness infrastructure exists per
recent commits; not yet run against a trained model).

## Deployment readiness

| Item | Status |
|---|---|
| Frontend production build | Done -- `npm run build` green, outputs static `dist/` |
| Backend production start command | Done -- documented in `DEMO.md` (`uvicorn ... --host 0.0.0.0 --port "${PORT:-8000}"`) |
| CORS | Done -- `REPURPOSEMAP_CORS_ORIGINS`, never combines `*` with credentials |
| Frontend API base URL | Done -- `VITE_API_BASE_URL`, falls back to same-origin |
| Health endpoint | Done -- `GET /health` |
| System-status endpoint | Done -- `GET /api/v1/system/info`, reports model availability honestly |
| CI / automated gate on push | Missing -- no `.github/workflows`. Out of scope per current guidance unless it blocks Nov 30; it doesn't block a manual deploy. |
| Secrets in repo | Clean -- audited, none found; no `.env` files on disk, `.env.example` only |
| Large/raw files in repo | Clean -- `data/raw/`, Kaggle recovery/scratch dirs correctly gitignored and untracked |
| Reproducible local Python setup | Done -- see "Known blockers" below for how this environment specifically got there; `README.md`'s documented `pip install -e ".[dev]"` path works on a normal machine |
| Deployment config files | Done -- `render.yaml` (backend) and `web/app/vercel.json` (frontend), generic, no domains or secrets. Inspected for correctness, not yet exercised against a real Render/Vercel account |
| Deployment smoke test | Done -- `scripts/smoke_test.sh` covers health, system-info, CORS (allowed/disallowed), OpenAPI servability, rank/rank-diseases honesty, evidence/disease-biology honest-failure (never an unhandled 500), and a no-secrets-exposed check |
| Precomputed ranking serving mechanism | Done -- `repurposemap.serving` (schema, loader, offline exporter CLI), `POST /api/v1/rank` and `GET /api/v1/rank/diseases` wired to it end-to-end including the frontend, with tests. See `API.md`'s "Precomputed ranking serving" section |
| Precomputed demo ranking data | Blocked, not a code gap -- every committed experiment artifact holds aggregate metrics only, never a per-disease candidate list, so `data/serving/` has nothing real to hold yet. Exporting one is a data step (`python -m repurposemap.serving.export_rankings`) once a model is trained and locked-test-evaluated, not a code change. See "Serving strategy" in `docs/DEPLOYMENT.md` |

## Known blockers (environment, not code) -- resolved

- **`pip install` cannot complete in this sandbox**: even a single small
  package's PyPI index page fails with a truncated-read error through the
  sandbox's proxy, on every retry (confirmed repeatedly). **Worked around**:
  resolved the full dependency closure for the non-GPU test surface
  (fastapi, pytest, httpx, networkx, pandas, numpy, + their transitive
  deps -- 25 packages total) via PyPI's JSON API + direct `curl` wheel
  downloads (reliable through the same proxy where pip's own HTTP stack
  was not), verified each wheel's zip integrity, then installed offline
  with `pip install --no-index --no-deps`. This produced a real, working
  `.venv` in this repo.
- **Result: 533 passed, 12 skipped, 0 failed** across the full non-GPU
  test surface, including every test in this session's own changed files
  (`test_api_cors.py`, `test_api_evidence.py`'s backstop test,
  `test_provenance_service.py`'s most-recent-final-training-record test)
  and the full pre-existing graph/scoring/ranking/evaluation/evidence/
  biology/provenance suite. Excluded only: files that `import torch`
  unconditionally (`cli.py`, `test_cli.py`, `test_primekg_cli.py`,
  `test_scoring_cli.py`, `test_rgcn_cli.py`, `test_rgcn_sampling.py`,
  `test_transe_cli.py`, and the `embeddings/` package) -- deliberately out
  of scope per the resource policy (torch is a large, unnecessary install
  for this test surface) and PLAN.md's model-training code, which the
  R-GCN freeze covers anyway.
- R-GCN validation search is **complete** (Kaggle kernel
  `shivaanbhalla/repurposemap-rgcn-validation-search`, version 8,
  `KernelWorkerStatus.COMPLETE`). All three configs (`rgcn-base`,
  `rgcn-wide`, `rgcn-moreBases`) ran their full 3/3-epoch budget, no
  NaN/Inf detected, source-archive SHA-256 verified against the frozen
  commit's expected hash. Applying the frozen selection rule, `rgcn-wide`
  wins on validation MRR (0.0392 vs. runner-up 0.0271, far past the
  0.0005 tie-break threshold). See
  `ml/experiments/rgcn_sampled_validation_search_0f23fe3.json` for the
  full record. **Final training and the locked-test evaluation have not
  started and require explicit owner approval before proceeding.**

## Live deployment-day checklist

Nothing below is marked done until actually exercised against a real
Render/Vercel account -- `render.yaml`/`web/app/vercel.json` have only been
inspected for correctness so far (see "Deployment config files" above).

**Backend (Render)**
- [ ] Render web service created from `render.yaml`
- [ ] Repo connected, build succeeds (`pip install -e ".[api]"`)
- [ ] `REPURPOSEMAP_CORS_ORIGINS` set to the real deployed frontend origin
- [ ] `GET /health` live
- [ ] `GET /api/v1/system/info` live
- [ ] Evidence/biology endpoints can reach PubMed/ClinicalTrials.gov/openFDA/DailyMed/Open Targets (outbound internet works from the host)

**Frontend (Vercel)**
- [ ] Vercel project created from `web/app/vercel.json`
- [ ] Repo connected, build succeeds (`npm install && npm run build`)
- [ ] `VITE_API_BASE_URL` set to the real deployed backend origin
- [ ] Frontend loads and renders `StatusPanel` without a console error

**Integration**
- [ ] CORS: a request from the deployed frontend origin to the deployed backend succeeds; a request from an arbitrary origin does not get the CORS header
- [ ] `GET /api/v1/rank/diseases` answers (empty list is fine)
- [ ] A supported-disease rank query returns `serving_mode: "precomputed"` once `data/serving/` has a real bundle; an unsupported one returns `model_status: "not_available"` honestly
- [ ] Evidence/disease-biology/target-scorecard queries return real data for a real drug/disease/gene
- [ ] A deliberately bad request (e.g. an empty disease name) renders an honest error in the UI, not a crash
- [ ] No secret-shaped value appears in `/api/v1/system/info` or `/openapi.json` on the deployed host (`scripts/smoke_test.sh <deployed-backend-url>`)

Run `./scripts/smoke_test.sh <deployed-backend-url>` against the real
deployed backend once it's live, in addition to walking this list by hand.
