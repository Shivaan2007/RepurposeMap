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
| Ranked candidates | **Stubbed** | `POST /api/v1/rank` always returns `model_status: "not_available"` with an empty candidate list. Correctly honest -- blocked on the R-GCN validation search (in progress on Kaggle; see "R-GCN status" below), not a code gap. |
| Reasoning paths | Partial | `graph/paths.py` + `scoring/path_quality.py` implement path-finding and a heuristic path-quality score; `RankingResponse.ExplanationPath` schema is wired end-to-end in the frontend (`CandidateList.tsx`). Not independently exercisable without a real ranking result, since it's reached only through the stubbed `/rank` endpoint. |
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
- **Result: 458 of 458 non-GPU-dependent tests pass**, zero failures,
  including every test in this session's own changed files
  (`test_api_cors.py`, `test_api_evidence.py`'s new backstop test) and the
  full pre-existing graph/scoring/ranking/evaluation/evidence/biology/
  provenance suite. Excluded only: the CLI tests and anything importing
  `repurposemap.embeddings`/`cli.py` (both import `torch` unconditionally)
  -- deliberately out of scope per the resource policy (torch is a large,
  unnecessary install for this test surface) and PLAN.md's model-training
  code, which the R-GCN freeze covers anyway.
- R-GCN validation search is actively `RUNNING` on Kaggle
  (`rgcn-moreBases` config remaining) -- frozen per instructions, monitored
  only.
