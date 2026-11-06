# RepurposeMap API

The boundary between the Python research engine and the future TypeScript
research workspace:

```
Python research engine -> FastAPI + Pydantic -> OpenAPI
    -> generated TypeScript client -> future React research workspace
```

Seven real endpoints exist today: `/health`, `/api/v1/system/info`,
`POST /api/v1/evidence`, `POST /api/v1/disease-biology`,
`POST /api/v1/target-scorecard`, `POST /api/v1/rank`, and
`GET /api/v1/rank/diseases`. `POST /api/v1/rank` is backed by the
precomputed serving mechanism (`repurposemap.serving`) rather than live
model inference -- see "Precomputed ranking serving" below. There is no
trained model wired in yet, and `data/serving/` is empty today, so in
practice this endpoint still always reports `model_status:
"not_available"`. The evidence/biology/scorecard lookup endpoints are
synchronous -- see "Evidence/safety domain" and "Disease biology domain"
below for what that means and why.

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

## Precomputed ranking serving

`POST /api/v1/rank` (`ml/src/repurposemap/api/ranking.py`) is backed by
`repurposemap.serving`, not live model inference:

1. `repurposemap.serving.schemas.CandidateRankingBundle` is the on-disk
   contract for one disease's precomputed ranking -- ranked candidates,
   ranking-method/provenance, candidate-universe description, and an
   explicit `generated_at`.
2. `repurposemap.serving.loader` reads validated bundles from
   `data/serving/*.json` (one file per disease, named `<disease_id>.json`).
   A missing or malformed file is reported honestly, never silently
   substituted.
3. `repurposemap.serving.export_rankings` is the offline CLI that builds
   a bundle from an already-ranked, already-approved candidate list and
   writes it to `data/serving/` -- it never computes or re-ranks
   anything itself (`python -m repurposemap.serving.export_rankings --help`).
4. The router matches a request's `disease_id` or `disease_name` against
   bundles in `data/serving/` by exact match only (case-insensitive for
   names) -- never a fuzzy guess at which bundle a free-text name means.

`data/serving/` is empty today (only a `.gitkeep`): no committed
experiment artifact anywhere holds a real per-disease ranked candidate
list yet (see `docs/ARCHITECTURE.md`). So in practice, every request to
`/api/v1/rank` today still returns `model_status: "not_available"`,
`serving_mode: "not_available"`, with an empty candidate list --
exactly the same honest response this endpoint has always returned,
just now backed by a real mechanism with nothing to serve yet rather
than a hardcoded stub. `GET /api/v1/rank/diseases` lists whichever
diseases do have a bundle, so a caller can discover this without a
failed query first.

Once a real model exists and passes its locked-test evaluation, wiring
it in means running `export_rankings.py` against its approved ranking
output for the demo diseases chosen for deployment -- no code change to
`ranking.py` itself is needed. `ModelStatus` in `system_info.py` is a
separate signal (whether a trained checkpoint is configured for
inference) and is unaffected by whether a serving bundle exists.

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

## Evidence/safety domain

`repurposemap.evidence` defines typed contracts (`EvidenceSource`,
`EvidenceRecord`, `PubMedEvidence`, `ClinicalTrialEvidence`,
`SafetyEvidence`, `EvidenceBundle`) and live adapters for all four named
sources: PubMed, ClinicalTrials.gov, openFDA and DailyMed. Each adapter
raises its own typed error on a real failure rather than returning an
empty or fabricated result; none classifies or rates evidence beyond a
coarse, deterministic `strength` derived only from the source's own
controlled vocabulary (PubMed's publication types, ClinicalTrials.gov's
`studyType`) -- never from free text.

`POST /api/v1/evidence` wires all four into one request:
`{"drug_name": ..., "disease_name": ..., "max_results": 5}` returns an
`EvidenceBundle` plus a `source_status` list saying whether each of the
four sources succeeded, with its error if not -- a failure in one source
never drops the other three's results, and is never silently reported as
"zero matches". **This call is synchronous**: all four external APIs are
queried in-process on the request thread, so a slow or rate-limited
source adds directly to response latency. `PLAN.md`'s tech stack calls
for this to move into a background job (RQ + Redis) once that
infrastructure exists in this repo -- it does not yet, so this is a
deliberate stepping stone, not the final design. The frontend's
`EvidencePanel` (`web/app/src/components/EvidencePanel.tsx`) is a thin,
synchronous caller of the same endpoint.

## Disease biology domain

`repurposemap.biology` defines typed contracts (`DiseaseBiologyModule`,
`TargetAssociation`, `Pathway`, `DiseaseMatch`, `TargetScorecard`,
`TractabilityAssessment`, `TissueExpression`, `TargetMatch`) and a live
Open Targets Platform adapter (`OpenTargetsAdapter`) for PLAN.md's
researcher-workflow steps 1 and 2. No API key is required.

**Step 1 ("What goes wrong in this disease?")**: the genes most
strongly associated with a disease, ranked by genetic evidence, with
their Reactome pathways.

`POST /api/v1/disease-biology` takes exactly one of `disease_name` /
`disease_id` (an EFO or MONDO ID) plus `max_targets`. A `disease_name` is
resolved to an ID first via Open Targets' own full-text search; the top
candidates it considered are returned in `resolved_from` so a caller can
see what else the name could have meant, not just the one picked. Unlike
`/api/v1/evidence` (four independent sources, so a partial result is the
normal case), this endpoint has one source: a resolution/lookup failure
raises a real HTTP error (`404` for an unresolvable name or unknown ID,
`502` for an Open Targets failure) rather than returning `200` with an
empty module. **This call is synchronous** for the same reason
`/api/v1/evidence` is -- see that section above. The frontend's
`DiseaseBiologyPanel` (`web/app/src/components/DiseaseBiologyPanel.tsx`)
is a thin, synchronous caller of the same endpoint.

**Step 2 ("Which proteins are worth hitting?")**: for one target, with
no disease context -- its tractability assessments (per drug modality:
small molecule, antibody, PROTAC, other), Open Targets' own
target-prioritisation factors (`geneticConstraint`, `mouseKOScore`,
`tissueSpecificity`, ... -- kept under their own Open Targets names,
never re-labelled), and its ten highest-expressing tissues.

`POST /api/v1/target-scorecard` takes exactly one of `gene_symbol` /
`ensembl_id`, with the same name-resolution and error-shape convention
as `/api/v1/disease-biology` (`resolved_from`, `404`/`502` on a real
failure). Also synchronous, for the same reason. The frontend's
`TargetScorecardPanel`
(`web/app/src/components/TargetScorecardPanel.tsx`) is a thin,
synchronous caller.

`repurposemap.hypothesis.RepurposingHypothesis` shows how a model score,
explanation paths, an evidence bundle, safety flags and provenance will
assemble into one researcher-facing object once a real ranking model
exists; `/api/v1/evidence` is not yet called from the ranking path.
