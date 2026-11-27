# Final release checklist (post-R-GCN-checkpoint)

**Status as of 2026-11-17: everything in this file is blocked on one thing —
the final, verified R-GCN checkpoint from the Colab/Kaggle run described in
`docs/COLAB_RGCN_FINAL_RUNBOOK.md`.** Non-scientific release-prep work is
otherwise complete (see "Repo audit" below). Nothing here changes the frozen
protocol, config, seed, split, candidate universe, or evaluation rule in
`ml/RGCN_EVALUATION_PROTOCOL.md` / `ml/RGCN_RUNBOOK.md`.

**Fast path:** `scripts/post_checkpoint.py` runs steps 0 to 4 and 8 as one stage each, with the verification and the exactly-once guard enforced in code (see `docs/POST_RGCN_COMMANDS.md`); `docs/DEPLOY_RUNBOOK.md` covers step 5; `docs/RELEASE_TEXT_TEMPLATES.md` holds the README, portfolio and resume text with `{{FILL}}` markers for the real numbers, and `python scripts/post_checkpoint.py placeholders` lists any left.

This is the exact, ordered, mechanical sequence to run once that checkpoint
exists. Each step names the precondition it depends on and the exact command
or file to touch — no step should require new design decisions.

## 0. Verify the final checkpoint (before touching anything else)

- [ ] `colab_provenance.json` (or the Kaggle equivalent) shows 3/3 epochs
      complete, finite losses, `is_final_training: true`,
      `selected_name: "rgcn-wide"`.
- [ ] Source commit == `0f23fe3631e276f36a8ba1e481da2beb634b6ccf`.
- [ ] Source archive SHA-256 == `bd26185f5db0ee0f5918936649c9f371d663034fad7bfff63939e3e9d1782085`.
- [ ] Frozen config matches exactly (`hidden_dim=128, num_bases=8,
      num_layers=2, margin=1.0, learning_rate=0.001, batch_size=1024,
      negatives_per_positive=4, seed=42, fanout_hop1=15, fanout_hop2=10,
      early_stopping_patience=None`).
- [ ] `final_checkpoint.pt` loads successfully and its own SHA-256 is
      recorded in the provenance artifact.

**Do not proceed past this point until every box above is checked.**

## 1. The one locked-test evaluation (exactly once)

- [ ] Confirm `assert_eligible_for_locked_test_evaluation` passes for this
      checkpoint.
- [ ] Run `python -m repurposemap rgcn-evaluate --split locked-test` exactly once, on
      the frozen 100-disease / 498-pair split, ranking the full 7,957-drug
      universe.
- [ ] Do **not** rerun it because of an unfavorable result — the result is
      reported as it comes out, per `ml/RGCN_EVALUATION_PROTOCOL.md`.
- [ ] Record the resulting experiment artifact path; every step below cites
      it by that path.

## 2. Fill in results

- [ ] `python ml/scripts/fill_final_results.py --artifact <the rgcn-evaluate
      --split locked-test artifact JSON>` — mechanically fills
      `docs/FINAL_RESULTS.md`'s R-GCN row. Do not hand-edit the numbers.
- [ ] Decide (owner call, flagged in `docs/FINAL_RESULTS.md` itself) whether
      to backfill macro AP for the already-evaluated path/popularity/TransE
      rows, or leave them blank with a footnote.
- [ ] Run the frozen significance method exactly as specified in
      `docs/FINAL_RESULTS.md` §3 (disease-level paired bootstrap, R-GCN vs.
      TransE, MRR and Hits@10, 95% percentile interval, 10,000+ iterations)
      and record whether each interval excludes zero.
- [ ] Write §4 ("Interpretation") in the same honest voice as README's
      existing TransE interpretation paragraph. State plainly whether R-GCN
      beat the path baseline, beat TransE, both, or neither — per the frozen
      decision rule (`ml/RGCN_EVALUATION_PROTOCOL.md`: improvement requires
      **both** MRR and Hits@10 higher than TransE's `0.0100` / `0.0161`).
- [ ] Update `README.md`'s "Project status" / "In progress" / "Roadmap"
      sections to say the locked-test evaluation is complete, with a link to
      `docs/FINAL_RESULTS.md` — do not duplicate the numbers inline beyond
      what's already there for TransE.

## 3. Real serving rankings

- [ ] Pick 3–5 diseases with real, interesting candidate structure (owner
      judgment call — favor diseases with known indications in PrimeKG for
      a meaningful case study later).
- [ ] For each: `python ml/scripts/export_model_rankings.py --export-dir
      <export> --checkpoint <final.pt> --model rgcn --disease-id <id> --top-k
      10 --ranking-method rgcn --source-experiment-file <final-locked-test-artifact>
      --output-dir data/serving/`
- [ ] Confirm each exported bundle passes `validate_bundle` (duplicate-rank
      and duplicate-drug-id checks) — the exporter should refuse to write an
      invalid one, but check logs.
- [ ] `GET /api/v1/rank/diseases` from a running backend now lists the real
      diseases; a `POST /api/v1/rank` for one of them returns
      `serving_mode: "precomputed"` instead of `model_status:
      "not_available"`.
- [ ] Load the frontend against this backend and confirm `CandidateList`
      renders a real ranked list with explanation paths, not the empty
      state.

## 4. Masked case studies

- [ ] Pick 2–3 (disease, known-drug) pairs from the diseases exported above. The disease must be a fit or validation disease, not one of the 100 locked-test diseases (their indications are absent from the training graph, so there is nothing to mask).
- [ ] `python ml/scripts/run_case_study.py --export-dir <export> --checkpoint
      <final.pt> --model rgcn --disease-id <id> --known-drug-id <id>
      --ranking-method rgcn --output case_studies/<slug>.json`
- [ ] Write up each one using `docs/CASE_STUDY_TEMPLATE.md`'s structure
      exactly — model rank, explanation paths, and live PubMed/
      ClinicalTrials.gov/DailyMed/openFDA/Open Targets context for the
      reviewer. Do not add any clinical-efficacy claim; the template's own
      gating note covers this.

## 5. Deploy

**Backend (Render)**
- [ ] Render web service created from `render.yaml`.
- [ ] Build succeeds (`pip install -e ".[api]"`).
- [ ] `REPURPOSEMAP_CORS_ORIGINS` set in the Render dashboard to the exact
      deployed Vercel origin (not `*`, never combined with credentials —
      see `ml/src/repurposemap/api/cors.py`).
- [ ] `GET /health` and `GET /api/v1/system/info` live.

**Frontend (Vercel)**
- [ ] Vercel project created from `web/app/vercel.json`.
- [ ] Build succeeds (`npm install && npm run build`).
- [ ] `VITE_API_BASE_URL` set in the Vercel dashboard to the exact deployed
      Render origin.
- [ ] Frontend loads and renders `StatusPanel` without a console error.

**No database or Redis required for V1.** If owner auth is ever needed for
either dashboard, stop at that boundary — it needs the owner's own
credentials, not something this session can do.

## 6. Public smoke tests

- [ ] `./scripts/smoke_test.sh <deployed-backend-url> <deployed-frontend-origin>` — health, system-info,
      CORS (allowed/disallowed), OpenAPI servability, rank/rank-diseases
      honesty, evidence/disease-biology honest-failure, no-secret-exposed
      check.
- [ ] Manually walk the "Live deployment-day checklist" in
      `V1_RELEASE_CHECKLIST.md`'s "Integration" section against the real
      deployed pair.
- [ ] Confirm a disease **without** an exported bundle still returns
      `model_status: "not_available"` honestly on the live deployment (not
      just locally).

## 7. Update docs/portfolio/demo materials

- [ ] `docs/DEMO_MATERIALS.md` — re-capture the main-query screenshot with a
      real ranked result (one of the diseases exported in step 3).
- [ ] `docs/PORTFOLIO.md`'s "Results placeholder" section — replace with a
      link to the now-filled `docs/FINAL_RESULTS.md`.
- [ ] `README.md` — confirm "Project status" / "Roadmap" reflect the
      completed R-GCN result (step 2 above) and that no stale "not started"
      language remains for final training or the locked test.

## 8. Final QA and release

- [ ] Full backend test suite green (`python -m pytest` from repo root —
      596 passed / 14 skipped as of 2026-11-17, torch-only tests skip
      without a GPU `.venv`; re-run with `ml` extras installed if exercising
      the embeddings/export scripts directly).
- [ ] Frontend `npm run build` green from `web/app`.
- [ ] OpenAPI drift check: `PYTHONPATH=ml/src python
      ml/scripts/export_openapi.py --output /tmp/check.json && diff
      openapi.json /tmp/check.json` — must be empty.
- [ ] `git status` clean, no stray scratch files.
- [ ] Commit the final-results/serving/case-study/doc changes.
- [ ] Push to `main`.
- [ ] Tag the release (e.g. `v1.0.0`).

## Repo audit (completed 2026-11-17, independent of final metrics)

Everything below was verified on commit `f1fcfef` and does not need to be
redone unless new code lands before the final checkpoint arrives:

- Backend test suite: 596 passed, 14 skipped (torch-only), 0 failed.
- Frontend: `npm run build` (`tsc --noEmit && vite build`) green.
- OpenAPI schema: no drift between committed `openapi.json` and a freshly
  exported one.
- `REPURPOSEMAP_CORS_ORIGINS` and `VITE_API_BASE_URL` both documented
  (`DEMO.md`, `docs/DEPLOYMENT.md`, `V1_RELEASE_CHECKLIST.md`,
  `web/app/.env.example`) and correctly wired in code
  (`ml/src/repurposemap/api/cors.py`, `web/app/src/api.ts`).
- `render.yaml` / `web/app/vercel.json` inspected: generic, no hardcoded
  domains or secrets, correct build/start commands.
- `data/serving/` correctly empty (just `.gitkeep`) — the precomputed-bundle
  mechanism (schema/loader/exporter), `POST /api/v1/rank` and `GET
  /api/v1/rank/diseases` are all wired end-to-end and tested, but there is
  nothing real to serve yet, which is the correct state before a final
  model exists.
- Frontend flows all have real loading/error/empty states wired, not just
  happy paths: `App.tsx` (loading/error for the ranking query),
  `CandidateList.tsx` (`not_available` empty state vs. a real `precomputed`
  result vs. "no candidates found"), `EvidencePanel.tsx` (loading/error,
  per-source `source_status` with failed/ok badges, Markdown research-brief
  download), `CandidateComparisonPanel.tsx` (per-column loading/error).
- README audited for stale or fabricated claims: none found. Every R-GCN
  claim is correctly stated as validation-only, with final training and the
  locked test explicitly marked not started/awaiting approval. No placeholder
  numbers are presented as real.
- `docs/FINAL_RESULTS.md` and `docs/CASE_STUDY_TEMPLATE.md` are honest `TBD`
  skeletons, explicitly gated against being filled in before a real,
  approved model exists.
- No secrets found in tracked files (checked for AWS key patterns, API
  key-shaped strings, private-key PEM blocks).
- `.gitignore` correctly excludes raw data, venvs, caches, Kaggle/Colab local
  scratch directories, and `colab_upload_bundle/` (added this session).
- Fixed this session: `V1_RELEASE_CHECKLIST.md` had a stale test count
  (`533 passed, 12 skipped` / `567 passed, 12 skipped` from an earlier
  session) that no longer matched the actual suite; corrected to the
  re-verified `596 passed, 14 skipped, 0 failed`.

## What still requires owner/browser action

- Actually creating the Render and Vercel projects (dashboard + account
  login — cannot be done from this environment).
- Setting `REPURPOSEMAP_CORS_ORIGINS` / `VITE_API_BASE_URL` to real deployed
  origins (depends on the above existing first).
- Choosing the 3–5 diseases and 2–3 case-study pairs in steps 3–4 above
  (a scientific/editorial judgment call, not a mechanical one).
- The macro-AP backfill decision in step 2.
- Monitoring and completing the Colab/Kaggle final-training run itself
  (`docs/COLAB_RGCN_FINAL_RUNBOOK.md`) — tracked in a separate session.
