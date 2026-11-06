# Demo script (current V1 state)

This walks through what's real and runnable **today**. It is not PLAN.md's
eventual 2-minute demo video script (that one assumes a trained ranking
model and published results, neither of which exist yet) -- this is the
honest version for right now. Update it once ranking is wired in.

Setup: follow `DEMO.md` to get both services running locally
(`localhost:8000` backend, `localhost:5173` frontend).

## 1. The problem, in one line

"RepurposeMap helps a researcher find existing drugs that might treat a
disease with no approved treatment, and -- more importantly right now --
never pretends to have an answer it doesn't have."

## 2. System status tells the truth

Open the app. Point at the status panel (`GET /api/v1/system/info`) --
it reports whether a ranking model is actually available. Today it
reports `False`, honestly, because no R-GCN/TransE checkpoint is wired in
yet. Say this is deliberate: "this never fakes a model being ready."

## 3. Search a disease

Type a disease into the main query (e.g. "pulmonary arterial
hypertension") and submit. The result panel explains, plainly, that
ranked candidates are not available for this disease in the current
deployed dataset -- not an empty table with no explanation. Point out
the list of diseases with a deployed ranking bundle shown above the
query box (empty today, via `GET /api/v1/rank/diseases`) -- the UI
tells you in advance what it can actually answer, rather than making
you find out by a failed query. [DEMO DISEASE]: once a real model's
approved ranking output is exported for a chosen demo disease, this
step becomes the actual candidate-list walkthrough -- see
`docs/DEPLOYMENT.md`'s "Serving strategy."

## 4. Disease biology (real, live data)

Scroll to "Disease biology." The same disease name is already filled in
(carried over from the main query). Submit it -- this calls Open Targets
live and shows the genes most strongly associated with the disease, ranked
by genetic evidence, with their pathways. Point out `resolved_from`: if the
disease name was ambiguous, the other candidates Open Targets considered
are shown too, not hidden.

## 5. Target scorecard (real, live data)

Pick one of the genes just shown (e.g. its approved symbol) and look it up
in "Target prioritization" -- tractability per drug modality, Open
Targets' own prioritisation factors kept under their original names, and
its top-expressing tissues.

## 6. Evidence lookup + exportable brief (real, live data)

Scroll to "Evidence lookup." Disease name is pre-filled; type a candidate
drug (e.g. "sildenafil"). Submit -- this queries PubMed, ClinicalTrials.gov,
openFDA and DailyMed live, in parallel-looking but actually sequential
per-source calls, and shows which sources succeeded. Click "Download brief
(Markdown)" -- this produces a real, shareable one-page evidence brief from
the data just fetched, with no server round-trip for the export itself.

## 7. Candidate comparison (real, live data, no fabricated ranking)

Scroll to "Candidate comparison." Add 2-3 drug names for the same disease.
Submit -- each column shows evidence-lookup counts side by side. Point out
the "Model score" and "Explanation paths" rows explicitly say "not
available (no trained model yet)" rather than being left blank or guessed.

## 8. Hypothesis notebook (local, honest about scope)

Add one of the candidates just compared as a tracked hypothesis, set its
status to "Reviewing," add a note. Mention this is deliberately
browser-local (no backend, no database) -- say so if asked, don't let it
be mistaken for a shared/multi-device feature.

## 9. Close on R-GCN status

"The missing piece is ranking data itself -- an R-GCN validation search
is actively running on Kaggle right now [give current config/status from
`V1_RELEASE_CHECKLIST.md`]. The serving mechanism that will carry a real
model's output into this exact UI is already built and tested
(`repurposemap.serving`); what's missing is [FINAL MODEL]'s approved
ranking output to export into it."

## What NOT to demo yet

- Ranked candidates with scores (no precomputed bundle exists yet, will show "not available").
- A results/KPI page (doesn't exist -- no model results to publish).
- PDF/BibTeX/RIS export (only Markdown export exists).
- Masked case-study replay as a finished result (harness infrastructure
  exists; not yet run against a trained model).
