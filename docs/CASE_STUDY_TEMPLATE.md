# Masked case study -- writeup template

Reusable structure for the 2-3 final masked case studies referenced in
`V1_RELEASE_CHECKLIST.md` and `docs/FINAL_RESULTS.md`. This is a narrative
writeup template for a human-readable report; the code-level masking and
result-construction machinery it reports on lives in
`ml/src/repurposemap/evaluation/case_study.py`
(`CaseStudySpec` / `mask_known_indication` / `confirm_masked` /
`build_case_study_result`).

**Gate: do not fill in a copy of this template with real content until a
final model has been trained, locked-test-evaluated, and approved.** Before
that, every field below is a placeholder, same as `docs/FINAL_RESULTS.md`.
Running a case study against an unapproved or in-progress checkpoint would
make its result unreportable later anyway, since
`build_case_study_result` already refuses to construct a result from a
graph that isn't confirmed-masked -- but nothing stops someone from doing
that with the wrong checkpoint, which this gate is here to prevent.

To start a real one: copy this file to
`docs/case_studies/<disease-slug>.md`, fill in every section, and link it
from `docs/FINAL_RESULTS.md` §5.

---

## Case study: `[DISEASE NAME]`

**Status:** `TBD` (draft / reviewed / published)

### 1. Disease

- Name and ID (PrimeKG node ID): `TBD`
- One-sentence plain description of the disease: `TBD`
- Why this disease was chosen for a case study (e.g. has a well-known,
  unambiguous existing indication to mask; is not among the locked-test
  diseases unless this *is* a locked-test disease and that's stated
  explicitly): `TBD`

### 2. Known masked indication

- Known drug (name and PrimeKG ID): `TBD`
- Source of the "known" claim (e.g. the PrimeKG indication edge itself, plus
  an independent citation that it's a real approved/established use): `TBD`
- Confirmation that `mask_known_indication` + `confirm_masked` were run
  before ranking (link the script/notebook/run that did this): `TBD`

### 3. Model rank

- Ranking method used: `TBD` (must match `docs/FINAL_RESULTS.md`'s approved
  final model -- never a validation-only or earlier checkpoint)
- Rank the masked drug received among the full candidate universe: `TBD`
- For context, the same drug's rank from the path baseline and from
  popularity, if available: `TBD`

### 4. Explanation paths

- Top explanation path(s) the path-quality module found between the masked
  drug and the disease, if any (only meaningful for the path baseline or any
  future path-conditioned explanation layer; R-GCN does not natively produce
  paths): `TBD`
- Plain-language read of what the path(s) suggest biologically, with an
  explicit note if the model's rank and the path explanation disagree: `TBD`

### 5. PubMed evidence

- Query used and number of results: `TBD`
- Most relevant 1-3 papers (title, year, one-line relevance): `TBD`
- Evidence-strength label from `repurposemap.evidence` (e.g.
  `observational`, `controlled_trial`): `TBD`

### 6. ClinicalTrials.gov context

- Number of trials found for this drug/disease pair: `TBD`
- Status of the most advanced trial (e.g. completed, terminated, recruiting)
  and why that matters for reading this case study: `TBD`

### 7. DailyMed / openFDA safety context

- Boxed warning present? `TBD`
- Contraindication text relevant to this disease/population, if any: `TBD`
- One line on whether anything here would give a reviewer pause, independent
  of the model's rank: `TBD`

### 8. Open Targets biology

- Genes/targets Open Targets associates with this disease, and whether the
  masked drug's known mechanism touches any of them: `TBD`
- Note any mismatch between the model's rank and the biological plausibility
  this section surfaces -- a case study that disagrees with itself across
  sections is more informative than one that's cherry-picked to agree: `TBD`

### 9. Limitations

- Candidate-universe caveat, restated for this specific case (which
  ranker's candidate pool was used): `TBD`
- Single-disease, single-drug anecdote -- explicitly not a benchmark result;
  the locked-test table in `docs/FINAL_RESULTS.md` is the benchmark claim,
  this is illustrative only: `TBD`
- Any reason this specific disease/drug pair might be easier or harder than
  a typical one (e.g. unusually well-studied, mechanistically obvious):
  `TBD`

### 10. Research-only disclaimer

Standard disclaimer, unmodified: this case study illustrates a research
hypothesis-generation process. It is not a treatment recommendation, does
not establish clinical efficacy, and should not inform any clinical
decision.
