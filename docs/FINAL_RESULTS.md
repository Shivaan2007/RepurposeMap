# Final results page (skeleton -- no numbers yet)

This is the structure for the eventual public "final results" writeup once
R-GCN's final training and single locked-test evaluation are complete and
approved. It exists now so that landing those numbers is a fill-in-the-blanks
step, not a from-scratch writeup under deadline pressure.

**Nothing in this file is a real result.** Every number below is `TBD`. Do
not fill in a row until the exact artifact it cites actually exists and has
been approved (see `V1_RELEASE_CHECKLIST.md`'s "Known blockers" and
`ml/RGCN_EVALUATION_PROTOCOL.md`). This file itself makes no scientific
claim -- it is release-closure scaffolding, not a results report.

## 1. One-line summary

`TBD` -- one sentence once R-GCN's locked-test result exists, in the same
honest voice as README's "Interpretation" paragraph for the TransE result
(state what the result means, not just the numbers).

## 2. Locked-test comparison table

Same four candidate rankers already in `README.md`'s "Final TransE result
(locked test)" table, extended with the R-GCN row and a macro AP column once
it's computed. The path/popularity/TransE values below are **copied
verbatim** from the existing locked-test table -- no new locked-test
evaluation is run to produce this file.

Fill the R-GCN row mechanically, not by hand:
`python ml/scripts/fill_final_results.py --artifact <the rgcn-evaluate --split locked-test --artifact JSON>`
-- it replaces the exact `TBD` row below with that artifact's real numbers
and refuses to run against anything but a `split: "locked-test"` artifact.

| | Hits@1 | Hits@3 | Hits@10 | MRR | Macro AP |
|---|---|---|---|---|---|
| Path baseline (bounded candidates) | 0.01004 | 0.04016 | 0.08635 | 0.03329 | `TBD` |
| Popularity baseline (full universe, *approximate*) | 0.0 | ~0.004 | ~0.014 | ~0.0062 | `TBD` |
| Final TransE (full universe) | 0.0020 | 0.0040 | 0.0161 | 0.0100 | `TBD` |
| **Final R-GCN (full universe)** | 0.0120 | 0.0281 | 0.0622 | 0.0334 | 0.0332 |

Macro AP is `TBD` even for the three already-evaluated rankers above: it was
computed for R-GCN's validation-search selection rule
(`ml/experiments/rgcn_sampled_validation_search_0f23fe3.json`) but has not
been computed against the locked test for path/popularity/TransE. Decide
before filling this in whether backfilling it for the existing three is worth
doing, or whether the R-GCN row alone should carry a macro AP value with the
others left blank and footnoted as "not computed."

### Candidate-universe caveat (restate, don't re-derive)

Copy README's existing caveat verbatim: the path baseline ranks a bounded,
per-disease discovered candidate set; popularity, TransE and R-GCN all rank
the full 7,957-drug universe. All four share the same held-out pairs and
metric formulas, but this is not the same ranking task across rows -- no
"R-GCN is N times better/worse than path" claim should be made from this
table alone.

## 3. Statistical significance -- method frozen, result still `TBD`

**Method (frozen before any R-GCN locked-test number exists, same spirit as
the R-GCN validation-search selection rule in `README.md`'s "Project
status": decided once, in advance, never adjusted after seeing the result
it will be applied to):**

**Disease-level paired bootstrap over the 100 locked-test diseases,
comparing R-GCN vs. TransE, on MRR and Hits@10.**

- Unit of resampling is the **disease**, not the individual held-out pair --
  this locked test holds out 100 diseases / 498 pairs (`README.md`'s
  "Evaluation protocol"), and the disease is the actual independent unit the
  held-out split was built around, so resampling individual pairs would
  overstate the effective sample size for a disease with several held-out
  drugs.
- For each bootstrap iteration: resample the 100 diseases with replacement,
  recompute MRR and Hits@10 for **both** R-GCN and TransE on that same
  resampled set of diseases (paired -- both rankers see the identical
  resampled diseases each iteration, not independently resampled sets), and
  record the two metrics' *difference* (R-GCN minus TransE) for that
  iteration.
- Across iterations (a four-digit iteration count, e.g. 10,000, is enough at
  this N and is cheap to compute -- no GPU, no retraining, pure
  resampling over already-computed per-disease scores), report the
  resulting distribution's 95% percentile interval for the MRR difference
  and the Hits@10 difference separately.
- **Report whether each interval excludes zero.** An interval that excludes
  zero supports a real (R-GCN vs. TransE) difference on that metric; an
  interval that includes zero does not, regardless of which point estimate
  is numerically higher -- state this plainly, don't just report point
  estimates and let the reader assume significance.
- Explicitly **not** in scope for this method: R-GCN vs. the path or
  popularity baselines (frozen here as TransE-only, since TransE is the
  other full-universe learned baseline and the most informative single
  comparison; a path/popularity comparison would need to additionally
  account for the candidate-universe mismatch in `§2`, which this paired
  resampling method does not attempt to solve).
- Still **not computed** -- this section freezes the method, not the
  result. Nothing here has been run against any R-GCN number, because no
  R-GCN locked-test number exists yet.

## 4. Interpretation

`TBD`. Write this the same way README's TransE interpretation paragraph was
written: state plainly whether R-GCN beat the path baseline, beat TransE,
beat neither, or beat both -- and what that does and doesn't mean
scientifically. Explicitly repeat: no result in this document establishes
clinical efficacy for any candidate drug.

## 5. Case studies

Link out to the filled-in case studies once they exist -- see
`docs/CASE_STUDY_TEMPLATE.md` for the per-case-study structure and
`ml/scripts/run_case_study.py` for the runner that produces each one's
`rank` and provenance JSON. Do not run or draft an actual case study until
a real model's ranking output has been locked-test-evaluated and approved
(see that template's own gating note).

## Owner sign-off before publishing

- [ ] Locked-test evaluation for R-GCN has actually run, exactly once, under
      `assert_eligible_for_locked_test_evaluation`'s checks.
- [ ] The result in this file matches the experiment JSON artifact's numbers
      exactly (copy-paste, not retyped).
- [ ] Significance section (`§3`) is filled in using the frozen paired-
      bootstrap method exactly as specified there -- not a different method
      chosen after seeing the result, and not left silently blank.
- [ ] `docs/PORTFOLIO.md`'s "Results placeholder" section is updated in the
      same commit as this file, so the two never disagree about what's known.
