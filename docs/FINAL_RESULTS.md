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

| | Hits@1 | Hits@3 | Hits@10 | MRR | Macro AP |
|---|---|---|---|---|---|
| Path baseline (bounded candidates) | 0.01004 | 0.04016 | 0.08635 | 0.03329 | `TBD` |
| Popularity baseline (full universe, *approximate*) | 0.0 | ~0.004 | ~0.014 | ~0.0062 | `TBD` |
| Final TransE (full universe) | 0.0020 | 0.0040 | 0.0161 | 0.0100 | `TBD` |
| **Final R-GCN (full universe)** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

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

## 3. Statistical significance (placeholder -- not yet done)

`TBD`. Nothing below is computed; this section exists so the question gets
asked before publication, not after.

- Decide a method before looking at any R-GCN locked-test number: e.g.
  bootstrap resampling over the 100 locked-test diseases (not the 498 pairs
  individually, to respect the disease-held-out unit) to get a confidence
  interval on each metric, per ranker.
- Report whether R-GCN's interval overlaps TransE's and the path baseline's.
  An R-GCN win with overlapping intervals is a different claim than a
  non-overlapping one -- say which it is, don't just report point estimates.
- If this is skipped for time, say so explicitly in the published writeup
  ("not bootstrapped; point estimates only, N=100 diseases") rather than
  silently omitting it.

## 4. Interpretation

`TBD`. Write this the same way README's TransE interpretation paragraph was
written: state plainly whether R-GCN beat the path baseline, beat TransE,
beat neither, or beat both -- and what that does and doesn't mean
scientifically. Explicitly repeat: no result in this document establishes
clinical efficacy for any candidate drug.

## 5. Case studies

Link out to the filled-in case studies once they exist -- see
`docs/CASE_STUDY_TEMPLATE.md` for the per-case-study structure. Do not run or
draft an actual case study until a real model's ranking output has been
locked-test-evaluated and approved (see that template's own gating note).

## Owner sign-off before publishing

- [ ] Locked-test evaluation for R-GCN has actually run, exactly once, under
      `assert_eligible_for_locked_test_evaluation`'s checks.
- [ ] The result in this file matches the experiment JSON artifact's numbers
      exactly (copy-paste, not retyped).
- [ ] Significance section (`§3`) is either filled in or explicitly marked
      skipped, not left silently blank.
- [ ] `docs/PORTFOLIO.md`'s "Results placeholder" section is updated in the
      same commit as this file, so the two never disagree about what's known.
