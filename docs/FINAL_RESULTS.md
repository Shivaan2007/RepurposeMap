# Final results page

The R-GCN final training and single locked-test evaluation are complete and
approved (`ml/experiments/rgcn_final_locked_test_ca9313b1.json`, evaluated
2026-11-26). This file is the final results writeup for that result.

## 1. One-line summary

The frozen sampled R-GCN (`rgcn-wide`), evaluated once on the 100-disease /
498-pair locked test, beat the final TransE baseline on both MRR (0.0334 vs.
0.0100) and Hits@10 (0.0622 vs. 0.0161) under the pre-registered decision
rule, though absolute performance on the full 7,957-drug universe remains
weak (the true drug lands in the top 10 for roughly 6 in 100 held-out
pairs); this does not establish clinical efficacy for any candidate.

## 2. Locked-test comparison table

Same four candidate rankers already in `README.md`'s "Final TransE result
(locked test)" table, extended with the R-GCN row and a macro AP column once
it's computed. The path/popularity/TransE values below are **copied
verbatim** from the existing locked-test table -- no new locked-test
evaluation is run to produce this file.

The R-GCN row was filled mechanically from `ml/experiments/rgcn_final_locked_test_ca9313b1.json`
(`ml/scripts/fill_final_results.py`), not by hand.

| | Hits@1 | Hits@3 | Hits@10 | MRR | Macro AP |
|---|---|---|---|---|---|
| Path baseline (bounded candidates) | 0.01004 | 0.04016 | 0.08635 | 0.03329 | not computed¹ |
| Popularity baseline (full universe, *approximate*) | 0.0 | ~0.004 | ~0.014 | ~0.0062 | not computed¹ |
| Final TransE (full universe) | 0.0020 | 0.0040 | 0.0161 | 0.0100 | not computed¹ |
| **Final R-GCN (full universe)** | 0.0120 | 0.0281 | 0.0622 | 0.0334 | 0.0332 |

¹ Macro AP was computed for R-GCN's validation-search selection rule and for the final R-GCN locked-test
run. It was not computed against the locked test for the path, popularity and TransE rows, and it is left
blank rather than backfilled: doing so would mean re-reading the locked-test labels for models whose one
evaluation has already run.

**Checkpoint provenance.** The R-GCN was trained on Google Colab (T4). The trained file's SHA-256 is
`8d5d52417035659f78e394f97fef59c96ae7979fc55171010ddb2b0ccf7987eb` (recorded in its `colab_provenance.json`).
The evaluation ran on a copy of it, SHA-256 `ca9313b1aaa4666c11c09b75063735b741efcfcdef67a5abf8c3b9ac14e1d35f`,
that differs only in one restored field, `locked_test_manifest_fingerprint`: the training launcher refuses an
export carrying that value, so the trained file could not record it and the locked-test eligibility gate would
have refused every manifest. The value was restored from the original local export and checked against the one
the TransE result recorded (`docs/POST_RGCN_COMMANDS.md`, `scripts/post_checkpoint.py stamp`). Weights and config are unchanged.

### Candidate-universe caveat (restate, don't re-derive)

Copy README's existing caveat verbatim: the path baseline ranks a bounded,
per-disease discovered candidate set; popularity, TransE and R-GCN all rank
the full 7,957-drug universe. All four share the same held-out pairs and
metric formulas, but this is not the same ranking task across rows -- no
"R-GCN is N times better/worse than path" claim should be made from this
table alone.

## 3. Statistical significance -- method frozen, not computed (see below)

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
**Result: not computed.** The paired bootstrap above needs per-disease
outcomes for both R-GCN and TransE on the identical resampled diseases.
TransE's own locked-test run (`ml/experiments/transe_stage_d_final_locked_test_c8997e9.json`)
predates the `--per-disease-output` flag and saved only aggregate metrics,
so no per-disease breakdown exists for it. Re-running TransE's locked-test
evaluation to capture that breakdown was not authorised in this closure (the
locked test is run at most once per model, and that run already happened,
months ago). So no significance interval is reported here, and the
comparison with TransE in `§4` rests on the point estimates in `§2` alone,
over 498 pairs from 100 diseases.

## 4. Interpretation

R-GCN scored above TransE on both MRR (0.0334 vs. 0.0100) and Hits@10 (0.0622
vs. 0.0161), so by the pre-registered rule (`§3`'s decision rule in
`ml/RGCN_EVALUATION_PROTOCOL.md`: an improvement requires both metrics
higher) it counts as an improvement over TransE on this split. No
significance interval is available for that difference (`§3`): the
comparison rests on these point estimates alone. Absolute performance is
still weak: on the full 7,957-drug universe, the true drug lands in the top
10 for about 6 in 100 held-out pairs (Hits@10 = 0.0622), and in the top 1 for
about 1 in 100 (Hits@1 = 0.0120). Against the path baseline, R-GCN's Hits@10
(0.0622) is below the path baseline's (0.0863) and its MRR is about equal (0.0334
vs. 0.0333); because the path baseline ranks a smaller candidate set (`§2`'s
caveat), no ranking between them is claimed. This does not establish clinical
efficacy for any candidate drug.

## 5. Case studies

Three masked case studies (known indication removed from the message-passing
graph at inference, re-encoded, then ranked against the full drug universe;
weights not retrained):

| Disease | Known drug | Rank |
|---|---|---|
| Breast neoplasm (35771) | Doxorubicin (14275) | [20](../case_studies/disease-35771.json) |
| Hereditary breast carcinoma (28347) | Bevacizumab (16510) | [151](../case_studies/disease-28347.json) |
| Breast cancer (36691) | Bevacizumab (16510) | [190](../case_studies/disease-36691.json) |

All three diseases are fit/validation diseases (not locked-test), chosen by
`scripts/pick_case_studies.py`'s fixed, model-blind rule before any model
output was examined. See `docs/CASE_STUDY_TEMPLATE.md`'s gating note: none
of these ranks establishes clinical efficacy for the candidate drug.

## Owner sign-off before publishing

- [x] Locked-test evaluation for R-GCN has actually run, exactly once, under
      `assert_eligible_for_locked_test_evaluation`'s checks.
- [x] The result in this file matches the experiment JSON artifact's numbers
      exactly (copy-paste, not retyped).
- [x] Significance section (`§3`) is filled in -- in this case, with the
      reason it could not be computed, per the frozen method's own
      requirements, not a different method chosen after seeing the result.
- [ ] `docs/PORTFOLIO.md`'s "Results placeholder" section is updated in the
      same commit as this file, so the two never disagree about what's known.
