# Sampled R-GCN: evaluation protocol (written before any R-GCN result exists)

**Status: approved and frozen by the project owner (recorded 2026-10-26).** The scientific choices below are not changed after this point. A change needs a new dated section with a stated reason, and only for a bug that makes the protocol impossible to execute.

## Why this exists

`ml/experiments/transe_stage_d_final_locked_test_c8997e9.json` records that the locked test is **already spent** on TransE and must not be used for tuning or selection of any later model. R-GCN may be scored on it once, as a final comparison under rules fixed in advance. These are those rules.

## Fixed in advance

- **Data:** the same export as TransE (`export_fingerprint` `8e41f866f9fb90b0017b9056c1eb69b585004e01e65e6c6e4d17b9dbf16cb1aa`, PrimeKG sha256 `0a5a010f…5c69`). If the rebuilt export's fingerprint differs, stop: the comparison is invalid.
- **Model selection:** validation only, using the existing rule (`SELECTION_RULE_DESCRIPTION` in `transe_baseline.py`): highest validation MRR; ties within the threshold go to macro AP, then to smaller dimension. The 3 configs are `SEARCH_CONFIGS` in `rgcn_sampled_train_gpu.py` and are not changed. Fanouts stay 15/10.
- **Search budget:** 3 epochs per config, validating every epoch.
- **Final training:** the winner trained once on `final_triples` for `best_val_epoch + 1` epochs. No early stopping and no second run.
- **Evaluation:** `rgcn-evaluate --split locked-test`, run **once**, on the 100 test diseases / 498 pairs, ranking the full 7,957-drug universe. The checkpoint must be complete (the runner prints `COMPLETE`) and pass `assert_eligible_for_locked_test_evaluation`.
- **No post-test changes:** whatever the result, no retraining, retuning, or re-selection. The result is reported as it comes out.

## What will be reported

Hits@1, Hits@3, Hits@10, MRR, macro AP, micro AP for R-GCN, next to the recorded TransE, popularity and path-baseline numbers, with the file's `fairness_caveat` attached every time the path baseline appears (it ranks a bounded candidate set; R-GCN, TransE and popularity rank all 7,957 drugs).

## Decision rule for "R-GCN improved on TransE"

R-GCN counts as an improvement over TransE only if **both** MRR and Hits@10 are higher on the locked test. With 498 pairs from 100 diseases, small gaps are within noise, so a disease-level bootstrap interval (resampling diseases) is reported with the result and the claim is worded accordingly. No claim of beating the path baseline is made from these numbers alone, given the candidate-set difference.

None of this is evidence of clinical usefulness; outputs are research hypotheses.
