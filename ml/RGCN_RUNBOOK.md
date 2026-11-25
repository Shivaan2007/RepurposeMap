# Sampled R-GCN: GPU run order

Runner: `ml/kaggle/rgcn_sampled_kaggle_runner.py`. Full usage is in its docstring. Rerun the same command every session; state lives in `--work-dir`, which must persist between sessions.

1. **Export** (local, needs the ~980 MB `kg.csv`): build the R-GCN export, upload it as a dataset (`rgcn-export`).
2. **`smoke`** (GPU, minutes): 1 epoch on a subset, stop on a zero budget, resume, with a validation pass. Check `smoke_report.json`: `resume_ok` must be true; `estimated_full_epoch_s` should be near the profiled ~4.7 h and memory must fit. Do not launch the search if it fails.
3. **`search`** (GPU, ~3 configs x 3 epochs x ~4.7 h, many sessions): rerun until `status: complete`. Keep `--max-wall-hours` at least one epoch (~5 h) below the session limit.
4. **`final`** (GPU): trains once on `final_triples` with the winner; rerun until `stopped_for_time_budget: False`.
5. **Locked-test evaluation** (local, exactly once): `python -m repurposemap rgcn-evaluate --split locked-test`.

## If Kaggle's weekly GPU quota is exhausted

Step 4 (final training) can run on Google Colab instead, with no change to
config, seed, split, or epoch count -- see
`docs/COLAB_RGCN_FINAL_RUNBOOK.md` and
`colab_rgcn_final_runner/colab_rgcn_final_launcher.py`. It calls this same
`train_sampled_final` function, just from a Colab-mounted Drive path
instead of a Kaggle dataset path. Kaggle is untouched and remains the
preferred provider once quota resets; never mix a Colab-trained epoch with
a Kaggle-trained one without confirming both ran the identical
hash-verified source/config.

## Protocol caveat (read before step 5)

`ml/experiments/transe_stage_d_final_locked_test_c8997e9.json` records that the locked test is **already spent** on TransE (`locked_test_is_now_spent: true`) and "must not be used as an iterative tuning or model-selection set for any future model (including R-GCN)". Select R-GCN on validation only, precommit the comparison protocol before looking at the locked test, and report it as a single final comparison.

Context for the decision to run R-GCN: TransE locked-test Hits@10 was 0.0161 vs 0.0863 for the path baseline (different candidate universes; see that file's `fairness_caveat`), so R-GCN is an attempt to use PrimeKG's relational structure that TransE did not.

Status: the validation search (steps 1-3) is complete on Kaggle --
`rgcn-wide` selected (see `ml/experiments/rgcn_sampled_validation_search_0f23fe3.json`).
Final training (step 4) has not started; it requires explicit owner
approval and is currently blocked on Kaggle's weekly GPU quota (see the
Colab fallback above). Step 5 remains strictly off-limits until a final,
approved checkpoint exists.
