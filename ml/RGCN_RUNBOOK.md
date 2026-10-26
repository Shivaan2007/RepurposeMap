# Sampled R-GCN: GPU run order

Runner: `ml/kaggle/rgcn_sampled_kaggle_runner.py`. Full usage is in its docstring. Rerun the same command every session; state lives in `--work-dir`, which must persist between sessions.

1. **Export** (local, needs the ~980 MB `kg.csv`): build the R-GCN export, upload it as a dataset (`rgcn-export`).
2. **`smoke`** (GPU, minutes): 1 epoch on a subset, stop on a zero budget, resume, with a validation pass. Check `smoke_report.json`: `resume_ok` must be true; `estimated_full_epoch_s` should be near the profiled ~4.7 h and memory must fit. Do not launch the search if it fails.
3. **`search`** (GPU, ~3 configs x 3 epochs x ~4.7 h, many sessions): rerun until `status: complete`. Keep `--max-wall-hours` at least one epoch (~5 h) below the session limit.
4. **`final`** (GPU): trains once on `final_triples` with the winner; rerun until `stopped_for_time_budget: False`.
5. **Locked-test evaluation** (local, exactly once): `repurposemap rgcn-evaluate --split locked-test`.

## Protocol caveat (read before step 5)

`ml/experiments/transe_stage_d_final_locked_test_c8997e9.json` records that the locked test is **already spent** on TransE (`locked_test_is_now_spent: true`) and "must not be used as an iterative tuning or model-selection set for any future model (including R-GCN)". Select R-GCN on validation only, precommit the comparison protocol before looking at the locked test, and report it as a single final comparison.

Context for the decision to run R-GCN: TransE locked-test Hits@10 was 0.0161 vs 0.0863 for the path baseline (different candidate universes; see that file's `fairness_caveat`), so R-GCN is an attempt to use PrimeKG's relational structure that TransE did not.

Status: the runner has not been executed on a GPU; `smoke` is the first real test of it.
