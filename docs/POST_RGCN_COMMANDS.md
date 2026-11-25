# Post-R-GCN-checkpoint commands (exact, ordered)

**Nothing in this file has been run.** It is the exact command sequence to
execute once `final_checkpoint.pt` reaches 3/3 epochs (see
`docs/COLAB_RGCN_FINAL_RUNBOOK.md`), verified against this repo's actual
code at commit `b12aecc` on 2026-11-19. It changes no frozen protocol,
config, seed, split, candidate universe, or evaluation rule
(`ml/RGCN_EVALUATION_PROTOCOL.md` / `ml/RGCN_RUNBOOK.md`).

Placeholders in angle brackets (`<...>`) are paths/IDs you fill in; nothing
else should need to change.

## Fast path: `scripts/post_checkpoint.py` (preferred)

One stage per step, same order as below, with the safety checks in code. Tested end
to end on synthetic data (verify, the single-run guard, evaluate, record, fill, export,
case study, serving validation), never on the real checkpoint. `<sha8>` is the first 8
hex digits of the checkpoint's SHA-256, which `verify` prints.

```bash
python scripts/post_checkpoint.py verify   --checkpoint <final.pt> --final-epoch 2 --provenance <colab_provenance.json>
python scripts/post_checkpoint.py evaluate --checkpoint <final.pt> --final-epoch 2 --provenance <colab_provenance.json> \
       --export-dir <export> --manifest <locked_test_manifest.json> --approve-single-locked-test-run
python scripts/post_checkpoint.py fill
python scripts/post_checkpoint.py export   --checkpoint <final.pt> --export-dir <export> --disease-id <id> [--disease-id <id> ...]
python scripts/post_checkpoint.py case-study --checkpoint <final.pt> --export-dir <export> --disease-id <id> --known-drug-id <id> --slug <name>
python scripts/post_checkpoint.py validate-serving
python scripts/post_checkpoint.py qa
```

What it enforces: `evaluate` runs `verify` first; refuses without the approval flag; refuses
if `ml/experiments/rgcn_final_locked_test_*.json` already exists (exactly once); refuses
to restart an attempt that wrote no record unless `--allow-restart-after-incomplete`.
It writes raw outputs to `ml/results/` and one schema-valid, read-only record to
`ml/experiments/rgcn_final_locked_test_<sha8>.json`. Do not put raw artifacts in
`ml/experiments/`: the provenance loader requires every file there to follow the
experiment schema, and a flat metrics file fails `test_real_experiment_files_parse_as_valid`.

Choosing case-study diseases: the known indication must be in the training graph, so
use a **fit or validation** disease. A locked-test disease's indications were held out
of `final_triples`, and `run_case_study.py` stops with "masking removed 0 rows". Exporting
a ranking for a test disease is fine; a case study on one is not possible.

The sections below are the same steps as raw commands, for reference.

## 0. Verify the checkpoint (before anything else)

Two independent checks, both already real code -- neither is new:

**a) The Colab/Kaggle run's own provenance artifact** (already written each
session by the launcher, per `docs/COLAB_RGCN_FINAL_RUNBOOK.md`):

```bash
cat /content/drive/MyDrive/RepurposeMap/rgcn_final/colab_provenance.json
```

Confirm by eye: `"epochs": 3`, `"final_epoch_reached": 2` (0-indexed, so 2 ==
the 3rd/last epoch), `"stopped_for_time_budget": false`,
`"is_final_training": true`, `"selected_name": "rgcn-wide"`,
`"source_commit": "0f23fe3631e276f36a8ba1e481da2beb634b6ccf"`,
`"source_archive_sha256":
"bd26185f5db0ee0f5918936649c9f371d663034fad7bfff63939e3e9d1782085"`, the
`config` block matches the frozen spec exactly, and
`"locked_test_evaluation_performed": false`.

**b) Load the checkpoint itself and print what it actually recorded**
(catches any mismatch between the provenance JSON and the checkpoint file
it describes):

```bash
PYTHONPATH=ml/src python -c "
from repurposemap.embeddings.checkpointing import load_checkpoint
ckpt = load_checkpoint('<final_checkpoint.pt>', map_location='cpu')
print('is_final_training:', ckpt.get('is_final_training'))
print('epoch:', ckpt.get('epoch'))
print('config:', ckpt.get('config'))
print('export_fingerprint:', ckpt.get('export_fingerprint'))
print('locked_test_manifest_fingerprint:', ckpt.get('locked_test_manifest_fingerprint'))
print('git_commit:', ckpt.get('git_commit'))
"
```

**Do not proceed past this point until both checks agree with the frozen
spec.** Step 1's own `--split locked-test` path re-runs these checks
programmatically anyway (`assert_eligible_for_locked_test_evaluation` in
`ml/src/repurposemap/embeddings/export.py`, invoked by `rgcn-evaluate`
before it reads a single score) and refuses to run if they fail -- step 0
here is a human-readable preview of the same gate, not a separate one.

## 1. The one locked-test evaluation -- prepared, NOT run

```bash
python -m repurposemap rgcn-evaluate \
    --export-dir <rgcn export dir, same one used for final training> \
    --checkpoint <final_checkpoint.pt> \
    --split locked-test \
    --manifest <LOCAL-ONLY locked_test_manifest.json> \
    --artifact ml/results/rgcn_final_locked_test_<sha8>_raw.json \
    --per-disease-output ml/results/rgcn_final_locked_test_<sha8>_per_disease.json
```

- `--manifest` is the same local-only file `transe-export` originally wrote
  (never uploaded anywhere); `assert_eligible_for_locked_test_evaluation`
  checks its fingerprint against the checkpoint's recorded expectation and
  refuses on any mismatch.
- `--artifact` is the aggregate-metrics JSON `fill_final_results.py` (step
  3) reads.
- `--per-disease-output` is **new this session** (`ml/src/repurposemap/cli.py`):
  it writes each held-out pair's `(disease_id, drug_id, rank)` to a second
  JSON file -- the input `compute_significance.py` (step 2) needs, since
  the aggregate artifact alone has no per-disease breakdown to resample.
  See "Known gap" below before relying on this for the comparison side.
- Run this **exactly once**, after step 0 passes. Do not rerun it because
  of an unfavorable result (`ml/RGCN_EVALUATION_PROTOCOL.md`).

**Known gap -- read before running step 2:** TransE's own locked-test
evaluation (`ml/experiments/transe_stage_d_final_locked_test_c8997e9.json`)
already ran, months ago, without a `--per-disease-output` flag (it didn't
exist yet) -- so no per-disease breakdown was ever saved for it, only the
aggregate numbers in that file. The frozen significance method (step 2)
needs **paired** per-disease outcomes for *both* R-GCN and TransE. Without
TransE's, step 2 cannot be run against real data yet. This is an owner
decision, not something resolved by this session (see "Blockers" below) --
recapturing it means reading the locked-test manifest again, which this
session was explicitly told not to do.

## 2. Significance: disease-level paired bootstrap (R-GCN vs. TransE)

```bash
python ml/scripts/compute_significance.py \
    --rgcn-outcomes ml/results/rgcn_final_locked_test_<sha8>_per_disease.json \
    --transe-outcomes <TransE's own --per-disease-output file -- see "Known gap" above> \
    --n-resamples 10000 \
    --seed 42 \
    --output ml/results/rgcn_vs_transe_significance_<sha8>.json
```

- New this session: `ml/scripts/compute_significance.py` +
  `ml/src/repurposemap/evaluation/significance.py`. Executes, exactly, the
  method already frozen in `docs/FINAL_RESULTS.md` §3 (resample the 100
  locked-test diseases with replacement, recompute MRR and Hits@10 for
  both models on each resample, report the 95% percentile interval of the
  R-GCN-minus-TransE difference, state plainly whether each interval
  excludes zero) -- it does not choose or adjust that method.
- Refuses to run if either input file's `"split"` isn't `"locked-test"`,
  or if the two files don't cover the identical set of disease IDs.
- Deterministic given `--seed 42` (same convention as the rest of this
  repo's frozen seed).
- Blocked on the "Known gap" in step 1 until TransE's per-disease outcomes
  exist. 10,000 resamples is cheap (pure NumPy, no GPU, no retraining) and
  runs in well under a second once both input files exist.

## 3. Fill in `docs/FINAL_RESULTS.md`

```bash
python ml/scripts/fill_final_results.py \
    --artifact ml/experiments/rgcn_final_locked_test_<sha8>.json   # the frozen record, not the raw file
```

- Existing script, unchanged this session. Replaces the exact R-GCN `TBD`
  row in `docs/FINAL_RESULTS.md`'s comparison table with that artifact's
  real `hits_at_1/3/10`, `mrr`, `macro_auprc` -- refuses if the artifact's
  `split` isn't `"locked-test"`, or if the exact TBD row isn't found
  (already filled in, or the table changed shape).
- Everything else in that file (§1 one-line summary, §3's significance
  *result* once step 2 produces one, §4 interpretation, §5 case-study
  links, the sign-off checklist) is a narrative judgment call for a
  person -- this script inserts numbers only.

## 4. Export real serving rankings (pick 3–5 diseases)

```bash
python ml/scripts/export_model_rankings.py \
    --export-dir <rgcn export dir> \
    --checkpoint <final_checkpoint.pt> \
    --model rgcn \
    --disease-id <id1> [--disease-id <id2> ...] \
    --top-k 10 \
    --ranking-method rgcn \
    --source-experiment-file ml/experiments/rgcn_final_locked_test_<sha8>.json \
    --output-dir data/serving/
```

- Existing script, unchanged this session. `--disease-id` is repeatable
  (one ranking per disease, same command). Refuses to run unless
  `--checkpoint`'s `is_final_training` is `true`.
- `--primekg-csv` defaults to `data/raw/primekg/kg.csv` if omitted (needed
  to resolve disease/drug names into the serving bundle).
- Optional `--raw-ranking-output-dir PATH` also writes each disease's raw
  pre-bundle ranking list, for an audit trail.
- After this, `GET /api/v1/rank/diseases` and `POST /api/v1/rank` on a
  running backend will serve these diseases for real
  (`serving_mode: "precomputed"`) instead of `model_status:
  "not_available"`.

## 5. Run masked case studies (pick 2–3 pairs from step 4's diseases)

```bash
python ml/scripts/run_case_study.py \
    --export-dir <rgcn export dir> \
    --checkpoint <final_checkpoint.pt> \
    --model rgcn \
    --disease-id <id> \
    --known-drug-id <id> \
    --ranking-method rgcn \
    --output case_studies/<slug>.json
```

- Existing script, unchanged this session. Reuses
  `evaluation.case_study`'s graph-level masking unchanged, plus an
  R-GCN-specific training-triples-level mask and re-encode (message-passing
  embeddings depend on which edges the model sees) -- read that script's
  own module docstring before using.
- Write each one up using `docs/CASE_STUDY_TEMPLATE.md`'s structure
  exactly; no clinical-efficacy claim, per that template's gating note.

---

## Fixes made this session (commands/paths/documentation only)

- Added `--per-disease-output PATH` to both `rgcn-evaluate` and
  `transe-evaluate` (`ml/src/repurposemap/cli.py`): writes each held-out
  pair's `(disease_id, drug_id, rank)` to JSON. Without this, the frozen
  significance method in `docs/FINAL_RESULTS.md` §3 had no way to execute
  at all -- the existing `--artifact` output is aggregate-only.
- Added `ml/src/repurposemap/evaluation/significance.py`
  (`paired_bootstrap_ci`) and `ml/scripts/compute_significance.py`: executes
  the already-frozen disease-level paired bootstrap method. No methodology
  decision was made here -- the method was already specified in
  `docs/FINAL_RESULTS.md` before this session.
- 11 new tests (`ml/tests/test_significance.py`,
  `ml/tests/test_compute_significance.py`, plus one new case each in
  `ml/tests/test_transe_cli.py` / `ml/tests/test_rgcn_cli.py` for the new
  flag), all passing. Full suite: 607 passed, 14 skipped (torch-only,
  unchanged), 0 failed.
- Corrected a real, pre-existing command-invocation bug: `ml/RGCN_RUNBOOK.md`
  and `docs/FINAL_RELEASE_CHECKLIST.md` both documented the locked-test
  command as bare `repurposemap rgcn-evaluate ...`, but no `repurposemap`
  console script exists in `pyproject.toml` (confirmed: `which repurposemap`
  finds nothing) -- the only working invocation is `python -m repurposemap
  ...`, per `README.md`'s own documented usage. Fixed in both files.
- No locked-test evaluation was run. No model, hyperparameter, config,
  seed, split, or evaluation-rule change was made.

## Blockers

- **The final checkpoint doesn't exist yet** -- everything above is
  prepared, not executed, pending the Colab/Kaggle run.
- **TransE per-disease outcomes are missing** (see step 1's "Known gap"):
  the already-spent TransE locked-test run only has aggregate metrics
  saved. The significance test in step 2 cannot run against real data
  until the owner decides how to get TransE's per-disease breakdown --
  options are (a) accept a one-time, read-only, non-retuning rerun of
  `transe-evaluate --split locked-test --per-disease-output PATH` against
  the existing TransE checkpoint (same deterministic inputs, same
  aggregate result, just also capturing the breakdown this time), or (b)
  report the R-GCN locked-test result without the TransE comparison's
  statistical-significance interval. Either way, this is a decision for
  the owner to make explicitly -- not something this session executed,
  since it was told not to touch the locked test at all.
- Choosing the 3–5 diseases (step 4) and 2–3 case-study pairs (step 5) is
  an editorial/scientific judgment call, not mechanical.
- Deployment (Render/Vercel) is intentionally out of scope for this
  document -- see `docs/FINAL_RELEASE_CHECKLIST.md` steps 5–8 for that,
  which come after everything above.
