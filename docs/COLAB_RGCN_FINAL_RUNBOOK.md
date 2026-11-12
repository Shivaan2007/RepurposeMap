# Running the frozen R-GCN final training on Google Colab (Kaggle fallback)

Use this only while the Kaggle weekly GPU quota is exhausted. It runs the
*exact same* frozen `rgcn-wide` final-training experiment described in
`ml/RGCN_RUNBOOK.md` / the Kaggle kernel at
`kaggle_rgcn_final_kernel/repurposemap-rgcn-final-training.py` -- same
source commit, same config, same `final_triples` split, same 3 epochs, same
"never touch the locked test" guarantee. The compute provider is the only
thing that changes.

The actual launcher is `colab_rgcn_final_runner/colab_rgcn_final_launcher.py`.
It reuses `repurposemap.embeddings.rgcn_sampled_train_gpu.train_sampled_final`
unchanged -- this script only verifies integrity, wires up paths, and calls it.

## One-time setup (owner, a few minutes)

1. In Google Drive, create a folder, e.g. `MyDrive/RepurposeMap/`.
2. Upload these three things into it, unchanged:
   - `kaggle_rgcn_final_v5_recovery/repurposemap-0f23fe3.tar.gz` (the
     already sha256-verified source archive -- same bytes the Kaggle kernel
     uses; sha256 `bd26185f5db0ee0f5918936649c9f371d663034fad7bfff63939e3e9d1782085`)
   - `kaggle_rgcn_v8_recovery/rgcn_search/search_log.json` (the real,
     completed validation-search log -- do not edit it)
   - Your R-GCN export dataset directory (the same one attached to the
     Kaggle kernel as `shivaanbhalla/repurposemap-transe-export`: must
     contain `triples.npz`, `metadata.json`, `validation_indications.json`
     and nothing locked-test-shaped) -- upload the whole folder as
     `MyDrive/RepurposeMap/rgcn_export/`.
3. If resuming a prior partial Colab run, also upload that session's
   `final_checkpoint.pt` into `MyDrive/RepurposeMap/rgcn_final/` (the
   launcher's `--work-dir`). Leave it out on a first run.

## Owner steps in Colab (every session)

1. Open a new Colab notebook: https://colab.research.google.com/
2. **Runtime -> Change runtime type -> T4 GPU** (or any available GPU), Save.
3. Run these cells, in order:

```python
# Cell 1 -- mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2 -- get the launcher script itself (the only file not already on Drive).
# Paste its contents, OR upload colab_rgcn_final_runner/colab_rgcn_final_launcher.py
# directly via the Colab file browser into /content/.
```

```python
# Cell 3 -- run it. Safe to re-run verbatim every session: it resumes from
# whatever epoch final_checkpoint.pt on Drive last reached.
!python /content/colab_rgcn_final_launcher.py \
    --archive-path "/content/drive/MyDrive/RepurposeMap/repurposemap-0f23fe3.tar.gz" \
    --export-dir   "/content/drive/MyDrive/RepurposeMap/rgcn_export" \
    --search-log   "/content/drive/MyDrive/RepurposeMap/search_log.json" \
    --work-dir     "/content/drive/MyDrive/RepurposeMap/rgcn_final" \
    --max-wall-hours 11
```

4. Watch the output. It prints, in order: CUDA check, archive sha256
   verification, repo extraction, the locked-test safety check,
   search_log.json verification (must say `selected=rgcn-wide,
   final_epochs=3`), export summary, then per-epoch training progress.
5. At the end it prints either:
   - `INCOMPLETE: trained through epoch N of 3` -- the Colab session hit
     its time limit (or your `--max-wall-hours` budget) partway through.
     **Just reconnect and re-run the same Cell 3** next session; it resumes
     from the checkpoint already saved on Drive. Nothing is lost.
   - `FINAL TRAINING COMPLETE: 3 epochs.` -- done. The checkpoint at
     `MyDrive/RepurposeMap/rgcn_final/final_checkpoint.pt` is ready. Do
     **not** run any locked-test evaluation from Colab -- that remains a
     separate, local, one-time step per the frozen protocol.

Each epoch is checkpointed to Drive as soon as it completes (not just at the
end of the session), and a disconnect mid-epoch simply redoes that epoch
from its start on the next run -- no partial-epoch state is ever treated as
final. A `colab_provenance.json` is also written to `--work-dir` after each
session recording the source commit, archive hash, exact config, epochs
reached, and checkpoint sha256.

## Why this is scientifically identical, not just similar

- Same source bytes: the launcher verifies the uploaded archive's SHA-256
  against the exact value recorded in the frozen spec and the Kaggle
  kernel, then extracts and imports from it -- it does not run a fresh
  `git archive` (which would produce different gzip bytes for identical
  content and could never match a hash pinned from one specific prior run).
- Same selection: the launcher reads the real `search_log.json` produced by
  the completed validation search and refuses to continue if its
  `selected_name` isn't `rgcn-wide` or its derived epoch count isn't 3 --
  it never recomputes or overrides this.
- Same training code: it calls `train_sampled_final` from the extracted,
  hash-verified source tree, the same function the Kaggle kernel calls.
  No training logic lives in the launcher.
- Same safety gate: before touching the export, it runs the same
  locked-test-absence check as the Kaggle kernel (`locked_test_manifest.json`
  must be absent, `metadata.json` must have no `test_indications` key and a
  null `locked_test_manifest_fingerprint`) and refuses to continue otherwise.
- Only difference: the GPU vendor/runtime (Colab's T4 vs Kaggle's T4) and
  whatever exact torch build each platform ships. Neither is a scientific
  parameter in the frozen config.

## Returning to Kaggle later

Nothing about the Kaggle kernel, its dataset, or its kernel-metadata is
touched by any of this. If the weekly Kaggle GPU quota resets before Colab
finishes, the Kaggle kernel remains a valid way to resume the same run from
whichever `final_checkpoint.pt` is most recently complete -- but never mix
checkpoints between a Colab-trained epoch and a Kaggle-trained epoch unless
you've confirmed both ran the identical verified source/config (which this
launcher ensures). If in doubt, finish the run on one provider rather than
hopping between them mid-run.
