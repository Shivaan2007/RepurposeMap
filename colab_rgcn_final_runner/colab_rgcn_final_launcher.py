"""Thin, provider-neutral launcher for the frozen R-GCN final-training run,
for use on Google Colab (or any other CUDA box) instead of Kaggle.

This is operational infrastructure ONLY. It does not reimplement any
training logic -- it imports and calls the exact same functions the Kaggle
final-training kernel (`kaggle_rgcn_final_kernel/repurposemap-rgcn-final-training.py`)
uses: `repurposemap.embeddings.rgcn_sampled_train_gpu.train_sampled_final`,
with the exact same frozen `rgcn-wide` config, the exact same real
`search_log.json` (from the completed validation search, not re-derived or
recomputed), and the exact same 3-epoch count. It never imports or calls
anything related to locked-test evaluation.

Reuses, rather than re-derives, provenance:
  - EXPECTED_SOURCE_COMMIT / EXPECTED_ARCHIVE_SHA256 below are copied
    verbatim from the Kaggle final kernel.
  - The source archive it verifies is the SAME already-verified
    `repurposemap-0f23fe3.tar.gz` bytes used there (sha256 ==
    EXPECTED_ARCHIVE_SHA256), not a fresh `git archive` (a fresh archive's
    gzip bytes differ even for identical content, so it would never match
    a hash recorded from a specific prior archive run; re-deriving it would
    only add risk for no benefit).
  - The search_log.json it reads must be the one real file produced by the
    completed validation search (`kaggle_rgcn_v8_recovery/rgcn_search/search_log.json`),
    not a re-run or hand-edited copy -- this script only reads it and
    fails closed if its shape or selection disagrees with the frozen spec.

Usage (see ../docs/COLAB_RGCN_FINAL_RUNBOOK.md for the exact Colab cells):

    python colab_rgcn_final_launcher.py \\
        --archive-path /content/drive/MyDrive/RepurposeMap/repurposemap-0f23fe3.tar.gz \\
        --export-dir /content/drive/MyDrive/RepurposeMap/rgcn_export \\
        --search-log /content/drive/MyDrive/RepurposeMap/search_log.json \\
        --work-dir /content/drive/MyDrive/RepurposeMap/rgcn_final \\
        --max-wall-hours 11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path

EXPECTED_SOURCE_COMMIT = "0f23fe3631e276f36a8ba1e481da2beb634b6ccf"
EXPECTED_ARCHIVE_SHA256 = "bd26185f5db0ee0f5918936649c9f371d663034fad7bfff63939e3e9d1782085"
SELECTED_NAME = "rgcn-wide"
EXPECTED_FINAL_EPOCHS = 3  # best_val_epoch (2) + 1, from the frozen rule -- verified against
# the real search_log.json below at runtime, never trusted blindly.


def verify_and_extract_archive(archive_path: Path, extract_dir: Path) -> None:
    print(f"--- verifying source archive: {archive_path} ---")
    if not archive_path.is_file():
        raise SystemExit(f"REFUSING TO CONTINUE: archive not found at {archive_path}")
    raw = archive_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    print(f"archive sha256 (actual):   {actual}")
    print(f"archive sha256 (expected): {EXPECTED_ARCHIVE_SHA256}")
    if actual != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("REFUSING TO CONTINUE: archive SHA-256 does not match the frozen expected value.")
    print(f"Archive verified. expected source commit = {EXPECTED_SOURCE_COMMIT}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_dir)  # noqa: S202 -- trusted, hash-verified archive
    print(f"Extracted to {extract_dir}")


def add_repo_to_path(extract_dir: Path) -> None:
    src_dir = extract_dir / "ml" / "src"
    if not src_dir.is_dir():
        raise SystemExit(f"Expected {src_dir} after extraction; not found.")
    sys.path.insert(0, str(src_dir))
    print(f"Added to sys.path: {src_dir}")


def require_cuda() -> None:
    import torch

    print(f"torch version: {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("REFUSING TO CONTINUE: final training requires a GPU; none detected. "
                          "In Colab: Runtime -> Change runtime type -> T4 GPU.")
    print(f"cuda available: True, gpu = {torch.cuda.get_device_name(0)}")


def safety_check(export_dir: Path) -> None:
    """Identical in substance to the Kaggle final kernel's safety_check:
    refuse to continue if anything locked-test-shaped is anywhere near the
    export this run would train on."""
    if not export_dir.is_dir():
        raise SystemExit(f"REFUSING TO CONTINUE: export dir not found at {export_dir}")
    present = sorted(p.name for p in export_dir.glob("*"))
    print(f"Files in export dataset: {present}")
    required = {"triples.npz", "metadata.json", "validation_indications.json"}
    missing = required - set(present)
    if missing:
        raise SystemExit(f"REFUSING TO CONTINUE: export dataset is missing required file(s): {missing}")
    if "locked_test_manifest.json" in present:
        raise SystemExit("REFUSING TO CONTINUE: locked_test_manifest.json is present in the export dataset.")
    metadata = json.loads((export_dir / "metadata.json").read_text())
    if "test_indications" in metadata:
        raise SystemExit("REFUSING TO CONTINUE: metadata.json contains a 'test_indications' key.")
    if metadata.get("locked_test_manifest_fingerprint") is not None:
        raise SystemExit("REFUSING TO CONTINUE: metadata.json carries a non-null locked_test_manifest_fingerprint.")
    print("Safety check PASSED: no locked-test material present in the export dataset.")


def load_and_verify_search_log(search_log_path: Path) -> dict:
    if not search_log_path.is_file():
        raise SystemExit(f"REFUSING TO CONTINUE: search_log.json not found at {search_log_path}. "
                          "Upload the real one from the completed validation search "
                          "(kaggle_rgcn_v8_recovery/rgcn_search/search_log.json); do not hand-edit or re-run it.")
    search_log = json.loads(search_log_path.read_text())
    if search_log.get("status") != "complete":
        raise SystemExit("REFUSING TO CONTINUE: search_log.json status is not 'complete'.")
    if search_log.get("selected_name") != SELECTED_NAME:
        raise SystemExit(
            f"REFUSING TO CONTINUE: search_log.json selected_name={search_log.get('selected_name')!r}, "
            f"expected {SELECTED_NAME!r}. The scientific experiment is frozen; this script will not "
            "train a different model."
        )
    run = next((r for r in search_log["runs"] if r["name"] == SELECTED_NAME), None)
    if run is None or run.get("best_val_epoch") is None:
        raise SystemExit("REFUSING TO CONTINUE: search_log.json has no best_val_epoch for the selected run.")
    epochs = run["best_val_epoch"] + 1
    if epochs != EXPECTED_FINAL_EPOCHS:
        raise SystemExit(
            f"REFUSING TO CONTINUE: derived final epoch count ({epochs}) does not match the frozen "
            f"expected value ({EXPECTED_FINAL_EPOCHS}). Do not override; stop and get this reconciled."
        )
    print(f"search_log.json verified: selected={SELECTED_NAME}, final_epochs={epochs} (best_val_epoch={run['best_val_epoch']})")
    return search_log


def restore_checkpoint_if_present(work_dir: Path) -> dict | None:
    from repurposemap.embeddings.checkpointing import load_checkpoint

    ckpt_path = work_dir / "final_checkpoint.pt"
    if not ckpt_path.exists():
        print("No prior final_checkpoint.pt found -- starting final training fresh.")
        return None
    resume = load_checkpoint(ckpt_path, map_location="cpu")
    print(f"Resuming final training from checkpoint at epoch {resume.get('epoch')}")
    return resume


def write_provenance(work_dir: Path, *, search_log: dict, selected_config, epochs: int,
                      out: dict, elapsed_h: float) -> None:
    ckpt_path = work_dir / "final_checkpoint.pt"
    ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else None
    provenance = {
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "selected_name": SELECTED_NAME,
        "config": {
            "hidden_dim": selected_config.hidden_dim, "num_bases": selected_config.num_bases,
            "num_layers": selected_config.num_layers, "margin": selected_config.margin,
            "learning_rate": selected_config.learning_rate, "batch_size": selected_config.batch_size,
            "negatives_per_positive": selected_config.negatives_per_positive, "seed": selected_config.seed,
            "fanout_hop1": selected_config.fanout_hop1, "fanout_hop2": selected_config.fanout_hop2,
            "early_stopping_patience": selected_config.early_stopping_patience,
        },
        "epochs": epochs,
        "is_final_training": True,
        "training_split": "final_triples",
        "compute_provider": "google_colab",
        "final_epoch_reached": out.get("final_epoch"),
        "stopped_for_time_budget": out.get("stopped_for_time_budget"),
        "this_session_elapsed_hours": elapsed_h,
        "final_checkpoint_sha256": ckpt_sha256,
        "locked_test_evaluation_performed": False,
        "validation_search_rerun": False,
    }
    (work_dir / "colab_provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\n--- provenance ---")
    print(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-path", type=Path, required=True,
                         help="path to the already-verified repurposemap-0f23fe3.tar.gz")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--search-log", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                         help="durable dir (e.g. on mounted Drive) for checkpoints/provenance; must persist across sessions")
    parser.add_argument("--extract-dir", type=Path, default=Path("/content/RepurposeMap"))
    parser.add_argument("--max-wall-hours", type=float, default=11.0,
                         help="stop cleanly after the epoch that crosses this budget, before Colab's own session limit")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    require_cuda()
    verify_and_extract_archive(args.archive_path, args.extract_dir)
    add_repo_to_path(args.extract_dir)
    safety_check(args.export_dir)
    search_log = load_and_verify_search_log(args.search_log)

    from repurposemap.embeddings.export import load_export
    from repurposemap.embeddings.rgcn_sampled_train_gpu import SEARCH_CONFIGS, train_sampled_final

    selected_config = dict(SEARCH_CONFIGS)[SELECTED_NAME]
    run = next(r for r in search_log["runs"] if r["name"] == SELECTED_NAME)
    epochs = run["best_val_epoch"] + 1

    print("\n--- loading export (graph-free) ---")
    loaded = load_export(args.export_dir)
    print(f"n_entities={len(loaded.entity_index)} n_relations={len(loaded.relation_index)} "
          f"n_final_triples={len(loaded.final_triples)}")

    resume_from = restore_checkpoint_if_present(args.work_dir)
    ckpt_path = args.work_dir / "final_checkpoint.pt"

    print(f"\n--- running up to {args.max_wall_hours:.2f}h of final training this session ---")
    t0 = time.monotonic()
    out = train_sampled_final(
        loaded,
        selected_config=selected_config,
        epochs=epochs,
        search_log=search_log,
        checkpoint_path=ckpt_path,
        git_commit=EXPECTED_SOURCE_COMMIT,
        git_dirty=False,
        device="cuda",
        resume_from=resume_from,
        max_wall_time_s=args.max_wall_hours * 3600,
    )
    elapsed_h = (time.monotonic() - t0) / 3600
    print(f"\nSession elapsed: {elapsed_h:.2f}h")
    print(f"final_epoch: {out.get('final_epoch')}")
    print(f"stopped_for_time_budget: {out.get('stopped_for_time_budget')}")

    if out.get("final_epoch", -1) + 1 < epochs:
        print(f"\nINCOMPLETE: trained through epoch {out.get('final_epoch')} of {epochs}. "
              "Re-run this same command next session (it resumes from the checkpoint on Drive).")
    else:
        print(f"\nFINAL TRAINING COMPLETE: {epochs} epochs. final_checkpoint.pt is ready for the "
              "one locked-test evaluation (a separate, explicit, later step -- NOT run by this script).")

    write_provenance(args.work_dir, search_log=search_log, selected_config=selected_config,
                      epochs=epochs, out=out, elapsed_h=elapsed_h)

    print("\nsafety_and_protocol: final_training_performed=True, locked_test_evaluation_performed=False, "
          f"validation_search_rerun=False, selected_name_used={SELECTED_NAME}, epochs_used={epochs}, "
          "compute_provider=google_colab. No locked-test manifest was attached or read.")


if __name__ == "__main__":
    main()
