"""Thin Kaggle/Colab runner for TransE training (Stage D).

Not a source of business logic -- every real decision (what the 3 search
configs are, the selection rule, the training loop, checkpoint format)
lives in ``repurposemap.embeddings``/``repurposemap.evaluation`` and is
imported here, not reimplemented. This script only wires those functions
together for a GPU notebook session and prints a profiling report.

Usage (as a Kaggle/Colab notebook cell, or `python transe_kaggle_runner.py ...`):

    # 1. Install the package (torch is normally preinstalled on both
    #    platforms; this just gets `repurposemap` importable).
    #    !pip install -e /kaggle/input/repurposemap-src   (or equivalent)

    # 2. Profile only (the default -- always safe to run first):
    python transe_kaggle_runner.py --export-dir /kaggle/input/transe-export

    # 3. The real 3-config search (only once profiling looks sane):
    python transe_kaggle_runner.py --export-dir /kaggle/input/transe-export \\
        --run-search --search-log-out /kaggle/working/search_log.json \\
        --device cuda

    # 4. Final training, after inspecting the search log:
    python transe_kaggle_runner.py --export-dir /kaggle/input/transe-export \\
        --run-final --search-log /kaggle/working/search_log.json \\
        --final-checkpoint-out /kaggle/working/final_checkpoint.pt --device cuda

This script never imports ``repurposemap.embeddings.export``'s
``load_locked_test_manifest`` or references a locked-test manifest path at
all -- there is nothing here that could embed a real test label even by
mistake. Download ``final_checkpoint.pt`` back to your own machine and run
the exactly-once locked-test evaluation there
(``repurposemap transe-evaluate --split locked-test --manifest ...``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from repurposemap.embeddings.export import load_export
from repurposemap.embeddings.positive_store import build_positive_store
from repurposemap.embeddings.transe_train_gpu import (
    GPUTrainingConfig,
    run_validation_search,
    train_final,
)


def profile(export_dir: Path, *, subset_size: int, device: str) -> dict:
    """Train a handful of batches on a bounded real-triple subset -- plumbing
    validation, explicitly NOT a performance benchmark (the subset and
    epoch count are both far too small to represent the real run)."""
    loaded = load_export(export_dir)
    tuning_triples = loaded.tuning_triples
    rng = np.random.default_rng(0)
    size = min(subset_size, len(tuning_triples))
    subset = tuning_triples[rng.choice(len(tuning_triples), size=size, replace=False)]

    n_entities = len(loaded.entity_index)
    n_relations = len(loaded.relation_index)
    positive_store = build_positive_store(subset, n_relations, n_entities)
    config = GPUTrainingConfig(embedding_dim=64, epochs=1, batch_size=1024, negatives_per_positive=4, device=device)

    started = time.monotonic()
    from repurposemap.embeddings.transe_train_gpu import train_gpu

    result = train_gpu(subset, n_entities, n_relations, loaded.type_pools, positive_store, config)
    elapsed = time.monotonic() - started

    examples_per_sec = size / elapsed if elapsed > 0 else float("inf")
    n_full_triples = len(tuning_triples)
    estimated_full_epoch_s = n_full_triples / examples_per_sec if examples_per_sec > 0 else float("inf")

    report = {
        "subset_size": size,
        "full_tuning_triples": n_full_triples,
        "elapsed_s": elapsed,
        "examples_per_sec": examples_per_sec,
        "batches_per_sec": (size / config.batch_size) / elapsed if elapsed > 0 else float("inf"),
        "estimated_full_epoch_s": estimated_full_epoch_s,
        "train_loss": result.history[0].train_loss if result.history else None,
        "sampler_stats": result.sampler.stats.to_dict(),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        report["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile-subset-size", type=int, default=500_000)
    parser.add_argument("--run-search", action="store_true", help="run the real 3-config search (not just profile)")
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--search-log-out", type=Path, default=None)
    parser.add_argument("--run-final", action="store_true", help="run final training (requires --search-log)")
    parser.add_argument("--search-log", type=Path, default=None)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--final-checkpoint-out", type=Path, default=None)
    args = parser.parse_args()

    print("--- profiling (plumbing validation only, not a benchmark) ---")
    profile_report = profile(args.export_dir, subset_size=args.profile_subset_size, device=args.device)
    print(json.dumps(profile_report, indent=2))

    if not args.run_search and not args.run_final:
        print("\nStopping after profiling (default). Pass --run-search to run the real 3-config search.")
        return

    if args.run_search:
        loaded = load_export(args.export_dir)
        print("\n--- running the 3 predefined validation configs ---")
        log = run_validation_search(
            loaded, epochs=args.search_epochs, validate_every=2,
            output_log_path=args.search_log_out,
        )
        print(f"selected: {log['selected_name']}")
        print(json.dumps(log["candidates"], indent=2))
        if args.search_log_out:
            print(f"search log written to {args.search_log_out}")

    if args.run_final:
        if args.search_log is None or args.final_checkpoint_out is None:
            raise SystemExit("--run-final requires --search-log and --final-checkpoint-out")
        loaded = load_export(args.export_dir)
        search_log = json.loads(args.search_log.read_text())
        from repurposemap.embeddings.transe_train_gpu import SEARCH_CONFIGS

        selected_config = dict(SEARCH_CONFIGS)[search_log["selected_name"]]
        if args.final_epochs is not None:
            epochs = args.final_epochs
        else:
            selected_run = next(r for r in search_log["runs"] if r["name"] == search_log["selected_name"])
            epochs = (selected_run["best_val_epoch"] or 0) + 1

        print(f"\n--- final training: {search_log['selected_name']}, {epochs} epochs ---")
        train_final(
            loaded, selected_config=selected_config, epochs=epochs, search_log=search_log,
            checkpoint_path=args.final_checkpoint_out,
        )
        print(f"final checkpoint written to {args.final_checkpoint_out}")
        print("Download this file and run the locked-test evaluation LOCALLY, exactly once.")


if __name__ == "__main__":
    main()
