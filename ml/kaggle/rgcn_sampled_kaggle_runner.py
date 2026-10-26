"""Thin Kaggle/Colab runner for the sampled R-GCN validation search and final training.

No business logic lives here: the 3 configs, selection rule, training loop
and checkpoint format all come from ``repurposemap.embeddings`` /
``repurposemap.evaluation``. This only wires them together for a GPU
session. Never touches a locked-test manifest.

Each epoch is ~4.7h on real PrimeKG, so the 3-config search spans many
sessions. Run THE SAME COMMAND every session; it resumes from
``--search-state`` and ``--checkpoint-dir`` (persist them between sessions,
e.g. as a Kaggle dataset or Drive folder). ``--max-wall-hours`` stops
cleanly after the epoch that crosses the budget, before the session's hard
limit.

    # Smoke test FIRST (minutes, not hours; plumbing check, not a benchmark):
    python rgcn_sampled_kaggle_runner.py smoke \\
        --export-dir /kaggle/input/rgcn-export --work-dir /kaggle/working/smoke \\
        --device cuda

    # Search (repeat each session until it prints status: complete):
    python rgcn_sampled_kaggle_runner.py search \\
        --export-dir /kaggle/input/rgcn-export --work-dir /kaggle/working \\
        --device cuda --max-wall-hours 10

    # Final training with the winning config (repeat until not stopped_for_time_budget):
    python rgcn_sampled_kaggle_runner.py final \\
        --export-dir /kaggle/input/rgcn-export --work-dir /kaggle/working \\
        --device cuda --max-wall-hours 10 [--epochs N]

Download ``final_checkpoint.pt`` and run the locked-test evaluation locally.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from repurposemap.embeddings.export import load_export
from repurposemap.embeddings.positive_store import build_positive_store
from repurposemap.embeddings.rgcn_sampling import build_relation_adjacency
from repurposemap.embeddings.rgcn_sampled_train_gpu import (
    SEARCH_CONFIGS,
    train_sampled_gpu,
    run_sampled_validation_search,
    train_sampled_final,
)
from repurposemap.embeddings.checkpointing import load_checkpoint


def smoke(loaded, work: Path, *, device: str, subset_size: int) -> dict:
    """Train 1 epoch on a bounded triple subset, then stop on a zero time
    budget and resume from the written checkpoint. Plumbing validation only
    (training, checkpoint write, cross-process-style resume, memory/speed
    report); NOT a performance benchmark and never touches test data."""
    work.mkdir(parents=True, exist_ok=True)
    triples = loaded.tuning_triples
    rng = np.random.default_rng(0)
    size = min(subset_size, len(triples))
    subset = triples[rng.choice(len(triples), size=size, replace=False)]
    n_entities, n_relations = len(loaded.entity_index), len(loaded.relation_index)
    store = build_positive_store(subset, n_relations, n_entities)
    adjacency = build_relation_adjacency(subset)
    base = replace(SEARCH_CONFIGS[0][1], device=device, epochs=2, validate_every=1)
    ckpt = work / "smoke.pt"

    def run(resume_from, wall):
        return train_sampled_gpu(
            subset, n_entities, n_relations, loaded.type_pools, store, adjacency, base,
            resume_from=resume_from, checkpoint_path=ckpt, max_wall_time_s=wall,
            entity_index=loaded.entity_index, indication_relation_idx=loaded.indication_relation_idx,
            val_disease_ids=loaded.val_disease_ids, val_indications=loaded.val_indications,
            drug_ids=loaded.drug_ids,
        )

    t0 = time.monotonic()
    first = run(None, 0.0)  # budget already spent -> must stop after epoch 1
    first_s = time.monotonic() - t0
    resume = load_checkpoint(ckpt, map_location="cpu")  # same CPU load the search uses
    second = run(resume, None)
    report = {
        "subset_size": size, "full_tuning_triples": len(triples),
        "first_epoch_s": first_s,
        "estimated_full_epoch_s": first_s * len(triples) / size,
        "stopped_for_time_budget_first": first.stopped_for_time_budget,
        "resumed_final_epoch": second.final_epoch,
        "resume_ok": second.final_epoch == 1,
        "device": device,
    }
    if torch.cuda.is_available():
        report["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    (work / "smoke_report.json").write_text(json.dumps(report, indent=2))
    return report


def final_status(out: dict, epochs: int) -> str:
    """The checkpoint is written with is_final_training=True even when training
    stopped early on the time budget, so it would pass the locked-test
    eligibility gate. Say loudly when it is not actually finished. Finished means
    every epoch ran; hitting the budget on the very last epoch is still finished."""

    if out["final_epoch"] + 1 < epochs:
        return (f"INCOMPLETE: trained through epoch {out['final_epoch']} of {epochs}. "
                "Rerun this command; do NOT run the locked-test evaluation on this checkpoint yet.")
    return f"COMPLETE: {epochs} epochs. final_checkpoint.pt is ready for the one locked-test evaluation."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["smoke", "search", "final"])
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="persisted dir for state, checkpoints, logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--smoke-subset", type=int, default=200_000)
    parser.add_argument("--search-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None, help="final only: override epochs (default: best_val_epoch + 1)")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available; refusing to run a ~47h job on CPU")

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    max_wall_s = args.max_wall_hours * 3600 if args.max_wall_hours is not None else None
    loaded = load_export(args.export_dir)

    if args.mode == "smoke":
        report = smoke(loaded, work, device=args.device, subset_size=args.smoke_subset)
        print(json.dumps(report, indent=2))
        if not report["resume_ok"]:
            raise SystemExit("SMOKE FAILED: resume did not reach epoch 1; do not launch the full search")
        return

    search_log_path = work / "search_log.json"
    if args.mode == "search":
        configs = tuple((n, replace(c, device=args.device)) for n, c in SEARCH_CONFIGS)
        log = run_sampled_validation_search(
            loaded, search_configs=configs, epochs=args.search_epochs, validate_every=1,
            output_log_path=search_log_path, checkpoint_dir=work / "search_checkpoints",
            search_state_path=work / "search_state.json", max_wall_time_s=max_wall_s,
        )
        print(f"status: {log['status']}")
        if log["status"] == "in_progress":
            print(f"current_config: {log['current_config']} (epoch {log['current_epoch']}); rerun this command next session")
        else:
            print(f"selected: {log['selected_name']}")
        return

    if not search_log_path.exists():
        raise SystemExit(f"no search log at {search_log_path}; run `search` first")
    search_log = json.loads(search_log_path.read_text())
    if search_log.get("status") != "complete":
        raise SystemExit("search is not complete; finish `search` first")
    selected = dict(SEARCH_CONFIGS)[search_log["selected_name"]]
    epochs = args.epochs
    if epochs is None:
        run = next(r for r in search_log["runs"] if r["name"] == search_log["selected_name"])
        if run["best_val_epoch"] is None:
            raise SystemExit("no best_val_epoch recorded; pass --epochs")
        epochs = run["best_val_epoch"] + 1

    ckpt_path = work / "final_checkpoint.pt"
    resume = load_checkpoint(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
    out = train_sampled_final(
        loaded, selected_config=selected, epochs=epochs, search_log=search_log,
        checkpoint_path=ckpt_path, device=args.device, resume_from=resume, max_wall_time_s=max_wall_s,
    )
    print(f"stopped_for_time_budget: {out.get('stopped_for_time_budget')}")
    print(final_status(out, epochs))


if __name__ == "__main__":
    main()
