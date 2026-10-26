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
from dataclasses import replace
from pathlib import Path

import torch

from repurposemap.embeddings.export import load_export
from repurposemap.embeddings.rgcn_sampled_train_gpu import (
    SEARCH_CONFIGS,
    run_sampled_validation_search,
    train_sampled_final,
)
from repurposemap.embeddings.checkpointing import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["search", "final"])
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="persisted dir for state, checkpoints, logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--search-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None, help="final only: override epochs (default: best_val_epoch + 1)")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available; refusing to run a ~47h job on CPU")

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    max_wall_s = args.max_wall_hours * 3600 if args.max_wall_hours is not None else None
    loaded = load_export(args.export_dir)

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
