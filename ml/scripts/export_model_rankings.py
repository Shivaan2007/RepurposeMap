"""Post-model-training glue: the missing link between a final, approved
embedding checkpoint and ``data/serving/*.json``.

Pipeline this script is one step of (see ``docs/DEPLOYMENT.md``'s "Serving
strategy" and ``API.md``'s "Precomputed ranking serving"):

    final checkpoint
      -> per-disease ranking output          <- THIS SCRIPT
      -> validated CandidateRankingBundle    <- repurposemap.serving.export_rankings.build_bundle (reused, not duplicated)
      -> data/serving/*.json                 <- this script writes here directly
      -> API loader                          <- repurposemap.serving.loader (unchanged)
      -> frontend display                    <- unchanged

This script does pure inference, nothing else: for each requested disease,
score every drug in the training export's full drug universe against it
(reusing the exact same ``score_fn``/full-universe-argsort pattern
``repurposemap.cli``'s ``rgcn-evaluate``/``transe-evaluate`` already use for
metric computation), keep the top ``--top-k``, and hand the result to
``export_rankings.build_bundle`` for validation and ``data/serving/``.

It never computes, re-ranks, invents, or adjusts a score -- the model's own
``.score()``/``.encode()`` call is the only thing that produces numbers
here. It never reads ``locked_test_manifest.json`` -- there is no argument
for one, and nothing in this file imports
``load_locked_test_manifest``/``assert_eligible_for_locked_test_evaluation``.
Generating a serving bundle is inference over an arbitrary disease list the
owner chooses, not a locked-test evaluation.

Hard gate: refuses to run against any checkpoint whose
``checkpoint["is_final_training"]`` is not ``True``. This is mechanical, not
just a naming convention -- see ``ml/src/repurposemap/embeddings/checkpointing.py``.
This script cannot and does not verify the second, separate approval this
project requires before a checkpoint is "the" final model: that its
locked-test evaluation has actually been run (once) and the result has been
reviewed and approved by the owner, per
``ml/RGCN_EVALUATION_PROTOCOL.md``/``docs/FINAL_RESULTS.md``'s sign-off
checklist. Run this only after that approval, not before.

ID scheme note (read before running): ``--disease-id`` takes PrimeKG's own
internal node-index ID (a plain string like ``"38436"``) -- the same ID
``EntityIndex``/``LoadedExport.drug_ids``/``LoadedExport.test_disease_ids``
already use throughout the embeddings/evaluation pipeline, found via
``python -m repurposemap search --source primekg --query "..."``. This is
*not* the same ID scheme as the MONDO/EFO IDs
``/api/v1/disease-biology``/``/api/v1/target-scorecard`` use (those come
from Open Targets, a separate system) -- ``export_rankings.py``'s own
docstring example (``MONDO:0005015``) is illustrative only, not what this
pipeline's checkpoints actually key on. If the owner wants the serving
bundle's ``disease_id`` to be a MONDO-style ID instead (e.g. so the
frontend can share one ID across the ranking and disease-biology panels),
that join (PrimeKG node's ``external_id`` attribute, set by
``repurposemap.adapters.primekg.load_primekg``) is a deliberate decision to
make explicitly at that point, not something this script guesses at.

Usage::

    python ml/scripts/export_model_rankings.py \\
        --export-dir ml/data/transe_export \\
        --checkpoint final_rgcn_checkpoint.pt \\
        --model rgcn \\
        --disease-id 38436 --disease-id 1082 \\
        --top-k 10 \\
        --ranking-method rgcn \\
        --source-experiment-file ml/experiments/rgcn_final_locked_test_<commit>.json \\
        --output-dir data/serving/

Add ``--primekg-csv PATH`` if ``data/raw/primekg/kg.csv`` isn't at the
default location (``repurposemap.adapters.DEFAULT_PRIMEKG_CSV``) -- it is
only read for disease/drug display names, never for training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from repurposemap.adapters import DEFAULT_PRIMEKG_CSV, load_primekg
from repurposemap.embeddings.export import load_export
from repurposemap.embeddings.rgcn_model import RGCN
from repurposemap.embeddings.rgcn_train_gpu import build_message_passing_edges
from repurposemap.embeddings.transe_model import TransE
from repurposemap.embeddings.transe_train_gpu import load_checkpoint
from repurposemap.evaluation.artifact import get_git_commit
from repurposemap.serving.export_rankings import build_bundle
from repurposemap.serving.schemas import slugify_disease_id


class FinalModelRequiredError(RuntimeError):
    """Raised when ``--checkpoint`` is not marked ``is_final_training``.

    Generating a real serving bundle from an in-progress search/validation
    checkpoint would produce a bundle that looks real but isn't the
    approved result -- refused outright, never a warning to ignore."""


def _build_score_fn(model_name: str, checkpoint: dict, loaded):
    """Returns ``score_fn(disease_id, drug_ids) -> np.ndarray``, matching
    exactly the scoring code ``repurposemap.cli``'s ``_run_rgcn_evaluate``/
    ``_run_transe_evaluate`` already use -- not reimplemented differently
    here, so inference-for-serving and inference-for-metrics can never
    silently diverge."""

    n_entities = len(loaded.entity_index)
    n_relations = len(loaded.relation_index)

    if model_name == "transe":
        model = TransE(
            n_entities, n_relations,
            embedding_dim=checkpoint["config"]["embedding_dim"], distance=checkpoint["config"]["distance"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        def score_fn(disease_id, drug_ids):
            disease_idx = loaded.entity_index.id_to_index[disease_id]
            drug_indices = torch.as_tensor([loaded.entity_index.id_to_index[d] for d in drug_ids], dtype=torch.long)
            rel = torch.full_like(drug_indices, loaded.indication_relation_idx)
            tail = torch.full_like(drug_indices, disease_idx)
            with torch.no_grad():
                return model.score(drug_indices, rel, tail).numpy()

        return score_fn

    if model_name == "rgcn":
        model = RGCN(
            n_entities, n_relations,
            hidden_dim=checkpoint["config"]["hidden_dim"], num_bases=checkpoint["config"]["num_bases"],
            num_layers=checkpoint["config"]["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        edge_h_np, edge_r_np, edge_t_np = build_message_passing_edges(loaded.final_triples, n_relations)
        with torch.no_grad():
            entity_embeddings = model.encode(
                torch.as_tensor(edge_h_np, dtype=torch.long), torch.as_tensor(edge_r_np, dtype=torch.long),
                torch.as_tensor(edge_t_np, dtype=torch.long), n_entities,
            )

        def score_fn(disease_id, drug_ids):
            disease_idx = loaded.entity_index.id_to_index[disease_id]
            drug_indices = torch.as_tensor([loaded.entity_index.id_to_index[d] for d in drug_ids], dtype=torch.long)
            rel = torch.full_like(drug_indices, loaded.indication_relation_idx)
            tail = torch.full_like(drug_indices, disease_idx)
            with torch.no_grad():
                return model.score(entity_embeddings, drug_indices, rel, tail).numpy()

        return score_fn

    raise ValueError(f"unknown --model {model_name!r}, expected 'transe' or 'rgcn'")


def rank_disease(
    disease_id: str,
    *,
    score_fn,
    drug_ids: tuple[str, ...],
    drug_names: dict[str, str],
    top_k: int,
) -> list[dict]:
    """The full-universe ranking for one disease, truncated to ``top_k`` --
    same ``argsort(-scores)`` full-universe ordering
    ``evaluate_transe_on_diseases`` already uses, just returning the
    ranking itself instead of discarding everything but one true drug's
    rank."""

    scores = np.asarray(score_fn(disease_id, drug_ids))
    if scores.shape != (len(drug_ids),):
        raise ValueError(f"score_fn returned shape {scores.shape}, expected ({len(drug_ids)},)")

    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {
            "drug_id": drug_ids[i],
            "drug_name": drug_names.get(drug_ids[i], drug_ids[i]),
            "rank": rank,
            "score": float(scores[i]),
            "explanation_paths": [],
        }
        for rank, i in enumerate(order, start=1)
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["transe", "rgcn"], required=True)
    parser.add_argument("--primekg-csv", type=Path, default=DEFAULT_PRIMEKG_CSV)
    parser.add_argument(
        "--disease-id", dest="disease_ids", action="append", required=True,
        help="PrimeKG node-index ID of a disease to generate a ranking for -- repeatable",
    )
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--ranking-method", required=True, help='e.g. "rgcn" or "transe" -- stored in the bundle')
    parser.add_argument("--source-experiment-file", default=None)
    parser.add_argument("--source-commit", default=None, help="defaults to the current git commit if omitted")
    parser.add_argument("--output-dir", type=Path, default=Path("data/serving"))
    parser.add_argument(
        "--raw-ranking-output-dir", type=Path, default=None,
        help="also write each disease's raw pre-bundle ranking list here, for an audit trail",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if not checkpoint.get("is_final_training"):
        raise FinalModelRequiredError(
            f"{args.checkpoint} is not marked is_final_training=True -- refusing to export a serving bundle "
            "from a non-final checkpoint. Only run this against the approved final checkpoint."
        )

    loaded = load_export(args.export_dir)
    score_fn = _build_score_fn(args.model, checkpoint, loaded)

    graph = load_primekg(args.primekg_csv)
    drug_names = {drug_id: graph.nodes[drug_id]["name"] for drug_id in loaded.drug_ids if drug_id in graph.nodes}

    source_commit = args.source_commit
    if source_commit is None:
        source_commit, _is_dirty = get_git_commit()

    generated_at = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_ranking_output_dir is not None:
        args.raw_ranking_output_dir.mkdir(parents=True, exist_ok=True)

    for disease_id in args.disease_ids:
        if disease_id not in loaded.entity_index.id_to_index:
            raise ValueError(f"disease_id {disease_id!r} is not in this export's entity index")
        if disease_id in graph.nodes and graph.nodes[disease_id].get("node_type") != "disease":
            raise ValueError(
                f"disease_id {disease_id!r} has node_type={graph.nodes[disease_id].get('node_type')!r} in "
                "PrimeKG, not 'disease' -- check this is the right ID"
            )
        disease_name = graph.nodes[disease_id]["name"] if disease_id in graph.nodes else disease_id

        candidates = rank_disease(
            disease_id, score_fn=score_fn, drug_ids=loaded.drug_ids, drug_names=drug_names, top_k=args.top_k
        )

        if args.raw_ranking_output_dir is not None:
            raw_path = args.raw_ranking_output_dir / f"{slugify_disease_id(disease_id)}.json"
            raw_path.write_text(json.dumps(candidates, indent=2) + "\n")

        bundle = build_bundle(
            disease_id=disease_id,
            disease_name=disease_name,
            ranking_method=args.ranking_method,
            candidate_universe_description=(
                f"full PrimeKG drug universe used by {args.model} training ({len(loaded.drug_ids)} drugs)"
            ),
            candidate_universe_size=len(loaded.drug_ids),
            candidate_records=candidates,
            top_k=args.top_k,
            source_experiment_file=args.source_experiment_file,
            source_commit=source_commit,
            generated_at=generated_at,
        )
        output_path = args.output_dir / f"{slugify_disease_id(disease_id)}.json"
        output_path.write_text(json.dumps(bundle.model_dump(), indent=2) + "\n")
        print(f"Wrote {len(bundle.candidates)} candidate(s) for {disease_name!r} ({disease_id}) to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
