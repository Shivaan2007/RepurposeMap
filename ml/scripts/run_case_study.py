"""Post-model-training glue: run one real masked case study against the
final, approved embedding checkpoint. Prepared now so that once a final
model exists and is approved, 2-3 real case studies (see
``docs/CASE_STUDY_TEMPLATE.md``) can be run immediately rather than
designed under deadline pressure.

Reuses ``repurposemap.evaluation.case_study`` (``CaseStudySpec``,
``mask_known_indication``, ``confirm_masked``, ``build_case_study_result``,
``MaskingNotConfirmedError``) completely unchanged for the graph-level
masking guarantee that module already has tests for: the known indication
edge is removed from the loaded PrimeKG graph, in both directions, and
confirmed absent before any result is built.

**Why this script also masks at the triples level, not just the graph
level (read before using for R-GCN):** R-GCN's embeddings come from message
passing over ``final_triples`` -- the fixed array baked into the training
export, not from a live graph object (the embeddings pipeline is
deliberately graph-free; see ``docs/ARCHITECTURE.md``). Removing the known
indication edge from the *graph* alone (what
``evaluation.case_study.mask_known_indication`` does) proves nothing about
what the *model* actually saw, because the model never looks at that graph
again at inference time -- it was trained once on ``final_triples``, which
already contains that edge (every non-test indication is in
``final_triples`` by construction). A masked case study that only masks
the graph and then scores with the checkpoint's existing embeddings would
be a procedural masked case study, not an actually honest one: the model's
embeddings would still have been computed *with* that edge in the
message-passing input.

So for R-GCN specifically, this script additionally removes the matching
row(s) from ``final_triples`` and **recomputes** ``model.encode()`` on the
masked triples, using the same already-trained weights (no retraining --
``encode()`` is a forward pass, not a training step). Only the resulting,
honestly-masked embeddings are used to score the case study's candidate
list. TransE has no message-passing/encode step at all -- its embeddings
are fixed per-entity parameters set entirely during training, unaffected by
any triples-level masking here -- so for ``--model transe`` the triples-level
masking step is skipped as a documented no-op, not silently done and
ignored.

This script does not read or need ``locked_test_manifest.json`` -- no
argument for one exists. A case study is not a locked-test evaluation.

Usage::

    python ml/scripts/run_case_study.py \\
        --export-dir ml/data/transe_export \\
        --checkpoint final_rgcn_checkpoint.pt \\
        --model rgcn \\
        --primekg-csv data/raw/primekg/kg.csv \\
        --disease-id 38436 --known-drug-id 1234 \\
        --ranking-method rgcn \\
        --output case_studies/some_disease.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from repurposemap.adapters import DEFAULT_PRIMEKG_CSV, load_primekg
from repurposemap.embeddings.export import load_export
from repurposemap.embeddings.rgcn_model import RGCN
from repurposemap.embeddings.rgcn_train_gpu import build_message_passing_edges
from repurposemap.embeddings.transe_model import TransE
from repurposemap.embeddings.transe_train_gpu import load_checkpoint
from repurposemap.evaluation.case_study import (
    CaseStudySpec,
    build_case_study_result,
    mask_known_indication,
)


class FinalModelRequiredError(RuntimeError):
    """Same gate as ``export_model_rankings.py`` -- refuses any checkpoint
    not marked ``is_final_training``."""


def _mask_triples(final_triples: np.ndarray, *, indication_relation_idx: int, drug_idx: int, disease_idx: int) -> np.ndarray:
    """``final_triples`` with every row matching the known indication
    between ``drug_idx``/``disease_idx`` removed, checked in both
    directions -- the triples-array analog of
    ``evaluation.case_study.mask_known_indication``, needed because
    ``LoadedExport`` has no graph object to mask (see module docstring)."""

    h, r, t = final_triples[:, 0], final_triples[:, 1], final_triples[:, 2]
    forward = (h == drug_idx) & (r == indication_relation_idx) & (t == disease_idx)
    backward = (h == disease_idx) & (r == indication_relation_idx) & (t == drug_idx)
    keep = ~(forward | backward)
    return final_triples[keep]


def _confirm_triples_masked(masked_triples: np.ndarray, *, indication_relation_idx: int, drug_idx: int, disease_idx: int) -> bool:
    h, r, t = masked_triples[:, 0], masked_triples[:, 1], masked_triples[:, 2]
    forward = (h == drug_idx) & (r == indication_relation_idx) & (t == disease_idx)
    backward = (h == disease_idx) & (r == indication_relation_idx) & (t == drug_idx)
    return not bool(np.any(forward | backward))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["transe", "rgcn"], required=True)
    parser.add_argument("--primekg-csv", type=Path, default=DEFAULT_PRIMEKG_CSV)
    parser.add_argument("--disease-id", required=True, help="PrimeKG node-index ID -- see export_model_rankings.py's ID-scheme note")
    parser.add_argument("--known-drug-id", required=True, help="PrimeKG node-index ID of the already-known indication to mask")
    parser.add_argument("--ranking-method", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if not checkpoint.get("is_final_training"):
        raise FinalModelRequiredError(
            f"{args.checkpoint} is not marked is_final_training=True -- refusing to run a case study "
            "against a non-final checkpoint."
        )

    loaded = load_export(args.export_dir)
    graph = load_primekg(args.primekg_csv)

    spec = CaseStudySpec(
        disease_id=args.disease_id,
        disease_name=graph.nodes[args.disease_id]["name"] if args.disease_id in graph.nodes else args.disease_id,
        known_drug_id=args.known_drug_id,
        known_drug_name=graph.nodes[args.known_drug_id]["name"] if args.known_drug_id in graph.nodes else args.known_drug_id,
        ranking_method=args.ranking_method,
        notes=args.notes,
    )

    # Graph-level mask -- unchanged, already-tested module. confirm_masked
    # runs again inside build_case_study_result below (raises
    # MaskingNotConfirmedError on failure), so it isn't duplicated here.
    masked_graph = mask_known_indication(graph, spec)

    n_entities = len(loaded.entity_index)
    n_relations = len(loaded.relation_index)
    disease_idx = loaded.entity_index.id_to_index[args.disease_id]
    drug_idx = loaded.entity_index.id_to_index[args.known_drug_id]

    if args.model == "transe":
        model = TransE(
            n_entities, n_relations,
            embedding_dim=checkpoint["config"]["embedding_dim"], distance=checkpoint["config"]["distance"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        # No message-passing/encode step -- triples-level masking is a
        # documented no-op for TransE (see module docstring).

        def score_fn(drug_ids):
            drug_indices = torch.as_tensor([loaded.entity_index.id_to_index[d] for d in drug_ids], dtype=torch.long)
            rel = torch.full_like(drug_indices, loaded.indication_relation_idx)
            tail = torch.full_like(drug_indices, disease_idx)
            with torch.no_grad():
                return model.score(drug_indices, rel, tail).numpy()

    else:
        masked_triples = _mask_triples(
            loaded.final_triples, indication_relation_idx=loaded.indication_relation_idx,
            drug_idx=drug_idx, disease_idx=disease_idx,
        )
        if not _confirm_triples_masked(
            masked_triples, indication_relation_idx=loaded.indication_relation_idx, drug_idx=drug_idx, disease_idx=disease_idx
        ):
            raise RuntimeError("triples-level masking did not remove the known indication edge -- aborting")
        if len(masked_triples) == len(loaded.final_triples):
            raise RuntimeError(
                f"masking removed 0 rows from final_triples for disease={args.disease_id!r} "
                f"drug={args.known_drug_id!r} -- the known indication was not found where expected; "
                "check the IDs before trusting this case study's result"
            )

        model = RGCN(
            n_entities, n_relations,
            hidden_dim=checkpoint["config"]["hidden_dim"], num_bases=checkpoint["config"]["num_bases"],
            num_layers=checkpoint["config"]["num_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        edge_h_np, edge_r_np, edge_t_np = build_message_passing_edges(masked_triples, n_relations)
        with torch.no_grad():
            entity_embeddings = model.encode(
                torch.as_tensor(edge_h_np, dtype=torch.long), torch.as_tensor(edge_r_np, dtype=torch.long),
                torch.as_tensor(edge_t_np, dtype=torch.long), n_entities,
            )

        def score_fn(drug_ids):
            drug_indices = torch.as_tensor([loaded.entity_index.id_to_index[d] for d in drug_ids], dtype=torch.long)
            rel = torch.full_like(drug_indices, loaded.indication_relation_idx)
            tail = torch.full_like(drug_indices, disease_idx)
            with torch.no_grad():
                return model.score(entity_embeddings, drug_indices, rel, tail).numpy()

    scores = np.asarray(score_fn(loaded.drug_ids))
    order = np.argsort(-scores, kind="stable")
    ranked_drug_ids = [loaded.drug_ids[i] for i in order]
    rank = ranked_drug_ids.index(args.known_drug_id) + 1 if args.known_drug_id in ranked_drug_ids else None

    case_result = build_case_study_result(
        masked_graph,
        spec,
        rank,
        provenance={
            "candidate_universe_size": len(loaded.drug_ids),
            "triples_level_masking": (
                "not applicable -- TransE has no message-passing step for triples-level masking to affect"
                if args.model == "transe"
                else "final_triples row(s) for this disease/drug removed before re-encoding (see module docstring)"
            ),
        },
        limitations=(
            "Single disease/drug anecdote -- not a benchmark result; see docs/FINAL_RESULTS.md for the "
            "benchmark locked-test table.",
            "Candidate universe is the full PrimeKG drug universe used by this model's training -- see "
            "docs/FINAL_RESULTS.md's candidate-universe caveat.",
        ),
    )

    result = {
        "spec": {
            "disease_id": case_result.spec.disease_id,
            "disease_name": case_result.spec.disease_name,
            "known_drug_id": case_result.spec.known_drug_id,
            "known_drug_name": case_result.spec.known_drug_name,
            "ranking_method": case_result.spec.ranking_method,
            "notes": case_result.spec.notes,
        },
        "rank": case_result.rank,
        "provenance": case_result.provenance,
        "limitations": list(case_result.limitations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Case study for {spec.disease_name!r} / {spec.known_drug_name!r}: rank={rank}. Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
