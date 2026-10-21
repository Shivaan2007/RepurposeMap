"""Command-line interface for local graph exploration.

Run from the repository root:

    python -m repurposemap stats
    python -m repurposemap search --query "drug"
    python -m repurposemap paths --source-name "DEMO Drug Alpha" --target-name "DEMO Disease Zeta"
    python -m repurposemap stats --source primekg
    python -m repurposemap search --source primekg --query "sildenafil"
    python -m repurposemap paths --source primekg --source-name "sildenafil" --target-name "pulmonary arterial hypertension"
    python -m repurposemap explain-paths --source primekg --source-name "sildenafil" --target-id 38436
    python -m repurposemap rank-drugs --source primekg --disease "pulmonary arterial hypertension" --top-k 10
    python -m repurposemap evaluate-baseline --source primekg --test-diseases 20 --seed 42

The default source is the synthetic demo data in data/sample/. ``--source primekg``
reads PrimeKG's kg.csv from data/raw/primekg/. Output is for exploration only. It is
not a treatment recommendation, and a path in the graph is not evidence of efficacy.
The rank-drugs and evaluate-baseline outputs are research hypothesis rankings, not
treatment recommendations. The scoring commands need ``--source primekg``, because
their relation policy is written for PrimeKG.

The CLI only parses arguments and formats output. All graph logic lives in
repurposemap.graph, repurposemap.scoring, repurposemap.ranking and
repurposemap.evaluation. PrimeKG parsing lives in repurposemap.adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import networkx as nx
import torch

from repurposemap.adapters import DEFAULT_PRIMEKG_CSV, compute_file_fingerprint, load_primekg
from repurposemap.embeddings.export import (
    assert_eligible_for_locked_test_evaluation,
    build_export,
    load_export,
    load_locked_test_manifest,
    write_export,
    write_locked_test_manifest,
)
from repurposemap.embeddings.positive_store import build_positive_store
from repurposemap.embeddings.rgcn_model import RGCN
from repurposemap.embeddings.rgcn_train_gpu import (
    SEARCH_CONFIGS as RGCN_SEARCH_CONFIGS,
)
from repurposemap.embeddings.rgcn_train_gpu import (
    build_message_passing_edges,
)
from repurposemap.embeddings.rgcn_train_gpu import (
    train_final as rgcn_train_final,
)
from repurposemap.embeddings.rgcn_train_gpu import (
    train_gpu as rgcn_train_gpu,
)
from repurposemap.embeddings.transe_model import TransE
from repurposemap.embeddings.transe_train_gpu import (
    SEARCH_CONFIGS,
    load_checkpoint,
    train_final,
    train_gpu,
)
from repurposemap.evaluation.artifact import get_git_commit
from repurposemap.evaluation.transe_baseline import evaluate_transe_on_diseases
from repurposemap.evaluation import (
    DEFAULT_EVAL_MAX_CANDIDATES,
    DEFAULT_EVAL_MAX_EDGE_CHECKS_PER_DRUG,
    DEFAULT_EVAL_TIME_LIMIT_PER_DRUG_S,
    DiseaseEvaluationResult,
    EvaluationConfig,
    build_artifact,
    evaluate_disease_split,
    run_with_checkpoint,
)
from repurposemap.graph import (
    DEFAULT_MAX_PATH_LENGTH,
    DEFAULT_MAX_PATHS,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TIME_LIMIT_S,
    EntityMatch,
    GraphStats,
    PathSearchResult,
    compute_stats,
    entity_match,
    find_exact_name_matches,
    find_paths_detailed,
    format_path,
    load_graph,
    search_entities,
)
from repurposemap.ranking import DrugCandidate, DrugRanking, rank_drugs
from repurposemap.ranking.drugs import (
    DEFAULT_MAX_BACKWARD_EDGE_CHECKS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_EDGE_CHECKS_PER_DRUG,
    DEFAULT_PATHS_PER_DRUG,
    DEFAULT_RANK_PATH_LENGTH,
    DEFAULT_TIME_LIMIT_PER_DRUG_S,
)
from repurposemap.scoring import (
    PathScore,
    RankedPath,
    RelationPolicy,
    NeighbourCounts,
    explain_ranking,
    policy_view,
    rank_paths,
    score_path,
)

DEFAULT_SAMPLE_CSV = Path("data") / "sample" / "synthetic_demo_graph.csv"
SOURCES = ("sample", "primekg")
DEFAULT_CANDIDATE_PATHS = 200
DEFAULT_EXPLAIN_RAW_RESULTS = 5
DEFAULT_TOP_K = 10
DEFAULT_EVAL_SEED = 0
DEFAULT_TEST_DISEASES = 20
DEFAULT_TEST_FRACTION = 0.1
SCORING_COMMANDS = ("explain-paths", "rank-drugs", "evaluate-baseline", "transe-export")
# transe-train/transe-evaluate/rgcn-train/rgcn-evaluate all consume only a
# local export package (see repurposemap.embeddings.export) -- they must
# never trigger a PrimeKG/NetworkX graph load, unlike every command above.
# R-GCN reuses the exact same export package as TransE, so there is no
# separate rgcn-export command.
GRAPH_FREE_COMMANDS = ("transe-train", "transe-evaluate", "rgcn-train", "rgcn-evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m repurposemap",
        description="Explore a local biomedical graph. Research use only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("stats", help="print node, edge and type counts")
    _add_source_arguments(stats)

    search = subparsers.add_parser("search", help="find entities whose names contain a query")
    search.add_argument("--query", required=True, help="text to match, ignoring case")
    search.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"maximum results to print (default {DEFAULT_SEARCH_LIMIT})",
    )
    _add_source_arguments(search)

    paths = subparsers.add_parser("paths", help="find directed paths between two entities")
    _add_endpoint_arguments(paths, "source", "starting")
    _add_endpoint_arguments(paths, "target", "ending")
    paths.add_argument(
        "--max-path-length",
        type=int,
        default=DEFAULT_MAX_PATH_LENGTH,
        help=f"maximum number of hops (default {DEFAULT_MAX_PATH_LENGTH})",
    )
    paths.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_PATHS,
        help=f"maximum paths to print (default {DEFAULT_MAX_PATHS})",
    )
    paths.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT_S,
        help=f"seconds before the search stops early (default {DEFAULT_TIME_LIMIT_S:g})",
    )
    _add_source_arguments(paths)

    explain = subparsers.add_parser(
        "explain-paths",
        help="compare raw shortest paths with quality-ranked paths between two entities",
    )
    _add_endpoint_arguments(explain, "source", "starting")
    _add_endpoint_arguments(explain, "target", "ending")
    explain.add_argument(
        "--max-path-length", type=int, default=DEFAULT_MAX_PATH_LENGTH,
        help=f"maximum number of hops (default {DEFAULT_MAX_PATH_LENGTH})",
    )
    explain.add_argument(
        "--max-results", type=int, default=DEFAULT_EXPLAIN_RAW_RESULTS,
        help=f"raw shortest paths to show (default {DEFAULT_EXPLAIN_RAW_RESULTS})",
    )
    explain.add_argument(
        "--candidate-paths", type=int, default=DEFAULT_CANDIDATE_PATHS,
        help=f"candidate paths to score and rank (default {DEFAULT_CANDIDATE_PATHS})",
    )
    explain.add_argument(
        "--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S,
        help=f"seconds per path search (default {DEFAULT_TIME_LIMIT_S:g})",
    )
    _add_source_arguments(explain)

    rank = subparsers.add_parser(
        "rank-drugs",
        help="rank drug candidates for a disease by path quality (research hypothesis ranking)",
    )
    disease = rank.add_mutually_exclusive_group(required=True)
    disease.add_argument("--disease", help="disease name; must match exactly one disease node")
    disease.add_argument("--disease-id", help="canonical ID of the disease node")
    rank.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"ranked drugs to print (default {DEFAULT_TOP_K})",
    )
    _add_ranking_arguments(rank)
    _add_source_arguments(rank)

    evaluate = subparsers.add_parser(
        "evaluate-baseline",
        help="proper disease-based evaluation: hide held-out diseases' treatments, then rank",
    )
    evaluate.add_argument(
        "--test-diseases", type=int, default=DEFAULT_TEST_DISEASES,
        help=f"maximum number of test diseases (default {DEFAULT_TEST_DISEASES})",
    )
    evaluate.add_argument(
        "--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
        help=f"share of diseases held out before the --test-diseases cap (default {DEFAULT_TEST_FRACTION:g})",
    )
    evaluate.add_argument(
        "--seed", type=int, default=DEFAULT_EVAL_SEED,
        help=f"seed for the split and the random baseline, so runs repeat (default {DEFAULT_EVAL_SEED})",
    )
    _add_ranking_arguments(
        evaluate,
        max_candidates_default=DEFAULT_EVAL_MAX_CANDIDATES,
        time_limit_default=DEFAULT_EVAL_TIME_LIMIT_PER_DRUG_S,
        max_edge_checks_default=DEFAULT_EVAL_MAX_EDGE_CHECKS_PER_DRUG,
    )
    evaluate.add_argument(
        "--checkpoint", type=Path, default=None,
        help=(
            "save each disease's result to this file as it finishes, and skip diseases "
            "already recorded there; use for a run large enough that losing it to an "
            "interruption would be expensive (default: off, nothing saved)"
        ),
    )
    evaluate.add_argument(
        "--artifact", type=Path, default=None,
        help=(
            "write a compact JSON result artifact (provenance, config, all three "
            "baselines' metrics, counts, runtime) to this path (default: off)"
        ),
    )
    _add_source_arguments(evaluate)

    transe_export = subparsers.add_parser(
        "transe-export",
        help="write the local TransE export package, and a separate LOCAL-ONLY locked-test manifest",
    )
    transe_export.add_argument("--output-dir", type=Path, required=True, help="directory for triples.npz/metadata.json/validation_indications.json (upload this, nothing else)")
    transe_export.add_argument("--manifest-path", type=Path, required=True, help="where to write locked_test_manifest.json; must NOT be inside --output-dir, and is never uploaded")
    transe_export.add_argument("--seed", type=int, default=42)
    transe_export.add_argument("--test-fraction", type=float, default=0.1)
    transe_export.add_argument("--max-test-diseases", type=int, default=100)
    transe_export.add_argument("--val-fraction", type=float, default=0.1)
    _add_source_arguments(transe_export)

    transe_train = subparsers.add_parser(
        "transe-train",
        help="train TransE from an export package only -- never loads PrimeKG/NetworkX",
    )
    transe_train.add_argument("--export-dir", type=Path, required=True)
    transe_train.add_argument(
        "--config-name", choices=["search-dim64", "search-dim128", "search-l1", "final"], required=True,
    )
    transe_train.add_argument("--epochs", type=int, default=None, help="override the config's default epoch count")
    transe_train.add_argument("--validate-every", type=int, default=2)
    transe_train.add_argument("--device", default="cpu")
    transe_train.add_argument("--output", type=Path, required=True, help="checkpoint path to write")
    transe_train.add_argument(
        "--search-log", type=Path, default=None,
        help="required for --config-name final: the JSON log recording the 3-config search's result",
    )

    transe_evaluate = subparsers.add_parser(
        "transe-evaluate",
        help="evaluate a TransE checkpoint -- validation by default; locked-test only with --manifest and a final checkpoint",
    )
    transe_evaluate.add_argument("--export-dir", type=Path, required=True)
    transe_evaluate.add_argument("--checkpoint", type=Path, required=True)
    transe_evaluate.add_argument("--split", choices=["validation", "locked-test"], default="validation")
    transe_evaluate.add_argument(
        "--manifest", type=Path, default=None,
        help="required for --split locked-test: the LOCAL-ONLY locked_test_manifest.json",
    )
    transe_evaluate.add_argument("--artifact", type=Path, default=None)

    rgcn_train = subparsers.add_parser(
        "rgcn-train",
        help="train R-GCN from the same export package TransE uses -- never loads PrimeKG/NetworkX",
    )
    rgcn_train.add_argument("--export-dir", type=Path, required=True)
    rgcn_train.add_argument(
        "--config-name", choices=["rgcn-base", "rgcn-wide", "rgcn-moreBases", "final"], required=True,
    )
    rgcn_train.add_argument("--epochs", type=int, default=None, help="override the config's default epoch count")
    rgcn_train.add_argument("--validate-every", type=int, default=2)
    rgcn_train.add_argument("--device", default="cpu")
    rgcn_train.add_argument("--output", type=Path, required=True, help="checkpoint path to write")
    rgcn_train.add_argument(
        "--search-log", type=Path, default=None,
        help="required for --config-name final: the JSON log recording the 3-config search's result",
    )

    rgcn_evaluate = subparsers.add_parser(
        "rgcn-evaluate",
        help="evaluate an R-GCN checkpoint -- validation by default; locked-test only with --manifest and a final checkpoint",
    )
    rgcn_evaluate.add_argument("--export-dir", type=Path, required=True)
    rgcn_evaluate.add_argument("--checkpoint", type=Path, required=True)
    rgcn_evaluate.add_argument("--split", choices=["validation", "locked-test"], default="validation")
    rgcn_evaluate.add_argument(
        "--manifest", type=Path, default=None,
        help="required for --split locked-test: the LOCAL-ONLY locked_test_manifest.json",
    )
    rgcn_evaluate.add_argument("--artifact", type=Path, default=None)

    return parser


def _add_ranking_arguments(
    parser: argparse.ArgumentParser,
    *,
    max_candidates_default: int = DEFAULT_MAX_CANDIDATES,
    time_limit_default: float = DEFAULT_TIME_LIMIT_PER_DRUG_S,
    max_edge_checks_default: int = DEFAULT_MAX_EDGE_CHECKS_PER_DRUG,
) -> None:
    parser.add_argument(
        "--max-path-length", type=int, default=DEFAULT_RANK_PATH_LENGTH,
        help=f"maximum hops from drug to disease (default {DEFAULT_RANK_PATH_LENGTH})",
    )
    parser.add_argument(
        "--paths-per-drug", type=int, default=DEFAULT_PATHS_PER_DRUG,
        help=f"shortest candidate paths kept per drug (default {DEFAULT_PATHS_PER_DRUG})",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=max_candidates_default,
        help=f"drug candidates searched; the rest are skipped and reported (default {max_candidates_default})",
    )
    parser.add_argument(
        "--time-limit-per-drug", type=float, default=time_limit_default,
        help=f"seconds per drug search (default {time_limit_default:g})",
    )
    parser.add_argument(
        "--max-edge-checks-per-drug", type=int, default=max_edge_checks_default,
        help=f"edge checks per drug search (default {max_edge_checks_default:,})",
    )
    parser.add_argument(
        "--max-backward-edge-checks", type=int, default=DEFAULT_MAX_BACKWARD_EDGE_CHECKS,
        help=(
            "edge checks for the one search that finds candidates; bounds a densely "
            f"connected disease (default {DEFAULT_MAX_BACKWARD_EDGE_CHECKS:,})"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in GRAPH_FREE_COMMANDS:
            # transe-train/transe-evaluate consume only a local export
            # package -- _load_source (PrimeKG/NetworkX) is never called
            # for these, intentionally, not just by convention.
            output = _run_graph_free_command(args)
        else:
            if args.command in SCORING_COMMANDS and args.source != "primekg":
                raise ValueError(f"{args.command} uses the PrimeKG relation policy; add --source primekg")
            graph = _load_source(args.source, args.csv)
            output = _run_command(args, graph)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="sample",
        help="dataset to load: 'sample' (synthetic demo, default) or 'primekg'",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="edge file to load instead of the source's default file",
    )


def _add_endpoint_arguments(parser: argparse.ArgumentParser, role: str, word: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        f"--{role}-name",
        dest=f"{role}_name",
        help=f"name of the {word} entity; fails if several entities share the name",
    )
    group.add_argument(
        f"--{role}-id",
        dest=f"{role}_id",
        help=f"canonical ID of the {word} entity, as shown by the search command",
    )


def _load_source(source: str, csv: Path | None) -> nx.MultiDiGraph:
    if source == "primekg":
        return load_primekg(csv if csv is not None else DEFAULT_PRIMEKG_CSV)
    return load_graph(csv if csv is not None else DEFAULT_SAMPLE_CSV)


def _run_command(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    if args.command == "stats":
        return render_stats(compute_stats(graph))
    if args.command == "search":
        matches = search_entities(graph, args.query, limit=args.max_results)
        return render_search(args.query, matches)
    if args.command == "paths":
        source = resolve_endpoint(graph, "source", args.source_name, args.source_id)
        target = resolve_endpoint(graph, "target", args.target_name, args.target_id)
        result = find_paths_detailed(
            graph,
            source,
            target,
            max_length=args.max_path_length,
            max_paths=args.max_results,
            time_limit_s=args.time_limit,
        )
        return render_paths(
            entity_match(graph, source),
            entity_match(graph, target),
            args.max_path_length,
            result,
        )
    if args.command == "explain-paths":
        return _run_explain(args, graph)
    if args.command == "rank-drugs":
        return _run_rank_drugs(args, graph)
    if args.command == "evaluate-baseline":
        return _run_evaluate(args, graph)
    if args.command == "transe-export":
        return _run_transe_export(args, graph)
    raise ValueError(f"unknown command: {args.command}")


def _run_transe_export(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    output_dir: Path = args.output_dir
    manifest_path: Path = args.manifest_path
    if output_dir.resolve() in (manifest_path.resolve(), *manifest_path.resolve().parents):
        raise ValueError("--manifest-path must not be inside --output-dir; it must never be uploaded with the export")

    csv_path = args.csv if args.csv is not None else DEFAULT_PRIMEKG_CSV
    bundle = build_export(
        graph, csv_path=csv_path, seed=args.seed, test_fraction=args.test_fraction,
        max_test_diseases=args.max_test_diseases, val_fraction=args.val_fraction,
    )
    write_export(bundle, output_dir)
    write_locked_test_manifest(bundle, manifest_path)

    return "\n".join(
        [
            f"Export package written to {output_dir} (upload only this directory)",
            f"Locked-test manifest written to {manifest_path} -- LOCAL ONLY, do not upload this file",
            f"n_final_triples: {len(bundle.arrays.final_triples)}",
            f"n_tuning_triples: {len(bundle.arrays.final_triples) - int(bundle.arrays.val_indication_mask.sum())}",
            f"val_indication_mask_true_count: {int(bundle.arrays.val_indication_mask.sum())}",
            f"n_entities: {len(bundle.metadata['entity_index']['index_to_id'])}",
            f"n_relations: {len(bundle.metadata['relation_index']['index_to_id'])}",
            f"n_val_indications: {len(bundle.val_indications)}",
            f"n_test_indications_in_separate_manifest: {len(bundle.test_indications)}",
            f"export_fingerprint: {bundle.metadata['export_fingerprint']}",
            f"primekg_sha256: {bundle.metadata['primekg_fingerprint']['sha256']}",
            f"git_commit: {bundle.metadata['git_commit']}",
            f"git_dirty: {bundle.metadata['git_dirty']}",
        ]
    )


def _run_graph_free_command(args: argparse.Namespace) -> str:
    if args.command == "transe-train":
        return _run_transe_train(args)
    if args.command == "transe-evaluate":
        return _run_transe_evaluate(args)
    if args.command == "rgcn-train":
        return _run_rgcn_train(args)
    if args.command == "rgcn-evaluate":
        return _run_rgcn_evaluate(args)
    raise ValueError(f"unknown graph-free command: {args.command}")


def _run_transe_train(args: argparse.Namespace) -> str:
    loaded = load_export(args.export_dir)
    config_map = dict(SEARCH_CONFIGS)

    if args.config_name == "final":
        if args.search_log is None:
            raise ValueError("--config-name final requires --search-log (the recorded 3-config search result)")
        search_log = json.loads(args.search_log.read_text())
        selected_config = config_map[search_log["selected_name"]]

        if args.epochs is not None:
            epochs = args.epochs
        else:
            selected_run = next(r for r in search_log["runs"] if r["name"] == search_log["selected_name"])
            if selected_run["best_val_epoch"] is None:
                raise ValueError("search log has no recorded best_val_epoch for the selected config; pass --epochs explicitly")
            epochs = selected_run["best_val_epoch"] + 1  # best_val_epoch is 0-indexed

        git_commit, git_dirty = get_git_commit()
        checkpoint = train_final(
            loaded, selected_config=selected_config, epochs=epochs, search_log=search_log,
            checkpoint_path=args.output, git_commit=git_commit, git_dirty=git_dirty, device=args.device,
        )
        return "\n".join(
            [
                f"Final checkpoint written to {args.output}",
                "is_final_training: true",
                f"selected_config: {search_log['selected_name']}",
                f"epochs: {epochs}",
                f"export_fingerprint: {checkpoint['export_fingerprint']}",
            ]
        )

    if args.config_name not in config_map:
        raise ValueError(f"unknown --config-name {args.config_name!r}")
    base_config = config_map[args.config_name]
    epochs = args.epochs if args.epochs is not None else base_config.epochs
    config = replace(base_config, epochs=epochs, device=args.device, validate_every=args.validate_every)

    n_entities = len(loaded.entity_index)
    n_relations = len(loaded.relation_index)
    tuning_triples = loaded.tuning_triples
    positive_store = build_positive_store(tuning_triples, n_relations, n_entities)

    result = train_gpu(
        tuning_triples, n_entities, n_relations, loaded.type_pools, positive_store, config,
        entity_index=loaded.entity_index, indication_relation_idx=loaded.indication_relation_idx,
        val_disease_ids=loaded.val_disease_ids, val_indications=loaded.val_indications, drug_ids=loaded.drug_ids,
        checkpoint_path=args.output,
    )
    log_path = args.output.with_suffix(".log.json")
    log_path.write_text(
        json.dumps(
            {
                "name": args.config_name,
                "config": config.to_dict(),
                "history": [r.to_dict() for r in result.history],
                "stopped_early": result.stopped_early,
            },
            indent=2,
        )
    )
    return "\n".join(
        [
            f"Checkpoint written to {args.output}",
            f"Run log written to {log_path}",
            f"final_epoch: {result.final_epoch}",
            f"stopped_early: {result.stopped_early}",
        ]
    )


def _run_transe_evaluate(args: argparse.Namespace) -> str:
    loaded = load_export(args.export_dir)
    # Local evaluation always runs on CPU regardless of what device the
    # checkpoint was trained on (checkpoint["config"]["device"] is left
    # exactly as recorded -- it is never read to pick an evaluation
    # device, here or anywhere else in this function).
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")

    model = TransE(
        len(loaded.entity_index), len(loaded.relation_index),
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

    if args.split == "validation":
        result = evaluate_transe_on_diseases(score_fn, loaded.val_disease_ids, loaded.val_indications, loaded.drug_ids)
    else:
        if args.manifest is None:
            raise ValueError("--split locked-test requires --manifest PATH (the local-only locked_test_manifest.json)")
        manifest = load_locked_test_manifest(args.manifest)
        assert_eligible_for_locked_test_evaluation(checkpoint, manifest)
        result = evaluate_transe_on_diseases(score_fn, manifest.test_disease_ids, manifest.test_indications, loaded.drug_ids)

    lines = [
        f"split: {args.split}",
        f"n_pairs: {len(result.outcomes)}",
        f"Hits@1: {result.hits_at(1):.4f}",
        f"Hits@3: {result.hits_at(3):.4f}",
        f"Hits@10: {result.hits_at(10):.4f}",
        f"MRR: {result.mrr:.4f}",
        f"macro AP: {result.macro_auprc:.6f}",
        f"micro AP: {result.micro_auprc:.6f}",
    ]
    if args.artifact is not None:
        args.artifact.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "n_pairs": len(result.outcomes),
                    "hits_at_1": result.hits_at(1),
                    "hits_at_3": result.hits_at(3),
                    "hits_at_10": result.hits_at(10),
                    "mrr": result.mrr,
                    "macro_auprc": result.macro_auprc,
                    "micro_auprc": result.micro_auprc,
                },
                indent=2,
            )
        )
        lines.append(f"artifact written to {args.artifact}")
    return "\n".join(lines)


def _run_rgcn_train(args: argparse.Namespace) -> str:
    loaded = load_export(args.export_dir)
    config_map = dict(RGCN_SEARCH_CONFIGS)

    if args.config_name == "final":
        if args.search_log is None:
            raise ValueError("--config-name final requires --search-log (the recorded 3-config search result)")
        search_log = json.loads(args.search_log.read_text())
        selected_config = config_map[search_log["selected_name"]]

        if args.epochs is not None:
            epochs = args.epochs
        else:
            selected_run = next(r for r in search_log["runs"] if r["name"] == search_log["selected_name"])
            if selected_run["best_val_epoch"] is None:
                raise ValueError("search log has no recorded best_val_epoch for the selected config; pass --epochs explicitly")
            epochs = selected_run["best_val_epoch"] + 1  # best_val_epoch is 0-indexed

        git_commit, git_dirty = get_git_commit()
        checkpoint = rgcn_train_final(
            loaded, selected_config=selected_config, epochs=epochs, search_log=search_log,
            checkpoint_path=args.output, git_commit=git_commit, git_dirty=git_dirty, device=args.device,
        )
        return "\n".join(
            [
                f"Final checkpoint written to {args.output}",
                "is_final_training: true",
                f"selected_config: {search_log['selected_name']}",
                f"epochs: {epochs}",
                f"export_fingerprint: {checkpoint['export_fingerprint']}",
            ]
        )

    if args.config_name not in config_map:
        raise ValueError(f"unknown --config-name {args.config_name!r}")
    base_config = config_map[args.config_name]
    epochs = args.epochs if args.epochs is not None else base_config.epochs
    config = replace(base_config, epochs=epochs, device=args.device, validate_every=args.validate_every)

    n_entities = len(loaded.entity_index)
    n_relations = len(loaded.relation_index)
    tuning_triples = loaded.tuning_triples
    positive_store = build_positive_store(tuning_triples, n_relations, n_entities)

    result = rgcn_train_gpu(
        tuning_triples, n_entities, n_relations, loaded.type_pools, positive_store, config,
        entity_index=loaded.entity_index, indication_relation_idx=loaded.indication_relation_idx,
        val_disease_ids=loaded.val_disease_ids, val_indications=loaded.val_indications, drug_ids=loaded.drug_ids,
        checkpoint_path=args.output,
    )
    log_path = args.output.with_suffix(".log.json")
    log_path.write_text(
        json.dumps(
            {
                "name": args.config_name,
                "config": config.to_dict(),
                "history": [r.to_dict() for r in result.history],
                "stopped_early": result.stopped_early,
            },
            indent=2,
        )
    )
    return "\n".join(
        [
            f"Checkpoint written to {args.output}",
            f"Run log written to {log_path}",
            f"final_epoch: {result.final_epoch}",
            f"stopped_early: {result.stopped_early}",
        ]
    )


def _run_rgcn_evaluate(args: argparse.Namespace) -> str:
    loaded = load_export(args.export_dir)
    # Local evaluation always runs on CPU regardless of what device the
    # checkpoint was trained on -- same rule as _run_transe_evaluate.
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")

    n_entities = len(loaded.entity_index)
    n_relations = len(loaded.relation_index)
    model = RGCN(
        n_entities, n_relations,
        hidden_dim=checkpoint["config"]["hidden_dim"], num_bases=checkpoint["config"]["num_bases"],
        num_layers=checkpoint["config"]["num_layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # The expensive full-graph message-passing forward pass runs exactly
    # once here, against final_triples (the same triples the checkpoint
    # was trained on) -- never once per disease inside score_fn below.
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

    if args.split == "validation":
        result = evaluate_transe_on_diseases(score_fn, loaded.val_disease_ids, loaded.val_indications, loaded.drug_ids)
    else:
        if args.manifest is None:
            raise ValueError("--split locked-test requires --manifest PATH (the local-only locked_test_manifest.json)")
        manifest = load_locked_test_manifest(args.manifest)
        assert_eligible_for_locked_test_evaluation(checkpoint, manifest)
        result = evaluate_transe_on_diseases(score_fn, manifest.test_disease_ids, manifest.test_indications, loaded.drug_ids)

    lines = [
        f"split: {args.split}",
        f"n_pairs: {len(result.outcomes)}",
        f"Hits@1: {result.hits_at(1):.4f}",
        f"Hits@3: {result.hits_at(3):.4f}",
        f"Hits@10: {result.hits_at(10):.4f}",
        f"MRR: {result.mrr:.4f}",
        f"macro AP: {result.macro_auprc:.6f}",
        f"micro AP: {result.micro_auprc:.6f}",
    ]
    if args.artifact is not None:
        args.artifact.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "n_pairs": len(result.outcomes),
                    "hits_at_1": result.hits_at(1),
                    "hits_at_3": result.hits_at(3),
                    "hits_at_10": result.hits_at(10),
                    "mrr": result.mrr,
                    "macro_auprc": result.macro_auprc,
                    "micro_auprc": result.micro_auprc,
                },
                indent=2,
            )
        )
        lines.append(f"artifact written to {args.artifact}")
    return "\n".join(lines)


def _run_explain(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    policy = RelationPolicy()
    source = resolve_endpoint(graph, "source", args.source_name, args.source_id)
    target = resolve_endpoint(graph, "target", args.target_name, args.target_id)
    raw = find_paths_detailed(
        graph, source, target,
        max_length=args.max_path_length,
        max_paths=args.max_results,
        time_limit_s=args.time_limit,
    )
    # Candidates come from the policy view, so excluded edges cannot fill the cap.
    candidates = find_paths_detailed(
        policy_view(graph, policy), source, target,
        max_length=args.max_path_length,
        max_paths=args.candidate_paths,
        time_limit_s=args.time_limit,
    )
    counts = NeighbourCounts(graph, policy)
    ranked = rank_paths(candidates.paths, policy, counts)
    raw_scores = [score_path(steps, policy, counts) for steps in raw.paths]
    return render_explain(
        entity_match(graph, source),
        entity_match(graph, target),
        raw.paths,
        raw_scores,
        raw,
        ranked[: args.max_results],
        len(ranked),
        candidates,
    )


def _run_rank_drugs(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    disease = resolve_endpoint(graph, "disease", args.disease, args.disease_id, node_type="disease")
    ranking = rank_drugs(
        graph,
        disease,
        neighbour_counts=NeighbourCounts(graph, RelationPolicy()),
        max_path_length=args.max_path_length,
        paths_per_drug=args.paths_per_drug,
        max_candidates=args.max_candidates,
        time_limit_per_drug_s=args.time_limit_per_drug,
        max_edge_checks_per_drug=args.max_edge_checks_per_drug,
        max_backward_edge_checks=args.max_backward_edge_checks,
    )
    return render_rank_drugs(ranking, args.top_k)


def _run_evaluate(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    config = EvaluationConfig(
        seed=args.seed,
        test_fraction=args.test_fraction,
        max_test_diseases=args.test_diseases,
        max_path_length=args.max_path_length,
        paths_per_drug=args.paths_per_drug,
        max_candidates=args.max_candidates,
        time_limit_per_drug_s=args.time_limit_per_drug,
        max_edge_checks_per_drug=args.max_edge_checks_per_drug,
        max_backward_edge_checks=args.max_backward_edge_checks,
    )
    if args.checkpoint is None:
        result = evaluate_disease_split(graph, config)
    else:
        result = run_with_checkpoint(graph, config, args.checkpoint, on_disease_done=_print_disease_progress)

    if args.artifact is not None:
        _write_artifact(args, graph, result)
    return render_evaluation(result)


def _write_artifact(args: argparse.Namespace, graph: nx.MultiDiGraph, result: DiseaseEvaluationResult) -> None:
    csv_path = args.csv if args.csv is not None else DEFAULT_PRIMEKG_CSV
    fingerprint = compute_file_fingerprint(csv_path)
    popularity_universe_size = sum(1 for _, data in graph.nodes(data=True) if data.get("node_type") == "drug")
    artifact = build_artifact(result, fingerprint, popularity_universe_size=popularity_universe_size)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Artifact written to {args.artifact}", flush=True)


def _print_disease_progress(disease_id: str, outcomes: list) -> None:
    best_rank = min((o.path_rank for o in outcomes if o.path_rank is not None), default=None)
    where = f"best rank {best_rank}" if best_rank is not None else "no true drug ranked"
    print(
        f"  [checkpoint] {outcomes[0].disease_name!r} ({disease_id}): "
        f"{len(outcomes)} true drug(s), {outcomes[0].n_candidates} candidates, {where}",
        flush=True,
    )


def resolve_endpoint(
    graph: nx.MultiDiGraph,
    role: str,
    name: str | None,
    node_id: str | None,
    *,
    node_type: str | None = None,
) -> str:
    """Turn a user's name or ID into one canonical node ID.

    An ID must exist in the graph. A name must match exactly one entity. If several
    entities share the name, the error lists each one with its ID so the user can
    choose, and nothing is guessed. ``node_type`` limits both lookups to that type.
    """
    if node_id is not None:
        if node_id not in graph:
            raise ValueError(f"{role} entity not found: {node_id!r}")
        if node_type is not None and graph.nodes[node_id]["node_type"] != node_type:
            raise ValueError(f"{role} entity {node_id!r} is not a {node_type} node")
        return node_id

    matches = find_exact_name_matches(graph, name)
    if node_type is not None:
        matches = [match for match in matches if match.node_type == node_type]
    if not matches:
        suffix = f" with type {node_type}" if node_type is not None else ""
        raise ValueError(f"{role} entity not found: {name!r}{suffix}")
    if len(matches) > 1:
        lines = [f"{role} name {name!r} matches {len(matches)} entities. Use --{role}-id:"]
        lines += [_candidate_line(match) for match in matches]
        raise ValueError("\n".join(lines))
    return matches[0].id


def render_stats(stats: GraphStats) -> str:
    lines = [
        f"Total nodes: {stats.n_nodes}",
        f"Total edges: {stats.n_edges}",
        "Node type counts:",
        *_indented_counts(stats.node_type_counts),
        "Relation type counts:",
        *_indented_counts(stats.relation_counts),
    ]
    return "\n".join(lines)


def render_search(query: str, matches: list[EntityMatch]) -> str:
    if not matches:
        return f"No entities match {query!r}."
    lines = [f"{len(matches)} match(es) for {query!r}:"]
    lines += [_search_line(match) for match in matches]
    return "\n".join(lines)


def render_paths(
    source: EntityMatch,
    target: EntityMatch,
    max_length: int,
    result: PathSearchResult,
) -> str:
    source_label = _endpoint_label(source)
    target_label = _endpoint_label(target)
    if not result.paths:
        if result.truncated:
            return (
                f"Search stopped early ({result.stop_reason}) before a path from "
                f"{source_label} to {target_label} was found. This is not proof that "
                f"no path exists. Try a larger --time-limit or a shorter --max-path-length."
            )
        return f"No directed path from {source_label} to {target_label} within {max_length} hop(s)."

    blocks = [f"{len(result.paths)} path(s) from {source_label} to {target_label}:"]
    for index, steps in enumerate(result.paths, start=1):
        blocks.append(f"\n[{index}] {len(steps)} hop(s)\n{format_path(steps)}")
    if result.truncated:
        blocks.append(
            f"\nSearch stopped early ({result.stop_reason}). More paths may exist."
        )
    return "\n".join(blocks)


def render_explain(
    source: EntityMatch,
    target: EntityMatch,
    raw_paths: list,
    raw_scores: list[PathScore],
    raw_result: PathSearchResult,
    ranked: list[RankedPath],
    n_valid: int,
    candidates: PathSearchResult,
) -> str:
    lines = [
        f"Path quality for {_endpoint_label(source)} -> {_endpoint_label(target)}.",
        "Quality scores are heuristics, not evidence that a drug works.",
        "",
        "Raw shortest paths (full graph, every relation):",
    ]
    if not raw_paths:
        lines.append("  none found" + (f" (search stopped: {raw_result.stop_reason})" if raw_result.truncated else ""))
    for index, (steps, score) in enumerate(zip(raw_paths, raw_scores, strict=True), start=1):
        status = "valid" if score.valid else "invalid under relation policy"
        lines.append(f"  [{index}] {len(steps)} hop(s), {status}")
        lines.append(_indent(format_path(steps), 6))
        lines += [f"      flag: {flag}" for flag in score.flags]

    shown = f", showing {len(ranked)} of {n_valid}" if n_valid > len(ranked) else ""
    lines += ["", f"Quality-ranked paths (valid candidates only{shown}):"]
    if not ranked:
        lines.append(
            f"  none of {len(candidates.paths)} candidate path(s) is valid under the relation policy"
            if candidates.paths else "  no candidate path found"
        )
    for index, ranked_path in enumerate(ranked, start=1):
        score = ranked_path.score
        lines.append(
            f"  [{index}] total {score.total_score:.3f} = length {score.length_score:.3f}"
            f" + relation {score.relation_score:.3f} - hub {score.hub_penalty:.3f}"
            f"  ({len(ranked_path.steps)} hop(s), raw candidate #{ranked_path.raw_index + 1})"
        )
        lines.append("      hops: " + ", ".join(score.hop_categories))
        lines.append(_indent(format_path(list(ranked_path.steps)), 6))
        lines += [f"      flag: {flag}" for flag in score.flags]
    if ranked:
        lines += ["", "Why the ranking is in this order:"]
        lines += [f"  {line}" for line in explain_ranking(ranked)]
    if candidates.truncated:
        lines.append(f"\nCandidate search stopped early ({candidates.stop_reason}). More paths may exist.")
    return "\n".join(lines)


def render_rank_drugs(ranking: DrugRanking, top_k: int) -> str:
    lines = [
        "Research hypothesis ranking. Not a treatment recommendation.",
        f"Disease: {ranking.disease_name!r} (id {ranking.disease_id})",
        f"Candidate drugs searched: {ranking.n_candidates}"
        + (" (candidate cap reached, the rest were skipped)" if ranking.candidate_cap_hit else ""),
        f"Drugs with a valid path: {len(ranking.ranked)}. Without one: {ranking.n_without_path}.",
    ]
    if ranking.n_truncated_searches:
        lines.append(f"Searches stopped at a limit: {ranking.n_truncated_searches}. Their paths may be incomplete.")
    if ranking.backward_search_truncated:
        lines.append(
            "The candidate search itself stopped at its edge-check budget. "
            "Some farther candidates may be missing, biased toward the ones found first."
        )
    if not ranking.ranked:
        lines.append("No candidate drug has a valid path to this disease within the hop limit.")
        lines.append("This is not proof that no drug could work.")
        return "\n".join(lines)

    for position, candidate in enumerate(ranking.ranked[:top_k], start=1):
        lines += ["", _rank_line(position, candidate)]
        best = candidate.best_path
        lines.append(f"    best path ({len(best.steps)} hop(s), total {best.score.total_score:.3f}):")
        lines.append(_indent(format_path(list(best.steps)), 8))
        lines += [f"        flag: {flag}" for flag in best.score.flags]
    return "\n".join(lines)


def render_evaluation(result: DiseaseEvaluationResult) -> str:
    """Render a disease-based evaluation. Research use only, not a treatment recommendation."""
    config_line = ", ".join(f"{key}={value}" for key, value in result.config.to_dict().items())
    lines = [
        "Path baseline evaluation — research use only",
        "Research hypothesis evaluation. Not a treatment recommendation.",
        f"Configuration: {config_line}",
        f"Train diseases: {result.n_train_diseases}   Test diseases: {result.n_test_diseases}",
        f"Train indication edges: {result.n_train_indications}   "
        f"Held-out (test) indication edges: {result.n_test_indications}",
        f"Diseases with candidate cap reached: {result.n_diseases_with_candidate_cap_hit}   "
        f"with a truncated per-drug search: {result.n_diseases_with_truncated_searches}   "
        f"with the candidate-finding search truncated: {result.n_diseases_with_backward_search_truncated}",
        "Random baseline: candidate-conditioned (shuffles the path baseline's own candidates only). "
        "Popularity baseline: ranks the complete drug universe, independent of the path search.",
        "",
    ]
    for outcome in result.outcomes:
        where = f"rank {outcome.path_rank}" if outcome.path_rank is not None else "not ranked"
        random_where = f"rank {outcome.random_rank}" if outcome.random_rank is not None else "not ranked"
        popularity_where = f"rank {outcome.popularity_rank}" if outcome.popularity_rank is not None else "not ranked"
        lines.append(
            f"  {outcome.drug_name} -> {outcome.disease_name}: {where} of {outcome.n_candidates} candidates "
            f"(random: {random_where}, popularity: {popularity_where})"
        )
    lines.append("")
    for k in result.config.top_ks:
        lines.append(
            f"Hits@{k}: {result.hits_at(k):.3f}  "
            f"(random: {result.random_hits_at(k):.3f}, popularity: {result.popularity_hits_at(k):.3f})"
        )
    lines.append(f"MRR: {result.mrr:.3f}  (random: {result.random_mrr:.3f}, popularity: {result.popularity_mrr:.3f})")
    lines.append(f"Runtime: {result.runtime_s:.1f}s")
    lines.append("")
    lines.append(
        "A handful of test diseases cannot support a claim of predictive performance. "
        "Compare against the random and popularity baselines above before reading anything into these numbers."
    )
    return "\n".join(lines)


def _rank_line(position: int, candidate: DrugCandidate) -> str:
    line = (
        f"{position:>3}. {candidate.drug_name}  id={candidate.drug_id}"
        f"  score={candidate.score:.3f}  valid_paths={candidate.n_qualifying_paths}"
        f"  targets={candidate.n_targets}"
    )
    if candidate.promiscuity_penalty:
        line += f"  promiscuity_penalty={candidate.promiscuity_penalty:g}"
    if candidate.search_truncated:
        line += "  (search stopped early)"
    return line


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _search_line(match: EntityMatch) -> str:
    line = f"  {match.name} ({match.node_type})"
    if match.id != match.name:
        line += f"  id={match.id}"
    if match.external_id:
        line += f"  external_id={match.external_id}"
    return line


def _candidate_line(match: EntityMatch) -> str:
    external = f", external_id={match.external_id}" if match.external_id else ""
    return f"  id={match.id}  {match.name} ({match.node_type}{external})"


def _endpoint_label(match: EntityMatch) -> str:
    if match.id == match.name:
        return repr(match.name)
    return f"{match.name!r} (id {match.id})"


def _indented_counts(counts: dict[str, int]) -> list[str]:
    return [f"  {label}: {count}" for label, count in counts.items()]
