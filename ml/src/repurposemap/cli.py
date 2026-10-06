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
    python -m repurposemap evaluate-baseline --source primekg --max-pairs 5

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
import sys
from pathlib import Path

import networkx as nx

from repurposemap.adapters import DEFAULT_PRIMEKG_CSV, load_primekg
from repurposemap.evaluation import (
    DEFAULT_EVAL_PAIRS,
    DEFAULT_EVAL_SEED,
    EvaluationResult,
    evaluate_baseline,
    select_pairs,
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
SCORING_COMMANDS = ("explain-paths", "rank-drugs", "evaluate-baseline")


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
        help="hold out known indications and report the held-out drug's rank",
    )
    evaluate.add_argument(
        "--max-pairs", type=int, default=DEFAULT_EVAL_PAIRS,
        help=f"indication pairs to evaluate (default {DEFAULT_EVAL_PAIRS})",
    )
    evaluate.add_argument(
        "--seed", type=int, default=DEFAULT_EVAL_SEED,
        help=f"seed for choosing pairs, so runs repeat (default {DEFAULT_EVAL_SEED})",
    )
    _add_ranking_arguments(evaluate)
    _add_source_arguments(evaluate)
    return parser


def _add_ranking_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-path-length", type=int, default=DEFAULT_RANK_PATH_LENGTH,
        help=f"maximum hops from drug to disease (default {DEFAULT_RANK_PATH_LENGTH})",
    )
    parser.add_argument(
        "--paths-per-drug", type=int, default=DEFAULT_PATHS_PER_DRUG,
        help=f"shortest candidate paths kept per drug (default {DEFAULT_PATHS_PER_DRUG})",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
        help=f"drug candidates searched; the rest are skipped and reported (default {DEFAULT_MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--time-limit-per-drug", type=float, default=DEFAULT_TIME_LIMIT_PER_DRUG_S,
        help=f"seconds per drug search (default {DEFAULT_TIME_LIMIT_PER_DRUG_S:g})",
    )
    parser.add_argument(
        "--max-edge-checks-per-drug", type=int, default=DEFAULT_MAX_EDGE_CHECKS_PER_DRUG,
        help=f"edge checks per drug search (default {DEFAULT_MAX_EDGE_CHECKS_PER_DRUG:,})",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
    raise ValueError(f"unknown command: {args.command}")


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
    )
    return render_rank_drugs(ranking, args.top_k)


def _run_evaluate(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    policy = RelationPolicy()
    pairs = select_pairs(graph, args.max_pairs, args.seed)
    result = evaluate_baseline(
        graph,
        pairs,
        neighbour_counts=NeighbourCounts(graph, policy),
        policy=policy,
        seed=args.seed,
        max_path_length=args.max_path_length,
        paths_per_drug=args.paths_per_drug,
        max_candidates=args.max_candidates,
        time_limit_per_drug_s=args.time_limit_per_drug,
        max_edge_checks_per_drug=args.max_edge_checks_per_drug,
    )
    return render_evaluation(result)


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


def render_evaluation(result: EvaluationResult) -> str:
    lines = [
        "Research hypothesis evaluation. Not a treatment recommendation.",
        f"Baseline sanity check on {result.n_pairs} held-out indication pair(s), seed {result.seed}.",
        "Small sample, not a trained model. These numbers are not predictive performance.",
        "",
    ]
    for outcome in result.outcomes:
        pair = outcome.pair
        where = f"rank {outcome.rank}" if outcome.rank is not None else "not ranked"
        lines.append(
            f"  {pair.drug_name} -> {pair.disease_name}: {where} of {outcome.n_ranked} ranked, "
            f"hit@10 {'yes' if outcome.hit_at_k else 'no'}, reciprocal rank {outcome.reciprocal_rank:.3f}"
        )
    lines += [
        "",
        f"Hits@10: {result.hits_at_10:.3f}",
        f"Mean reciprocal rank: {result.mean_reciprocal_rank:.3f}",
    ]
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
