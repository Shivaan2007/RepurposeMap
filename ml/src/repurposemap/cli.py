"""Command-line interface for local graph exploration.

Run from the repository root:

    python -m repurposemap stats
    python -m repurposemap search --query "drug"
    python -m repurposemap paths --source-name "DEMO Drug Alpha" --target-name "DEMO Disease Zeta"
    python -m repurposemap stats --source primekg
    python -m repurposemap search --source primekg --query "sildenafil"
    python -m repurposemap paths --source primekg --source-name "sildenafil" --target-name "pulmonary arterial hypertension"

The default source is the synthetic demo data in data/sample/. ``--source primekg``
reads PrimeKG's kg.csv from data/raw/primekg/. Output is for exploration only. It is
not a treatment recommendation, and a path in the graph is not evidence of efficacy.

The CLI only parses arguments and formats output. All graph logic lives in
repurposemap.graph, and PrimeKG parsing lives in repurposemap.adapters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

from repurposemap.adapters import DEFAULT_PRIMEKG_CSV, load_primekg
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

DEFAULT_SAMPLE_CSV = Path("data") / "sample" / "synthetic_demo_graph.csv"
SOURCES = ("sample", "primekg")


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
    raise ValueError(f"unknown command: {args.command}")


def resolve_endpoint(
    graph: nx.MultiDiGraph,
    role: str,
    name: str | None,
    node_id: str | None,
) -> str:
    """Turn a user's name or ID into one canonical node ID.

    An ID must exist in the graph. A name must match exactly one entity. If several
    entities share the name, the error lists each one with its ID so the user can
    choose, and nothing is guessed.
    """
    if node_id is not None:
        if node_id not in graph:
            raise ValueError(f"{role} entity not found: {node_id!r}")
        return node_id

    matches = find_exact_name_matches(graph, name)
    if not matches:
        raise ValueError(f"{role} entity not found: {name!r}")
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
