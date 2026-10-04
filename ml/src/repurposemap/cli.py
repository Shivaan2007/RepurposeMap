"""Command-line interface for local graph exploration.

Run from the repository root:

    python -m repurposemap stats
    python -m repurposemap search --query "drug"
    python -m repurposemap paths --source "DEMO Drug Alpha" --target "DEMO Disease Zeta"

The default graph is the synthetic demo data in data/sample/. This CLI only
explores a local edge table. Its output is not a treatment recommendation.

The CLI only parses arguments and formats output. All graph logic lives in
repurposemap.graph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx

from repurposemap.graph import (
    DEFAULT_MAX_PATH_LENGTH,
    DEFAULT_MAX_PATHS,
    DEFAULT_SEARCH_LIMIT,
    EntityMatch,
    GraphStats,
    PathStep,
    compute_stats,
    find_paths,
    format_path,
    load_graph,
    search_entities,
)

DEFAULT_CSV = Path("data") / "sample" / "synthetic_demo_graph.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m repurposemap",
        description="Explore a local biomedical edge table. Research use only.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser("stats", help="print node, edge and type counts")
    _add_csv_argument(stats)

    search = subparsers.add_parser("search", help="find entities whose names contain a query")
    search.add_argument("--query", required=True, help="text to match, ignoring case")
    search.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"maximum results to print (default {DEFAULT_SEARCH_LIMIT})",
    )
    _add_csv_argument(search)

    paths = subparsers.add_parser("paths", help="find directed paths between two entities")
    paths.add_argument("--source", required=True, help="exact name of the starting entity")
    paths.add_argument("--target", required=True, help="exact name of the ending entity")
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
    _add_csv_argument(paths)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = load_graph(args.csv)
        output = _run_command(args, graph)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


def _add_csv_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"edge table to load (default {DEFAULT_CSV})",
    )


def _run_command(args: argparse.Namespace, graph: nx.MultiDiGraph) -> str:
    if args.command == "stats":
        return render_stats(compute_stats(graph))
    if args.command == "search":
        matches = search_entities(graph, args.query, limit=args.max_results)
        return render_search(args.query, matches)
    if args.command == "paths":
        found = find_paths(
            graph,
            args.source,
            args.target,
            max_length=args.max_path_length,
            max_paths=args.max_results,
        )
        return render_paths(args.source, args.target, args.max_path_length, found)
    raise ValueError(f"unknown command: {args.command}")


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
    lines += [f"  {match.name} ({match.node_type})" for match in matches]
    return "\n".join(lines)


def render_paths(
    source: str,
    target: str,
    max_length: int,
    paths: list[list[PathStep]],
) -> str:
    if not paths:
        return f"No directed path from {source!r} to {target!r} within {max_length} hop(s)."
    blocks = [f"{len(paths)} path(s) from {source!r} to {target!r}:"]
    for index, steps in enumerate(paths, start=1):
        blocks.append(f"\n[{index}] {len(steps)} hop(s)\n{format_path(steps)}")
    return "\n".join(blocks)


def _indented_counts(counts: dict[str, int]) -> list[str]:
    return [f"  {label}: {count}" for label, count in counts.items()]
