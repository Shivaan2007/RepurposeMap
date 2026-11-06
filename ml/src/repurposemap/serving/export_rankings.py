"""Offline exporter: builds a validated ``CandidateRankingBundle`` from
already-computed, real ranking results and writes it to
``data/serving/``.

Deliberately does not compute a ranking itself -- it only assembles and
validates a bundle from already-produced, already-ranked candidate
records (e.g. a disease-based evaluation run's own output, or a future
trained model's inference output). Must never be run against the locked
test set, and never invents, re-ranks, or re-scores a candidate.

Usage::

    python -m repurposemap.serving.export_rankings \\
        --disease-id MONDO:0005015 \\
        --disease-name "type 2 diabetes mellitus" \\
        --ranking-method path_quality \\
        --candidate-universe-description "drugs reachable within 3 hops of the disease in the training graph" \\
        --candidates candidates.json \\
        --top-k 10 \\
        --output data/serving/MONDO_0005015.json

``candidates.json`` must be a JSON list of already-ranked
``{"drug_id", "drug_name", "rank", "score"}`` objects (``score`` and
``explanation_paths`` optional) -- this tool truncates to ``--top-k`` and
validates the result; it does not compute rank or score itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from repurposemap.serving.schemas import (
    CandidateRankingBundle,
    CandidateUniverseDescription,
    RankedCandidateRecord,
    RankingArtifactProvenance,
    validate_bundle,
)


def build_bundle(
    *,
    disease_id: str,
    disease_name: str,
    ranking_method: str,
    candidate_universe_description: str,
    candidate_records: list[dict],
    top_k: int,
    candidate_universe_size: int | None = None,
    source_experiment_file: str | None = None,
    source_commit: str | None = None,
    generated_at: str | None = None,
) -> CandidateRankingBundle:
    """Assemble and validate a bundle from already-ranked records.

    Raises ``ValueError`` if ``candidate_records`` is empty or
    ``top_k < 1``. Truncation to ``top_k`` is explicit -- the returned
    bundle's own ``len(candidates)`` tells the caller exactly how many
    survived, never a silent partial result."""

    if not candidate_records:
        raise ValueError(f"no candidate records supplied for disease {disease_id!r}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    ordered = sorted(candidate_records, key=lambda r: r["rank"])
    truncated = ordered[:top_k]

    candidates = [
        RankedCandidateRecord(
            drug_id=record["drug_id"],
            drug_name=record["drug_name"],
            rank=record["rank"],
            score=record.get("score"),
            explanation_paths=record.get("explanation_paths", []),
        )
        for record in truncated
    ]

    bundle = CandidateRankingBundle(
        disease_id=disease_id,
        disease_name=disease_name,
        ranking_method=ranking_method,
        provenance=RankingArtifactProvenance(
            ranking_method=ranking_method,
            source_experiment_file=source_experiment_file,
            source_commit=source_commit,
        ),
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        candidate_universe=CandidateUniverseDescription(
            description=candidate_universe_description,
            size=candidate_universe_size,
        ),
        candidates=candidates,
    )
    validate_bundle(bundle)
    return bundle


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--disease-id", required=True)
    parser.add_argument("--disease-name", required=True)
    parser.add_argument("--ranking-method", required=True)
    parser.add_argument("--candidate-universe-description", required=True)
    parser.add_argument("--candidate-universe-size", type=int, default=None)
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="JSON file: a list of already-ranked {drug_id, drug_name, rank, score?} objects",
    )
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--source-experiment-file", default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    candidate_records = json.loads(args.candidates.read_text())
    bundle = build_bundle(
        disease_id=args.disease_id,
        disease_name=args.disease_name,
        ranking_method=args.ranking_method,
        candidate_universe_description=args.candidate_universe_description,
        candidate_universe_size=args.candidate_universe_size,
        candidate_records=candidate_records,
        top_k=args.top_k,
        source_experiment_file=args.source_experiment_file,
        source_commit=args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle.model_dump(), indent=2) + "\n")
    print(f"Wrote {len(bundle.candidates)} candidate(s) for {bundle.disease_name!r} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
