"""Run the frozen disease-level paired bootstrap significance test (R-GCN
vs. TransE, MRR and Hits@10), per the method specified in
``docs/FINAL_RESULTS.md`` section 3. Executes that already-frozen method;
does not choose or adjust it.

Takes the two JSON files written by ``repurposemap rgcn-evaluate
--split locked-test --per-disease-output PATH`` and ``repurposemap
transe-evaluate --split locked-test --per-disease-output PATH``
(``ml/src/repurposemap/cli.py``'s ``_write_per_disease_outcomes``): each a
``{"split": "locked-test", "outcomes": [{"disease_id", "drug_id", "rank"}, ...]}``
record of every held-out pair. Refuses to run unless both files say
``split == "locked-test"`` and cover the identical set of disease IDs.

Usage::

    python ml/scripts/compute_significance.py \\
        --rgcn-outcomes  <rgcn per-disease-output JSON> \\
        --transe-outcomes <transe per-disease-output JSON> \\
        --n-resamples 10000 --seed 42 \\
        --output ml/experiments/rgcn_vs_transe_significance_<commit>.json

Deterministic given the same ``--seed``. Reports, for each of MRR and
Hits@10, the point estimates, the (R-GCN minus TransE) difference, its 95%
percentile bootstrap interval, and whether that interval excludes zero --
never just a point estimate presented as if it settled the question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repurposemap.evaluation.significance import PairedBootstrapResult, paired_bootstrap_ci  # noqa: E402

METRICS = ("mrr", "hits_at_10")


class SplitMismatchError(ValueError):
    """Raised when either input file isn't a locked-test per-disease output."""


def _load_outcomes(path: Path, *, label: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("split") != "locked-test":
        raise SplitMismatchError(
            f"{label} file {path}'s split={payload.get('split')!r}, expected 'locked-test' -- refusing to "
            "run a significance test against anything but the one locked-test evaluation"
        )
    return payload["outcomes"]


def result_to_dict(result: PairedBootstrapResult) -> dict:
    return {
        "metric": result.metric,
        "rgcn": result.point_estimate_a,
        "transe": result.point_estimate_b,
        "diff_rgcn_minus_transe": result.point_estimate_diff,
        "ci_95_low": result.ci_low,
        "ci_95_high": result.ci_high,
        "excludes_zero": result.excludes_zero,
        "n_resamples": result.n_resamples,
        "n_diseases": result.n_diseases,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rgcn-outcomes", type=Path, required=True)
    parser.add_argument("--transe-outcomes", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="optional: write the result JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    rgcn_outcomes = _load_outcomes(args.rgcn_outcomes, label="--rgcn-outcomes")
    transe_outcomes = _load_outcomes(args.transe_outcomes, label="--transe-outcomes")

    results = {
        metric: paired_bootstrap_ci(
            rgcn_outcomes, transe_outcomes, metric=metric, n_resamples=args.n_resamples, seed=args.seed,
        )
        for metric in METRICS
    }

    for metric, result in results.items():
        excludes = "excludes zero" if result.excludes_zero else "includes zero (not significant at 95%)"
        print(
            f"{metric}: R-GCN={result.point_estimate_a:.4f}  TransE={result.point_estimate_b:.4f}  "
            f"diff={result.point_estimate_diff:+.4f}  95% CI=[{result.ci_low:+.4f}, {result.ci_high:+.4f}]  "
            f"({excludes})"
        )

    if args.output is not None:
        payload = {
            "method": "disease-level paired bootstrap, R-GCN vs TransE, 95% percentile interval",
            "seed": args.seed,
            "n_resamples": args.n_resamples,
            "results": {metric: result_to_dict(result) for metric, result in results.items()},
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nresult written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
