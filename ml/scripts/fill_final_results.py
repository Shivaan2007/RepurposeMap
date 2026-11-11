"""Post-model-training glue: insert R-GCN's real locked-test numbers into
``docs/FINAL_RESULTS.md``'s comparison table mechanically, so landing them
is a one-command step instead of hand-editing markdown under deadline
pressure.

Takes the JSON artifact ``repurposemap rgcn-evaluate --split locked-test
--artifact PATH`` writes (``ml/src/repurposemap/cli.py``'s
``_run_rgcn_evaluate``: ``hits_at_1``, ``hits_at_3``, ``hits_at_10``,
``mrr``, ``macro_auprc``, ``micro_auprc``, ``split``) and replaces the
**exact** ``TBD`` row for R-GCN in ``docs/FINAL_RESULTS.md``'s "Locked-test
comparison table" with those numbers, formatted to 4 decimal places to
match the existing path/popularity/TransE rows already in that table.

Refuses to run if:
  - ``artifact["split"] != "locked-test"`` (a validation-only artifact must
    never be inserted into the final results table -- this script has no
    way to tell a validation run from a locked-test run except this field,
    so it trusts it and refuses rather than guessing).
  - the exact R-GCN ``TBD`` row isn't found in the target file (already
    filled in, or the table's structure changed) -- never a silent no-op,
    never a fuzzy/partial replacement.

Does not touch any other section of ``docs/FINAL_RESULTS.md`` (the
one-line summary, interpretation, significance result, case studies, and
sign-off checklist are narrative judgment calls for a person, not
mechanical insertion -- this script inserts numbers only).

Usage::

    python ml/scripts/fill_final_results.py \\
        --artifact ml/experiments/rgcn_final_locked_test_<commit>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FINAL_RESULTS_PATH = Path(__file__).resolve().parents[2] / "docs" / "FINAL_RESULTS.md"

_TBD_ROW = "| **Final R-GCN (full universe)** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |"


class ArtifactNotLockedTestError(ValueError):
    """Raised when ``--artifact``'s ``split`` field isn't ``"locked-test"``."""


class RowNotFoundError(ValueError):
    """Raised when the exact TBD row isn't found in the target file --
    never silently skipped or partially matched."""


def build_rgcn_row(artifact: dict) -> str:
    if artifact.get("split") != "locked-test":
        raise ArtifactNotLockedTestError(
            f"artifact's split={artifact.get('split')!r}, expected 'locked-test' -- refusing to insert a "
            "non-locked-test result into the final results table"
        )
    return (
        "| **Final R-GCN (full universe)** | "
        f"{artifact['hits_at_1']:.4f} | {artifact['hits_at_3']:.4f} | {artifact['hits_at_10']:.4f} | "
        f"{artifact['mrr']:.4f} | {artifact['macro_auprc']:.4f} |"
    )


def fill_final_results(artifact: dict, final_results_path: Path) -> str:
    """Returns the updated file content; does not write it (``main`` does,
    after this succeeds) -- keeps this function trivially testable against
    an in-memory string."""

    row = build_rgcn_row(artifact)
    content = final_results_path.read_text()
    if _TBD_ROW not in content:
        raise RowNotFoundError(
            f"did not find the exact R-GCN TBD row in {final_results_path} -- it may already be filled in, "
            "or the table's structure changed. Refusing to guess; edit the file by hand instead."
        )
    return content.replace(_TBD_ROW, row)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--final-results-path", type=Path, default=DEFAULT_FINAL_RESULTS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    artifact = json.loads(args.artifact.read_text())
    updated = fill_final_results(artifact, args.final_results_path)
    args.final_results_path.write_text(updated)
    print(
        f"Inserted R-GCN's locked-test row into {args.final_results_path} from {args.artifact}. "
        "The one-line summary, interpretation, significance result, case studies, and sign-off checklist "
        "in that file still need a person's judgment -- this only inserted the table row."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
