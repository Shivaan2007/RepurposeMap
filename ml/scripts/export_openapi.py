"""Export the FastAPI app's OpenAPI schema to a JSON file.

    python ml/scripts/export_openapi.py [--output openapi.json]

Deterministic: FastAPI's ``app.openapi()`` builds the schema purely from
the route/Pydantic-model definitions, so the same code always produces the
same schema (keys sorted here for a stable diff). Never starts a server,
never loads the PrimeKG graph.

Regenerate the TypeScript client after running this -- see web/client's
README / the root API.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "openapi.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"where to write the schema (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)

    from repurposemap.api.app import app

    schema = app.openapi()
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"OpenAPI schema written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
