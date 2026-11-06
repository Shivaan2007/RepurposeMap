"""Loads precomputed ``CandidateRankingBundle`` artifacts from disk.

Reads only ``*.json`` files directly inside ``SERVING_DIR`` (default:
``data/serving/`` at the repo root), never recursively and never from an
arbitrary caller-supplied path outside that directory's own
``serving_dir`` parameter (used only by tests, to point at an isolated
fixture directory). A missing bundle for a disease is reported as an
honest "not available" -- ``None`` -- never a fallback to a different
disease's bundle and never a fabricated one.

``data/serving/`` starts empty (only a ``.gitkeep``) because no real
per-disease ranking exists anywhere in this repo yet (see
``docs/ARCHITECTURE.md``'s "offline research/evaluation vs. online
serving/demo" section). This module is the mechanism; it has nothing to
serve until ``export_rankings.py`` is run against a real, approved
ranking output.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from repurposemap.serving.schemas import BundleValidationError, CandidateRankingBundle, validate_bundle

SERVING_DIR = Path(__file__).resolve().parents[4] / "data" / "serving"


class ServingArtifactError(RuntimeError):
    """Raised when a file under the serving directory exists but is not
    a valid bundle -- a data-integrity bug, never silently treated as
    "no bundle for this disease"."""


def _load_file(path: Path) -> CandidateRankingBundle:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingArtifactError(f"could not read/parse {path.name}: {exc}") from exc
    try:
        bundle = CandidateRankingBundle.model_validate(raw)
    except ValidationError as exc:
        raise ServingArtifactError(f"{path.name} does not match the serving bundle schema: {exc}") from exc
    try:
        validate_bundle(bundle)
    except BundleValidationError as exc:
        raise ServingArtifactError(str(exc)) from exc
    return bundle


def list_supported_diseases(serving_dir: Path | None = None) -> list[CandidateRankingBundle]:
    """Every validly-loadable bundle under ``serving_dir``, sorted by
    disease name. A malformed file is skipped here (not raised) since
    this powers a discovery listing meant to always answer -- call
    ``load_bundle_for_disease`` directly for the raised error on one
    specific disease's file."""

    directory = serving_dir if serving_dir is not None else SERVING_DIR
    if not directory.is_dir():
        return []
    bundles: list[CandidateRankingBundle] = []
    for path in sorted(directory.glob("*.json")):
        try:
            bundles.append(_load_file(path))
        except ServingArtifactError:
            continue
    return sorted(bundles, key=lambda b: b.disease_name)


def load_bundle_for_disease(disease_id: str, serving_dir: Path | None = None) -> CandidateRankingBundle | None:
    """The bundle for ``disease_id``, or ``None`` if no artifact file
    exists for it. Raises ``ServingArtifactError`` if a file for this
    disease exists but fails to parse or validate -- that is a
    data-integrity bug, not an absence, and must not be swallowed into
    an honest-looking "not available"."""

    directory = serving_dir if serving_dir is not None else SERVING_DIR
    if not directory.is_dir():
        return None
    path = directory / f"{disease_id}.json"
    if not path.is_file():
        return None
    return _load_file(path)
