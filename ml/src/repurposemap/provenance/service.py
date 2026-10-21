"""Read-only loader for ``ml/experiments/*.json``.

Deliberately narrow: this module reads exactly one directory, non
-recursively, and only files ending in ``.json`` directly inside it --
never an arbitrary user path, never PrimeKG, never a Torch checkpoint.
It must stay importable and callable without ``torch`` or the graph
adapters ever being imported.

Having an experiment JSON file here does **not** mean a usable model
exists (see ``ModelEligibility``'s own docstring) -- only an explicitly
configured checkpoint path (``CONFIGURED_CHECKPOINT``, currently ``None``
because final training has not completed) can make
``get_model_eligibility`` report something an API route is allowed to
call "available".
"""

from __future__ import annotations

import json
from pathlib import Path

from repurposemap.provenance.models import (
    DatasetProvenance,
    EvaluationSummary,
    ExperimentKind,
    ExperimentStatus,
    ModelEligibility,
    ModelProvenance,
    TrainingConfiguration,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[3] / "experiments"

# No eligible final-training checkpoint has been configured. This stays
# None until final training completes and a specific checkpoint path is
# explicitly wired in by hand -- never derived automatically from an
# experiment JSON file's own claims about itself.
CONFIGURED_CHECKPOINT: str | None = None

_REQUIRED_TOP_LEVEL_FIELDS = ("stage", "purpose", "generated_at", "provenance", "safety_and_protocol")


def _classify_kind(safety_and_protocol: dict, stage: str | None) -> tuple[ExperimentKind, bool, bool]:
    """Classify an experiment from its own self-reported protocol flags,
    never from the filename or stage text alone (those are read only as
    a fallback for development runs that predate the protocol block)."""

    final_training_performed = bool(safety_and_protocol.get("final_training_performed", False))
    locked_test_evaluation_performed = bool(safety_and_protocol.get("locked_test_evaluation_performed", False))
    three_config_search_performed = bool(safety_and_protocol.get("three_config_search_performed", False))

    if final_training_performed:
        return ExperimentKind.FINAL_TRAINING, True, locked_test_evaluation_performed
    if three_config_search_performed:
        return ExperimentKind.VALIDATION_SEARCH, False, locked_test_evaluation_performed
    if stage and "profile" in stage.lower():
        return ExperimentKind.GPU_PROFILE, False, locked_test_evaluation_performed
    return ExperimentKind.DEVELOPMENT, False, locked_test_evaluation_performed


def _extract_training_configuration(payload: dict) -> TrainingConfiguration | None:
    profile_run = payload.get("profile_run")
    if isinstance(profile_run, dict):
        return TrainingConfiguration(
            embedding_dim=profile_run.get("embedding_dim"),
            negatives_per_positive=profile_run.get("negatives_per_positive"),
            epochs=profile_run.get("epochs"),
        )
    return None


def _extract_dataset_provenance(payload: dict) -> DatasetProvenance | None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    primekg_export = provenance.get("primekg_export", {})
    return DatasetProvenance(
        source_commit=provenance.get("source_commit"),
        export_fingerprint=provenance.get("export_fingerprint"),
        primekg_sha256=provenance.get("primekg_sha256"),
        n_entities=primekg_export.get("n_entities") if isinstance(primekg_export, dict) else None,
        n_relations=primekg_export.get("n_relations") if isinstance(primekg_export, dict) else None,
    )


def _parse_experiment_file(path: Path) -> ModelProvenance:
    try:
        raw = path.read_text()
    except OSError as exc:
        return ModelProvenance(
            experiment_file=path.name,
            kind=ExperimentKind.UNKNOWN,
            status=ExperimentStatus.MALFORMED,
            parse_errors=[f"could not read file: {exc}"],
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ModelProvenance(
            experiment_file=path.name,
            kind=ExperimentKind.UNKNOWN,
            status=ExperimentStatus.MALFORMED,
            parse_errors=[f"invalid JSON: {exc}"],
        )

    if not isinstance(payload, dict):
        return ModelProvenance(
            experiment_file=path.name,
            kind=ExperimentKind.UNKNOWN,
            status=ExperimentStatus.MALFORMED,
            parse_errors=["experiment JSON must be an object"],
        )

    missing = [field for field in _REQUIRED_TOP_LEVEL_FIELDS if field not in payload]
    if missing:
        return ModelProvenance(
            experiment_file=path.name,
            kind=ExperimentKind.UNKNOWN,
            status=ExperimentStatus.MISSING_REQUIRED_FIELDS,
            stage=payload.get("stage"),
            purpose=payload.get("purpose"),
            generated_at=payload.get("generated_at"),
            parse_errors=[f"missing required field(s): {missing}"],
        )

    safety_and_protocol = payload.get("safety_and_protocol", {})
    if not isinstance(safety_and_protocol, dict):
        safety_and_protocol = {}

    kind, is_final_training, locked_test_evaluation_performed = _classify_kind(safety_and_protocol, payload.get("stage"))

    return ModelProvenance(
        experiment_file=path.name,
        kind=kind,
        status=ExperimentStatus.VALID,
        stage=payload.get("stage"),
        purpose=payload.get("purpose"),
        generated_at=payload.get("generated_at"),
        is_final_training=is_final_training,
        locked_test_evaluation_performed=locked_test_evaluation_performed,
        training_configuration=_extract_training_configuration(payload),
        evaluation_summary=None,
        dataset_provenance=_extract_dataset_provenance(payload),
    )


def load_experiments(experiments_dir: Path | None = None) -> list[ModelProvenance]:
    """Parse every ``*.json`` file directly inside ``experiments_dir``
    (default: ``ml/experiments``). Non-recursive, and never follows a
    path outside that one directory."""

    directory = experiments_dir if experiments_dir is not None else EXPERIMENTS_DIR
    if not directory.is_dir():
        return []
    return [_parse_experiment_file(path) for path in sorted(directory.glob("*.json")) if path.is_file()]


def get_most_recent_completed_stage(records: list[ModelProvenance]) -> str | None:
    """The ``stage`` of the most recently generated valid record, or
    ``None`` if nothing has parsed cleanly yet."""

    valid = [r for r in records if r.status == ExperimentStatus.VALID and r.generated_at]
    if not valid:
        return None
    return max(valid, key=lambda r: r.generated_at or "").stage


def get_model_eligibility(records: list[ModelProvenance]) -> ModelEligibility:
    """Whether the experiment records amount to an eligible final model.

    This reflects only what the records report about themselves --
    callers must still check ``CONFIGURED_CHECKPOINT`` before reporting
    a model as available (see this module's docstring)."""

    final_candidates = [r for r in records if r.status == ExperimentStatus.VALID and r.kind == ExperimentKind.FINAL_TRAINING and r.is_final_training]
    if not final_candidates:
        return ModelEligibility(eligible=False, reason="no final-training experiment record found")

    chosen = final_candidates[0]
    return ModelEligibility(eligible=True, reason="a final-training experiment record was found", final_training_experiment_file=chosen.experiment_file)


def is_model_available(records: list[ModelProvenance]) -> bool:
    """Whether the API may honestly report a usable model. Requires both
    an eligible final-training record AND an explicitly configured
    checkpoint path -- today ``CONFIGURED_CHECKPOINT`` is ``None``, so
    this is always ``False``."""

    return CONFIGURED_CHECKPOINT is not None and get_model_eligibility(records).eligible
