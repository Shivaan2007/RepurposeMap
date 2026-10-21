"""Typed contracts for the experiment/model provenance layer.

These describe what an experiment record *reported about itself*
(``ml/experiments/*.json``, each written by hand or by a Kaggle kernel at
the end of a run) -- never a locked-test label, a checkpoint tensor, or
anything that requires loading PrimeKG or Torch. See
``repurposemap.provenance.service`` for the one directory this is read
from.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ExperimentKind(str, Enum):
    """What an experiment record represents, read from its own
    self-reported ``safety_and_protocol`` flags -- never guessed from a
    filename alone."""

    DEVELOPMENT = "development"
    GPU_PROFILE = "gpu_profile"
    VALIDATION_SEARCH = "validation_search"
    FINAL_TRAINING = "final_training"
    UNKNOWN = "unknown"


class ExperimentStatus(str, Enum):
    """How cleanly an experiment JSON file parsed. ``VALID`` is the only
    status a record's fields should be trusted for."""

    VALID = "valid"
    MALFORMED = "malformed"
    MISSING_REQUIRED_FIELDS = "missing_required_fields"


class TrainingConfiguration(BaseModel):
    """Hyperparameters a training/search run reported about itself --
    never re-derived by loading a checkpoint."""

    embedding_dim: int | None = None
    distance: str | None = None
    learning_rate: float | None = None
    negatives_per_positive: int | None = None
    epochs: int | None = None
    device: str | None = None


class EvaluationSummary(BaseModel):
    """Aggregate validation metrics an experiment reported about itself.

    Deliberately holds **validation** metrics only -- never locked-test
    labels or per-pair results, which this layer never reads."""

    hits_at_1: float | None = None
    hits_at_3: float | None = None
    hits_at_10: float | None = None
    mrr: float | None = None
    macro_ap: float | None = None
    micro_ap: float | None = None


class DatasetProvenance(BaseModel):
    """Which graph export/commit an experiment ran against -- fingerprints
    only, never raw PrimeKG rows or locked-test content."""

    source_commit: str | None = None
    export_fingerprint: str | None = None
    primekg_sha256: str | None = None
    n_entities: int | None = None
    n_relations: int | None = None


class ModelProvenance(BaseModel):
    """One parsed experiment record.

    ``is_final_training`` and ``locked_test_evaluation_performed`` mirror
    the experiment's own ``safety_and_protocol`` block verbatim --
    presence of this record alone never implies a usable model exists;
    see ``ModelEligibility``."""

    experiment_file: str
    kind: ExperimentKind
    status: ExperimentStatus
    stage: str | None = None
    purpose: str | None = None
    generated_at: str | None = None
    is_final_training: bool = False
    locked_test_evaluation_performed: bool = False
    training_configuration: TrainingConfiguration | None = None
    evaluation_summary: EvaluationSummary | None = None
    dataset_provenance: DatasetProvenance | None = None
    parse_errors: list[str] = Field(default_factory=list)


class ModelEligibility(BaseModel):
    """Whether any parsed experiment amounts to a usable, eligible final
    model. ``eligible`` reflects what the experiment records themselves
    report -- actual ``available`` status (see ``system_info.py``) also
    requires an explicitly configured checkpoint path, which does not
    exist yet, so ``available`` must stay ``False`` even if this is
    ``True``."""

    eligible: bool
    reason: str
    final_training_experiment_file: str | None = None
