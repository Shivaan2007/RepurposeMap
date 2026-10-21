"""Read-only experiment/model provenance layer.

Exposes experiment/model status to the API without loading PrimeKG or
Torch model weights -- see ``service.py`` for the one directory it is
allowed to read from and ``models.py`` for the typed contracts.
"""

from __future__ import annotations

from repurposemap.provenance.models import (
    DatasetProvenance,
    EvaluationSummary,
    ExperimentKind,
    ExperimentStatus,
    ModelEligibility,
    ModelProvenance,
    TrainingConfiguration,
)
from repurposemap.provenance.service import (
    get_model_eligibility,
    get_most_recent_completed_stage,
    load_experiments,
)

__all__ = [
    "DatasetProvenance",
    "EvaluationSummary",
    "ExperimentKind",
    "ExperimentStatus",
    "ModelEligibility",
    "ModelProvenance",
    "TrainingConfiguration",
    "get_model_eligibility",
    "get_most_recent_completed_stage",
    "load_experiments",
]
