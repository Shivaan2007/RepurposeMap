"""Compact, reusable benchmark result artifact.

Bundles one `DiseaseEvaluationResult` with the dataset and code provenance
needed to make the numbers reproducible and comparable later (against a
future TransE or R-GCN run, or a later PrimeKG snapshot). Local output only,
the same treatment as a checkpoint file: not committed by this milestone.

``artifact_schema_version`` exists so a future format change (adding an
`auprc_metrics` key, a TransE/R-GCN comparison column, or a TxGNN-style split
descriptor) can be told apart from this version without ambiguity.
``to_dict()`` is a flat, ordinary dict rather than a rigid fixed shape,
specifically so such additions do not require breaking this one.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from repurposemap.adapters.primekg import PrimeKGFingerprint
from repurposemap.evaluation.disease_eval import DiseaseEvaluationResult

ARTIFACT_SCHEMA_VERSION = 1

# Describes the existing random baseline precisely: it shuffles the path
# baseline's own candidate list, not the complete drug universe. See
# repurposemap.evaluation.random_baseline for the full explanation.
RANDOM_BASELINE_METHODOLOGY = (
    "candidate-conditioned: shuffles only the path baseline's own discovered "
    "candidate list for each disease, not the complete drug universe. "
    "Answers how much of the path baseline's signal is from finding the "
    "right candidates at all, not how a baseline with no graph knowledge "
    "would do. Contrast with the popularity baseline, which ranks the "
    "complete drug universe independent of the path baseline's candidates."
)


def get_git_commit(repo_dir: str | Path | None = None) -> tuple[str | None, bool | None]:
    """(commit hash, is_dirty), best-effort. (None, None) if git or the repo is unavailable.

    Never raises: a missing git binary, a directory that is not a checkout,
    or any other failure is reported as unknown, not invented.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=True, timeout=10
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None


def _metrics_dict(result: DiseaseEvaluationResult, hits_at, mrr: float) -> dict:
    return {f"hits@{k}": hits_at(k) for k in result.config.top_ks} | {"mrr": mrr}


@dataclass(frozen=True)
class BenchmarkArtifact:
    artifact_schema_version: int
    provenance: PrimeKGFingerprint
    git_commit: str | None
    git_dirty: bool | None
    config: dict
    n_train_diseases: int
    n_test_diseases: int
    n_train_indications: int
    n_test_indications: int
    popularity_universe_size: int
    path_metrics: dict
    random_metrics: dict
    random_baseline_methodology: str
    popularity_metrics: dict
    n_diseases_with_candidate_cap_hit: int
    n_diseases_with_truncated_searches: int
    n_diseases_with_backward_search_truncated: int
    runtime_s: float
    generated_at: str

    def to_dict(self) -> dict:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "provenance": {
                "sha256": self.provenance.sha256,
                "size_bytes": self.provenance.size_bytes,
                "retrieved_date": self.provenance.retrieved_date,
                "dataverse_doi": self.provenance.dataverse_doi,
                "release_version": self.provenance.release_version,
                "snapshot_build_date": self.provenance.snapshot_build_date,
            },
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config": self.config,
            "n_train_diseases": self.n_train_diseases,
            "n_test_diseases": self.n_test_diseases,
            "n_train_indications": self.n_train_indications,
            "n_test_indications": self.n_test_indications,
            "popularity_universe_size": self.popularity_universe_size,
            "path_metrics": self.path_metrics,
            "random_metrics": self.random_metrics,
            "random_baseline_methodology": self.random_baseline_methodology,
            "popularity_metrics": self.popularity_metrics,
            "n_diseases_with_candidate_cap_hit": self.n_diseases_with_candidate_cap_hit,
            "n_diseases_with_truncated_searches": self.n_diseases_with_truncated_searches,
            "n_diseases_with_backward_search_truncated": self.n_diseases_with_backward_search_truncated,
            "runtime_s": self.runtime_s,
            "generated_at": self.generated_at,
        }


def build_artifact(
    result: DiseaseEvaluationResult,
    fingerprint: PrimeKGFingerprint,
    *,
    popularity_universe_size: int,
    repo_dir: str | Path | None = None,
) -> BenchmarkArtifact:
    """Assemble a `BenchmarkArtifact` from a finished evaluation result."""
    git_commit, git_dirty = get_git_commit(repo_dir)
    return BenchmarkArtifact(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        provenance=fingerprint,
        git_commit=git_commit,
        git_dirty=git_dirty,
        config=result.config.to_dict(),
        n_train_diseases=result.n_train_diseases,
        n_test_diseases=result.n_test_diseases,
        n_train_indications=result.n_train_indications,
        n_test_indications=result.n_test_indications,
        popularity_universe_size=popularity_universe_size,
        path_metrics=_metrics_dict(result, result.hits_at, result.mrr),
        random_metrics=_metrics_dict(result, result.random_hits_at, result.random_mrr),
        random_baseline_methodology=RANDOM_BASELINE_METHODOLOGY,
        popularity_metrics=_metrics_dict(result, result.popularity_hits_at, result.popularity_mrr),
        n_diseases_with_candidate_cap_hit=result.n_diseases_with_candidate_cap_hit,
        n_diseases_with_truncated_searches=result.n_diseases_with_truncated_searches,
        n_diseases_with_backward_search_truncated=result.n_diseases_with_backward_search_truncated,
        runtime_s=result.runtime_s,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
