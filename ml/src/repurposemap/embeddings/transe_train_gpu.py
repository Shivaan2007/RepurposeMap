"""The real TransE training loop: full-batch iteration, batched negative
sampling, full RNG-state checkpointing, and optional validation tracking.

Separate from ``transe_train.py``'s ``train_toy``, which stays exactly as
it is -- a tiny-synthetic-graph sanity check (Stage A). This module is
designed for the real, full-scale leakage-safe triple set (tuning or final,
~8.1M rows on real PrimeKG), consumes only plain arrays/dataclasses (no
NetworkX graph object, no PrimeKG CSV -- everything it needs comes from
``repurposemap.embeddings.export.load_export``'s output), and runs on CPU
or CUDA via ``config.device``.

Stage D explicitly does not run this against the real, full triple set yet
-- it is implemented and tested here against tiny synthetic data and a
small bounded local profiling subset only. The actual GPU-scale search and
final training happen afterward, by hand, on Kaggle/Colab.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from repurposemap.embeddings.batched_negative_sampling import BatchedTypeAwareNegativeSampler
from repurposemap.embeddings.negative_sampling import TypePools
from repurposemap.embeddings.positive_store import PositiveTripleStore, build_positive_store
from repurposemap.embeddings.transe_model import TransE
from repurposemap.embeddings.transe_train import margin_ranking_loss, seed_everything

if TYPE_CHECKING:
    from repurposemap.embeddings.export import LoadedExport

Distance = Literal["l1", "l2"]


@dataclass(frozen=True)
class GPUTrainingConfig:
    """Every real-trainer knob, with no hidden defaults -- mirrors
    ``TransETrainingConfig``'s discipline, extended with the knobs a
    full-scale, resumable, device-aware trainer needs that a toy loop does
    not (device, checkpoint cadence, validation cadence, early stopping,
    batched-sampler retry bounds)."""

    embedding_dim: int = 128
    distance: Distance = "l2"
    margin: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 1024
    negatives_per_positive: int = 4
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"
    checkpoint_every: int = 1
    max_vectorized_retries: int = 5
    max_scalar_retries: int = 10
    validate_every: int | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be at least 1, got {self.embedding_dim}")
        if self.distance not in ("l1", "l2"):
            raise ValueError(f"distance must be 'l1' or 'l2', got {self.distance!r}")
        if self.margin <= 0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.negatives_per_positive < 1:
            raise ValueError(f"negatives_per_positive must be at least 1, got {self.negatives_per_positive}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be at least 1, got {self.checkpoint_every}")
        if self.max_vectorized_retries < 1:
            raise ValueError(f"max_vectorized_retries must be at least 1, got {self.max_vectorized_retries}")
        if self.max_scalar_retries < 1:
            raise ValueError(f"max_scalar_retries must be at least 1, got {self.max_scalar_retries}")
        if self.validate_every is not None and self.validate_every < 1:
            raise ValueError(f"validate_every must be at least 1 or None, got {self.validate_every}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be at least 1 or None, got {self.early_stopping_patience}")

    def to_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "distance": self.distance,
            "margin": self.margin,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "negatives_per_positive": self.negatives_per_positive,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.device,
            "checkpoint_every": self.checkpoint_every,
            "max_vectorized_retries": self.max_vectorized_retries,
            "max_scalar_retries": self.max_scalar_retries,
            "validate_every": self.validate_every,
            "early_stopping_patience": self.early_stopping_patience,
        }


# Exactly 3 predefined configs (Stage D §4) -- not a grid search. Each
# entry is (name, config); `negatives_per_positive`/`margin`/`lr` fixed,
# only `embedding_dim`/`distance` vary, to keep the search to 3 runs.
SEARCH_CONFIGS: tuple[tuple[str, GPUTrainingConfig], ...] = (
    ("search-dim64", GPUTrainingConfig(embedding_dim=64, distance="l2", margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
    ("search-dim128", GPUTrainingConfig(embedding_dim=128, distance="l2", margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
    ("search-l1", GPUTrainingConfig(embedding_dim=64, distance="l1", margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    epoch_seconds: float
    sampler_stats: dict  # BatchedSamplerStats.to_dict(), cumulative-since-start
    val_metrics: dict | None = None

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "epoch_seconds": self.epoch_seconds,
            "sampler_stats": self.sampler_stats,
            "val_metrics": self.val_metrics,
        }


@dataclass
class GPUTrainingResult:
    model: TransE
    optimizer: torch.optim.Optimizer
    history: list[EpochRecord]
    final_epoch: int
    stopped_early: bool
    shuffle_generator: torch.Generator
    sampler: BatchedTypeAwareNegativeSampler


def _rng_state_dict(shuffle_generator: torch.Generator, sampler: BatchedTypeAwareNegativeSampler) -> dict:
    """Every RNG source this trainer uses -- see the Stage D plan's §3/§7.

    Not claimed to give exact-resume reproducibility anywhere except where
    the resume test (``tests/test_transe_train_gpu.py``) actually proves it.
    """
    cuda_states: list[Any] = []
    if torch.cuda.is_available():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "shuffle_generator_state": shuffle_generator.get_state().clone(),
        "numpy_sampler_bit_generator_state": sampler.get_state()["numpy_bit_generator_state"],
        "scalar_fallback_random_state": sampler.get_state()["scalar_random_state"],
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": cuda_states,
    }


def _restore_rng_state(rng_state: dict, shuffle_generator: torch.Generator, sampler: BatchedTypeAwareNegativeSampler) -> None:
    shuffle_generator.set_state(rng_state["shuffle_generator_state"])
    sampler.set_state(
        {
            "numpy_bit_generator_state": rng_state["numpy_sampler_bit_generator_state"],
            "scalar_random_state": rng_state["scalar_fallback_random_state"],
        }
    )
    torch.set_rng_state(rng_state["torch_cpu_rng_state"])
    if rng_state["torch_cuda_rng_states"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda_rng_states"])


def build_checkpoint(
    *,
    model: TransE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[EpochRecord],
    config: GPUTrainingConfig,
    shuffle_generator: torch.Generator,
    sampler: BatchedTypeAwareNegativeSampler,
    is_final_training: bool = False,
    extra: dict | None = None,
) -> dict:
    """The checkpoint dict saved to disk. ``extra`` adds final-training-only
    fields (export fingerprint, hyperparameter search summary, etc. -- see
    ``train_final``) without this function needing to know about them."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": [record.to_dict() for record in history],
        "config": config.to_dict(),
        "rng_state": _rng_state_dict(shuffle_generator, sampler),
        "is_final_training": is_final_training,
    }
    if extra:
        checkpoint.update(extra)
    return checkpoint


def save_checkpoint(checkpoint: dict, path: str | Path) -> None:
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, weights_only=False)


def train_gpu(
    triples: np.ndarray,
    n_entities: int,
    n_relations: int,
    type_pools: TypePools,
    positive_store: PositiveTripleStore,
    config: GPUTrainingConfig,
    *,
    resume_from: dict | None = None,
    validation_score_fn=None,
    entity_index=None,
    indication_relation_idx: int | None = None,
    val_disease_ids=None,
    val_indications=None,
    drug_ids=None,
    checkpoint_path: str | Path | None = None,
) -> GPUTrainingResult:
    """Train (or resume training) a TransE model on ``triples``.

    Two ways to validate: pass ``validation_score_fn`` directly (useful for
    tests, or any scoring decoupled from this call's own model), or pass
    ``entity_index`` + ``indication_relation_idx`` and this function builds
    the real score function itself, bound to the model it is training --
    the only way to validate against live, in-progress weights, since the
    model does not exist until this call creates it. If both are omitted,
    ``config.validate_every`` must be ``None``.

    ``triples`` is whichever array is "live" for the current phase --
    ``tuning_triples`` during the §5 search, ``final_triples`` for final
    training (§9). This function never reads, and has no parameter for,
    locked-test labels.

    If ``config.validate_every`` is set, ``validation_score_fn``,
    ``val_disease_ids``, ``val_indications``, and ``drug_ids`` must all be
    given; every ``validate_every`` epochs,
    ``repurposemap.evaluation.transe_baseline.evaluate_transe_on_diseases``
    is called to log Hits@K/MRR/AP, and (if ``config.early_stopping_patience``
    is set) to decide whether to stop early -- driven only by validation
    MRR, never by anything else.
    """
    device = torch.device(config.device)
    seed_everything(config.seed)

    model = TransE(n_entities, n_relations, embedding_dim=config.embedding_dim, distance=config.distance).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    shuffle_generator = torch.Generator().manual_seed(config.seed)
    sampler = BatchedTypeAwareNegativeSampler(
        type_pools,
        n_entities=n_entities,
        n_relations=n_relations,
        positive_store=positive_store,
        max_vectorized_retries=config.max_vectorized_retries,
        max_scalar_retries=config.max_scalar_retries,
        seed=config.seed,
    )

    def _live_model_score_fn(disease_id, candidate_drug_ids):
        """Scores every candidate drug against the model as it exists
        *right now* -- built here, not passed in, because the model this
        closes over does not exist until the lines above create it."""
        disease_idx = entity_index.id_to_index[disease_id]
        drug_indices = torch.as_tensor(
            [entity_index.id_to_index[d] for d in candidate_drug_ids], dtype=torch.long, device=device
        )
        rel = torch.full_like(drug_indices, indication_relation_idx)
        tail = torch.full_like(drug_indices, disease_idx)
        with torch.no_grad():
            return model.score(drug_indices, rel, tail).cpu().numpy()

    active_score_fn = validation_score_fn
    if active_score_fn is None and entity_index is not None and indication_relation_idx is not None:
        active_score_fn = _live_model_score_fn

    history: list[EpochRecord] = []
    start_epoch = 0
    best_val_mrr = float("-inf")
    epochs_without_improvement = 0

    if resume_from is not None:
        model.load_state_dict(resume_from["state_dict"])
        optimizer.load_state_dict(resume_from["optimizer_state_dict"])
        _restore_rng_state(resume_from["rng_state"], shuffle_generator, sampler)
        start_epoch = resume_from["epoch"] + 1
        history = [
            EpochRecord(
                epoch=r["epoch"], train_loss=r["train_loss"], epoch_seconds=r["epoch_seconds"],
                sampler_stats=r["sampler_stats"], val_metrics=r["val_metrics"],
            )
            for r in resume_from["history"]
        ]
        for record in history:
            if record.val_metrics is not None:
                best_val_mrr = max(best_val_mrr, record.val_metrics["mrr"])
        if history:
            # sampler.stats is cumulative-since-this-call by construction
            # (a fresh BatchedTypeAwareNegativeSampler was just created
            # above); restore it to the cumulative count at the point of
            # the checkpoint, so continuing training keeps reporting a
            # true running total across the resume, not a count that
            # silently resets to zero.
            last_stats = history[-1].sampler_stats
            sampler.stats.n_requested_negatives = last_stats["n_requested_negatives"]
            sampler.stats.n_initial_collisions = last_stats["n_initial_collisions"]
            sampler.stats.n_retry_rounds = last_stats["n_retry_rounds"]
            sampler.stats.n_retry_candidates_drawn = last_stats["n_retry_candidates_drawn"]
            sampler.stats.n_scalar_fallbacks = last_stats["n_scalar_fallbacks"]
            sampler.stats.n_unresolved = last_stats["n_unresolved"]

    h_all = torch.as_tensor(triples[:, 0], dtype=torch.long, device=device)
    r_all = torch.as_tensor(triples[:, 1], dtype=torch.long, device=device)
    t_all = torch.as_tensor(triples[:, 2], dtype=torch.long, device=device)
    n = len(triples)
    batch_size = min(config.batch_size, n)

    stopped_early = False

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.monotonic()
        permutation = torch.randperm(n, generator=shuffle_generator)
        epoch_loss_total, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_h, batch_r, batch_t = h_all[idx], r_all[idx], t_all[idx]
            batch_n = len(idx)

            # Negative sampling runs on CPU/NumPy regardless of `device`
            # (PositiveTripleStore/TypePools are NumPy-based); only the
            # resulting small index tensors move to `device`.
            batch_h_cpu = batch_h.cpu().numpy()
            batch_r_cpu = batch_r.cpu().numpy()
            batch_t_cpu = batch_t.cpu().numpy()

            neg_h_cols, neg_t_cols, valid_cols = [], [], []
            for _ in range(config.negatives_per_positive):
                neg_h, _neg_r, neg_t, valid = sampler.corrupt_batch(batch_h_cpu, batch_r_cpu, batch_t_cpu)
                neg_h_cols.append(neg_h)
                neg_t_cols.append(neg_t)
                valid_cols.append(valid)

            neg_h = torch.as_tensor(np.stack(neg_h_cols, axis=1), dtype=torch.long, device=device)
            neg_t = torch.as_tensor(np.stack(neg_t_cols, axis=1), dtype=torch.long, device=device)
            valid_mask = torch.as_tensor(np.stack(valid_cols, axis=1), dtype=torch.bool, device=device)
            neg_r = batch_r.unsqueeze(1).repeat(1, config.negatives_per_positive)

            pos_scores = model.score(batch_h, batch_r, batch_t)
            neg_scores = model.score(neg_h.reshape(-1), neg_r.reshape(-1), neg_t.reshape(-1)).reshape(
                batch_n, config.negatives_per_positive
            )

            loss = margin_ranking_loss(pos_scores, neg_scores, config.margin, valid_mask=valid_mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss detected at epoch {epoch}: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_entity_embeddings()

            epoch_loss_total += float(loss.item())
            n_batches += 1

        entity_weights = model.entity_embeddings.weight.detach()
        if not torch.isfinite(entity_weights).all():
            raise RuntimeError(f"non-finite entity embedding detected at the end of epoch {epoch}")
        relation_weights = model.relation_embeddings.weight.detach()
        if not torch.isfinite(relation_weights).all():
            raise RuntimeError(f"non-finite relation embedding detected at the end of epoch {epoch}")

        val_metrics = None
        if config.validate_every is not None and (epoch + 1) % config.validate_every == 0:
            if active_score_fn is None or val_disease_ids is None or val_indications is None or drug_ids is None:
                raise ValueError("validate_every is set but validation inputs were not all provided")
            from repurposemap.evaluation.transe_baseline import evaluate_transe_on_diseases

            val_result = evaluate_transe_on_diseases(active_score_fn, val_disease_ids, val_indications, drug_ids)
            val_metrics = {
                "hits_at_1": val_result.hits_at(1),
                "hits_at_3": val_result.hits_at(3),
                "hits_at_10": val_result.hits_at(10),
                "mrr": val_result.mrr,
                "macro_auprc": val_result.macro_auprc,
                "micro_auprc": val_result.micro_auprc,
            }
            if config.early_stopping_patience is not None:
                if val_metrics["mrr"] > best_val_mrr:
                    best_val_mrr = val_metrics["mrr"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=epoch_loss_total / max(n_batches, 1),
                epoch_seconds=time.monotonic() - epoch_start,
                sampler_stats=sampler.stats.to_dict(),
                val_metrics=val_metrics,
            )
        )

        if checkpoint_path is not None and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint = build_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch, history=history, config=config,
                shuffle_generator=shuffle_generator, sampler=sampler,
            )
            save_checkpoint(checkpoint, checkpoint_path)

        if (
            config.early_stopping_patience is not None
            and val_metrics is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    return GPUTrainingResult(
        model=model,
        optimizer=optimizer,
        history=history,
        final_epoch=history[-1].epoch if history else start_epoch - 1,
        stopped_early=stopped_early,
        shuffle_generator=shuffle_generator,
        sampler=sampler,
    )


# --------------------------------------------------------------------------
# Validation search (Stage D §4/§5/§8) and final training (§6/§9) --
# orchestration on top of train_gpu. Neither function ever reads or
# requires a locked-test manifest: they operate entirely on
# ``LoadedExport.tuning_triples``/``final_triples`` and
# ``LoadedExport.val_indications``.
# --------------------------------------------------------------------------


def run_validation_search(
    loaded_export: "LoadedExport",
    *,
    search_configs: tuple[tuple[str, GPUTrainingConfig], ...] = SEARCH_CONFIGS,
    epochs: int = 10,
    validate_every: int = 2,
    output_log_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Train each of the (exactly 3) predefined configs on ``tuning_triples``,
    evaluate each on validation diseases only, and apply the fixed
    selection rule. Returns -- and, if ``output_log_path`` is given, writes
    -- one flat JSON experiment log with every config's full history, the
    rule's text, and which index it picked.

    ``checkpoint_dir``, if given, is created and each config's checkpoint is
    written to ``checkpoint_dir / f"{name}.pt"`` (via ``train_gpu``'s own
    ``checkpoint_every``-driven saving, unchanged) -- so every config's full
    model state/RNG state is preserved, not just its logged metric history.

    Never touches locked-test data: ``loaded_export`` has none to touch.
    """
    from repurposemap.evaluation.transe_baseline import SearchCandidateResult, select_best_candidate

    tuning_triples = loaded_export.tuning_triples
    n_entities = len(loaded_export.entity_index)
    n_relations = len(loaded_export.relation_index)
    positive_store = build_positive_store(tuning_triples, n_relations, n_entities)

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    candidates: list[SearchCandidateResult] = []
    for name, base_config in search_configs:
        config = replace(base_config, epochs=epochs, validate_every=validate_every)
        checkpoint_path = Path(checkpoint_dir) / f"{name}.pt" if checkpoint_dir is not None else None
        started = time.monotonic()
        result = train_gpu(
            tuning_triples, n_entities, n_relations, loaded_export.type_pools, positive_store, config,
            entity_index=loaded_export.entity_index, indication_relation_idx=loaded_export.indication_relation_idx,
            val_disease_ids=loaded_export.val_disease_ids, val_indications=loaded_export.val_indications,
            drug_ids=loaded_export.drug_ids, checkpoint_path=checkpoint_path,
        )
        runtime_s = time.monotonic() - started

        val_records = [r for r in result.history if r.val_metrics is not None]
        best_record = max(val_records, key=lambda r: r.val_metrics["mrr"]) if val_records else None
        candidates.append(
            SearchCandidateResult(
                name=name,
                embedding_dim=config.embedding_dim,
                val_mrr=best_record.val_metrics["mrr"] if best_record else float("-inf"),
                val_macro_auprc=best_record.val_metrics["macro_auprc"] if best_record else 0.0,
            )
        )
        runs.append(
            {
                "name": name,
                "config": config.to_dict(),
                "history": [r.to_dict() for r in result.history],
                "runtime_s": runtime_s,
                "stopped_early": result.stopped_early,
                "best_val_epoch": best_record.epoch if best_record else None,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            }
        )

    selected_index = select_best_candidate(candidates)

    from repurposemap.evaluation.transe_baseline import SELECTION_RULE_DESCRIPTION

    log = {
        "rule": SELECTION_RULE_DESCRIPTION,
        "candidates": [
            {"name": c.name, "embedding_dim": c.embedding_dim, "val_mrr": c.val_mrr, "val_macro_auprc": c.val_macro_auprc}
            for c in candidates
        ],
        "selected_index": selected_index,
        "selected_name": candidates[selected_index].name,
        "runs": runs,
    }
    if output_log_path is not None:
        Path(output_log_path).write_text(json.dumps(log, indent=2))
    return log


def train_final(
    loaded_export: "LoadedExport",
    *,
    selected_config: GPUTrainingConfig,
    epochs: int,
    search_log: dict,
    checkpoint_path: str | Path,
    primekg_fingerprint: dict | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    device: str | None = None,
) -> dict:
    """Train exactly once on ``final_triples`` (validation diseases' indications
    restored) with the already-selected config, and save a fully
    self-describing final checkpoint.

    ``epochs`` must be decided from the winning §5 search run's observed
    plateau *before* this is called -- there is no validation signal
    during final training (validation diseases are now training data), so
    no early stopping is used here, by design.

    ``device``, if given, overrides ``selected_config.device`` (which
    defaults to ``"cpu"``) -- the selected config's hyperparameters
    (embedding_dim/distance/margin/lr/negatives_per_positive) are
    untouched either way; this only controls which device the identical
    computation runs on.
    """
    device_override = {"device": device} if device is not None else {}
    final_config = replace(
        selected_config, epochs=epochs, validate_every=None, early_stopping_patience=None, **device_override,
    )
    final_triples = loaded_export.final_triples
    n_entities = len(loaded_export.entity_index)
    n_relations = len(loaded_export.relation_index)
    positive_store = build_positive_store(final_triples, n_relations, n_entities)

    result = train_gpu(final_triples, n_entities, n_relations, loaded_export.type_pools, positive_store, final_config)

    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=final_config, shuffle_generator=result.shuffle_generator, sampler=result.sampler,
        is_final_training=True,
        extra={
            "entity_index": {"index_to_id": list(loaded_export.entity_index.index_to_id)},
            "relation_index": {"index_to_id": [list(p) for p in loaded_export.relation_index.index_to_id]},
            "indication_relation_idx": loaded_export.indication_relation_idx,
            "indication_relation_key": loaded_export.metadata["indication_relation_key"],
            "primekg_fingerprint": primekg_fingerprint if primekg_fingerprint is not None else loaded_export.metadata.get("primekg_fingerprint"),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "export_fingerprint": loaded_export.metadata["export_fingerprint"],
            # The EXPECTED fingerprint of the locked-test manifest this
            # checkpoint should eventually be evaluated against -- just a
            # hash, propagated from metadata.json (itself just a hash, no
            # labels). assert_eligible_for_locked_test_evaluation compares
            # this against a freshly recomputed fingerprint of whatever
            # manifest is actually supplied at evaluation time.
            "locked_test_manifest_fingerprint": loaded_export.metadata.get("locked_test_manifest_fingerprint"),
            "hyperparameter_search": {
                "rule": search_log["rule"],
                "candidates": search_log["candidates"],
                "selected_index": search_log["selected_index"],
                "selected_name": search_log["selected_name"],
            },
            "final_train_loss": result.history[-1].train_loss if result.history else None,
            "train_loss_curve": [r.train_loss for r in result.history],
            "normalization_assumptions": (
                "entity embeddings L2-unit-normalized after every optimizer step; "
                "relation embeddings not normalized"
            ),
        },
    )
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint
