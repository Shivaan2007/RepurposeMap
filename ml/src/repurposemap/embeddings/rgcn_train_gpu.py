"""The real R-GCN training loop: full-graph message passing once per epoch,
batched negative sampling, full RNG-state checkpointing, and optional
validation tracking.

Mirrors ``transe_train_gpu.py``'s structure wherever the two trainers agree
(device-aware config, batched sampler, checkpoint/resume, validation
cadence, early stopping, non-finite guards), importing the shared
checkpoint/RNG helpers from ``checkpointing.py`` rather than duplicating
them. Diverges exactly where R-GCN's architecture forces it to: see
``train_gpu``'s docstring for the one-encoder-forward-per-epoch strategy
this loop uses (the same reasoning as ``rgcn_train.py``'s toy loop, just
with device placement, batched sampling, and checkpointing layered on).

Like ``transe_train_gpu.py``, this module is implemented and tested here
against tiny synthetic data and a small bounded local profiling subset
only. The actual GPU-scale search and final training are a later,
separate, explicitly-approved milestone -- this module's own docstring and
the project's current README both say so.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from repurposemap.embeddings.batched_negative_sampling import BatchedTypeAwareNegativeSampler
from repurposemap.embeddings.checkpointing import (
    _restore_rng_state,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from repurposemap.embeddings.negative_sampling import TypePools
from repurposemap.embeddings.positive_store import PositiveTripleStore, build_positive_store
from repurposemap.embeddings.rgcn_model import RGCN, build_message_passing_edges
from repurposemap.embeddings.transe_train import margin_ranking_loss, seed_everything

if TYPE_CHECKING:
    from repurposemap.embeddings.export import LoadedExport


@dataclass(frozen=True)
class RGCNGPUTrainingConfig:
    """Every real-trainer knob, with no hidden defaults -- mirrors
    ``GPUTrainingConfig``'s discipline, with ``hidden_dim``/``num_bases``/
    ``num_layers`` in place of ``embedding_dim``/``distance``."""

    hidden_dim: int = 64
    num_bases: int = 8
    num_layers: int = 2
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
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be at least 1, got {self.hidden_dim}")
        if self.num_bases < 1:
            raise ValueError(f"num_bases must be at least 1, got {self.num_bases}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")
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
            "hidden_dim": self.hidden_dim,
            "num_bases": self.num_bases,
            "num_layers": self.num_layers,
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


# Exactly 3 predefined configs -- not a grid search, same discipline as
# TransE's own SEARCH_CONFIGS. `margin`/`learning_rate`/`negatives_per_positive`
# fixed; only `hidden_dim`/`num_bases` vary.
#
# `rgcn-moreBases` uses 30 bases, deliberately close to the message-passing
# relation count (66 = 2 * 33 real relations) -- a high-capacity, low-
# compression contrast against `rgcn-base`'s 8, not literally "one basis
# per relation" (30 < 66).
SEARCH_CONFIGS: tuple[tuple[str, RGCNGPUTrainingConfig], ...] = (
    ("rgcn-base", RGCNGPUTrainingConfig(hidden_dim=64, num_bases=8, margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
    ("rgcn-wide", RGCNGPUTrainingConfig(hidden_dim=128, num_bases=8, margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
    ("rgcn-moreBases", RGCNGPUTrainingConfig(hidden_dim=64, num_bases=30, margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    epoch_seconds: float
    sampler_stats: dict
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
    model: RGCN
    optimizer: torch.optim.Optimizer
    history: list[EpochRecord]
    final_epoch: int
    stopped_early: bool
    shuffle_generator: torch.Generator
    sampler: BatchedTypeAwareNegativeSampler


def train_gpu(
    triples: np.ndarray,
    n_entities: int,
    n_relations: int,
    type_pools: TypePools,
    positive_store: PositiveTripleStore,
    config: RGCNGPUTrainingConfig,
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
    """Train (or resume training) an R-GCN model on ``triples``.

    Same validation/resume/checkpoint/early-stopping contract as
    ``transe_train_gpu.train_gpu`` -- see that function's docstring for the
    shared parts. What's different here, and the reason this isn't just a
    copy-paste: TransE scores each triple independently, so its loop does
    pure mini-batch SGD (one ``optimizer.step()`` per batch). R-GCN's
    ``score`` needs entity embeddings that come from running message
    passing over the *whole* graph first -- computing those once, then
    stepping the optimizer after every mini-batch while reusing that same
    (now-stale) embeddings tensor, trains against representations that no
    longer match the model's current parameters, with no clean autograd
    graph left to recompute them from.

    So this loop runs the encoder **exactly once per epoch** (edges are
    static across epochs -- ``triples`` doesn't change mid-run), scores
    every mini-batch against that single embeddings tensor, and takes
    exactly one ``optimizer.step()`` per epoch. ``checkpoint_every`` and
    ``validate_every`` still operate at epoch granularity, same as TransE's
    trainer -- only the *inside* of an epoch changed shape, not the outer
    epoch-loop contract resume/checkpointing depend on.

    **Gradient accumulation, not "stack every batch's loss then one big
    backward."** An earlier version of this loop collected every batch's
    loss into a list and called ``.backward()`` once on their mean at the
    end of the epoch. Profiled against real PrimeKG-scale data (see
    ``ml/experiments/rgcn_gpu_feasibility_profile_1364665.json``), that
    design retained every mini-batch's forward computation graph
    simultaneously until that single end-of-epoch backward -- peak memory
    scaled with the number of batches, not just the encoder's edge count,
    projecting to ~59 GB at full PrimeKG scale against a 14.6 GB T4. This
    loop instead calls ``backward()`` once per batch (each scaled by
    ``1 / n_batches`` so the accumulated gradient equals the same true
    mean-loss gradient as before) with ``retain_graph=True``, so the one
    shared ``entity_embeddings`` graph survives repeated backward calls,
    but each batch's own small scoring subgraph is freed as soon as that
    batch's loss variable goes out of scope rather than staying alive for
    the whole epoch. Still exactly one ``optimizer.step()`` per epoch --
    only the backward mechanics changed.

    ``triples`` is whichever array is "live" for the current phase --
    ``tuning_triples`` during the search, ``final_triples`` for final
    training. This function never reads, and has no parameter for,
    locked-test labels. The message-passing edges it builds internally
    (via ``build_message_passing_edges``) are used only for the encoder's
    forward pass; negative sampling and the positive-triple store continue
    to operate on the original, undoubled ``triples``/``positive_store``.
    """
    device = torch.device(config.device)
    seed_everything(config.seed)

    model = RGCN(
        n_entities, n_relations, hidden_dim=config.hidden_dim, num_bases=config.num_bases, num_layers=config.num_layers
    ).to(device)
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

    edge_h_np, edge_r_np, edge_t_np = build_message_passing_edges(triples, n_relations)
    edge_h = torch.as_tensor(edge_h_np, dtype=torch.long, device=device)
    edge_r = torch.as_tensor(edge_r_np, dtype=torch.long, device=device)
    edge_t = torch.as_tensor(edge_t_np, dtype=torch.long, device=device)

    entity_embeddings: torch.Tensor | None = None

    def _live_model_score_fn(disease_id, candidate_drug_ids):
        """Scores every candidate drug against this epoch's just-computed
        entity embeddings -- reads ``entity_embeddings`` from the enclosing
        scope at call time (always right after that epoch's one encoder
        forward pass), not at closure-definition time."""
        disease_idx = entity_index.id_to_index[disease_id]
        drug_indices = torch.as_tensor(
            [entity_index.id_to_index[d] for d in candidate_drug_ids], dtype=torch.long, device=device
        )
        rel = torch.full_like(drug_indices, indication_relation_idx)
        tail = torch.full_like(drug_indices, disease_idx)
        with torch.no_grad():
            return model.score(entity_embeddings, drug_indices, rel, tail).cpu().numpy()

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

        # Exactly one encoder forward pass this epoch -- see this
        # function's docstring for why it must not run per batch.
        entity_embeddings = model.encode(edge_h, edge_r, edge_t, n_entities)

        permutation = torch.randperm(n, generator=shuffle_generator)
        n_batches_total = (n + batch_size - 1) // batch_size
        batch_start_indices = list(range(0, n, batch_size))

        # True per-batch gradient accumulation, not "stack every batch's
        # loss then one big backward": measured on real PrimeKG-scale data
        # (see ml/experiments/rgcn_gpu_feasibility_profile_1364665.json),
        # the latter retains every mini-batch's forward computation graph
        # simultaneously until the single epoch-end backward, which is
        # what actually drove peak memory to scale with batch count (not
        # just encoder edge count) -- projected ~59 GB at full PrimeKG
        # scale against a 14.6 GB T4. This loop instead calls
        # ``backward()`` once per batch (scaled by 1/n_batches_total so
        # the accumulated gradient equals the true mean-loss gradient,
        # exactly as before), with ``retain_graph=True`` so the single
        # shared ``entity_embeddings`` graph survives repeated backward
        # calls -- but each batch's own small scoring subgraph is freed as
        # soon as that batch's loss variable goes out of scope, instead of
        # all of them staying alive for the whole epoch. Still exactly one
        # ``optimizer.step()`` per epoch -- only the backward mechanics
        # changed, not the epoch-level contract checkpointing/resume rely on.
        optimizer.zero_grad()
        batch_loss_values: list[float] = []

        for start in batch_start_indices:
            idx = permutation[start : start + batch_size]
            batch_h, batch_r, batch_t = h_all[idx], r_all[idx], t_all[idx]
            batch_n = len(idx)

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

            pos_scores = model.score(entity_embeddings, batch_h, batch_r, batch_t)
            neg_scores = model.score(
                entity_embeddings, neg_h.reshape(-1), neg_r.reshape(-1), neg_t.reshape(-1)
            ).reshape(batch_n, config.negatives_per_positive)

            batch_loss = margin_ranking_loss(pos_scores, neg_scores, config.margin, valid_mask=valid_mask)
            if not torch.isfinite(batch_loss):
                raise RuntimeError(f"non-finite loss detected at epoch {epoch}: {batch_loss.item()}")

            (batch_loss / n_batches_total).backward(retain_graph=True)
            batch_loss_values.append(float(batch_loss.item()))

        optimizer.step()
        epoch_loss_value = sum(batch_loss_values) / max(len(batch_loss_values), 1)

        entity_weights = model.encoder.entity_embeddings.weight.detach()
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
                train_loss=epoch_loss_value,
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
# Validation search and final training -- orchestration on top of
# train_gpu. Neither function ever reads or requires a locked-test
# manifest: they operate entirely on ``LoadedExport.tuning_triples``/
# ``final_triples`` and ``LoadedExport.val_indications``.
#
# This milestone implements and tests these only against tiny synthetic
# data -- no real search or final training is run here. See the project
# README / this change's own commit for that explicit scope boundary.
# --------------------------------------------------------------------------


def run_validation_search(
    loaded_export: "LoadedExport",
    *,
    search_configs: tuple[tuple[str, RGCNGPUTrainingConfig], ...] = SEARCH_CONFIGS,
    epochs: int = 10,
    validate_every: int = 2,
    output_log_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Train each of the (exactly 3) predefined configs on ``tuning_triples``,
    evaluate each on validation diseases only, and apply the existing
    selection rule (``repurposemap.evaluation.transe_baseline``, unchanged
    and reused as-is -- it is model-agnostic).

    ``SearchCandidateResult.embedding_dim`` is populated with each config's
    ``hidden_dim`` -- that field's name is TransE-flavored, but the
    function it feeds (``select_best_candidate``) uses it only as "the
    model's capacity knob to prefer the smaller of, on a remaining tie",
    which applies identically here.

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
                embedding_dim=config.hidden_dim,
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
    selected_config: RGCNGPUTrainingConfig,
    epochs: int,
    search_log: dict,
    checkpoint_path: str | Path,
    primekg_fingerprint: dict | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    device: str | None = None,
) -> dict:
    """Train exactly once on ``final_triples`` (validation diseases'
    indications restored) with the already-selected config, and save a
    fully self-describing final checkpoint -- same shape and same
    eligibility-gate fields as TransE's ``train_final``
    (``is_final_training``, ``export_fingerprint``,
    ``locked_test_manifest_fingerprint``), so
    ``assert_eligible_for_locked_test_evaluation`` accepts either model's
    checkpoint unchanged.

    ``epochs`` must be decided from the winning search run's observed
    plateau *before* this is called -- there is no validation signal
    during final training (validation diseases are now training data), so
    no early stopping is used here, by design.
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
                "entity embeddings produced by relation-wise mean-normalized message passing "
                "(no post-hoc normalization step); DistMult relation embeddings not normalized"
            ),
        },
    )
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint
