"""The scalable, sampled-subgraph R-GCN training loop.

Sibling to ``rgcn_train_gpu.py``, not a replacement -- the full-graph
trainer stays exactly as it is, a documented feasibility baseline (see
``ml/experiments/rgcn_gpu_fullscale_profile_1ae8db5.json`` and
``ml/experiments/rgcn_gpu_nockpt_profile_1ae8db5.json`` for why full-graph
training is memory-feasible but operationally impractical at real
PrimeKG scale: ~2.69 h per epoch, dominated by gradient-checkpoint
recomputation through a graph shared across every mini-batch).

**The core shape change.** The full-graph trainer computes one shared set
of entity embeddings per epoch (via one expensive full-graph forward),
scores every mini-batch against that single tensor, and takes exactly one
``optimizer.step()`` per epoch -- the whole design exists because
batching the OPTIMIZER STEP, not the forward pass, was the only way to
avoid re-deriving entity embeddings mid-epoch. This trainer removes that
constraint entirely by sampling a small, bounded neighborhood fresh for
every mini-batch (``repurposemap.embeddings.rgcn_sampling``): each batch
gets its own independent encoder forward pass over just its sampled
subgraph, its own backward, and its own ``optimizer.step()`` -- ordinary
mini-batch SGD, the same shape TransE's trainer already uses. No shared
graph persists across batches, so there is no ``retain_graph=True``
anywhere in this file.

Implemented and tested here only against tiny synthetic data -- no real
PrimeKG training happens in this module's own test suite. Real-data
feasibility profiling (sampled batch throughput, memory, projected epoch
runtime) is a separate, later, explicitly-approved step, same precedent
as every other real-GPU-scale milestone in this project.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

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
from repurposemap.embeddings.rgcn_model import RGCN
from repurposemap.embeddings.rgcn_sampling import (
    RelationAdjacency,
    RelationAwareNeighborSampler,
    SamplingConfig,
    build_relation_adjacency,
)
from repurposemap.embeddings.transe_train import margin_ranking_loss, seed_everything


@dataclass(frozen=True)
class RGCNSampledTrainingConfig:
    """Every sampled-trainer knob, with no hidden defaults -- same
    discipline as ``RGCNGPUTrainingConfig``, extended with the two
    neighborhood-sampling fanouts. No ``checkpoint_every``-per-epoch
    constraint on the optimizer step (there isn't one here to begin with
    -- every batch already gets its own step), but ``checkpoint_every``
    is kept as an epoch-cadence knob for writing checkpoints to disk,
    same meaning as in the full-graph trainer.
    """

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
    fanout_hop1: int = 15
    fanout_hop2: int = 10
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
        if self.fanout_hop1 < 1:
            raise ValueError(f"fanout_hop1 must be at least 1, got {self.fanout_hop1}")
        if self.fanout_hop2 < 1:
            raise ValueError(f"fanout_hop2 must be at least 1, got {self.fanout_hop2}")
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

    def sampling_config(self) -> SamplingConfig:
        return SamplingConfig(fanout_hop1=self.fanout_hop1, fanout_hop2=self.fanout_hop2, seed=self.seed)

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
            "fanout_hop1": self.fanout_hop1,
            "fanout_hop2": self.fanout_hop2,
            "checkpoint_every": self.checkpoint_every,
            "max_vectorized_retries": self.max_vectorized_retries,
            "max_scalar_retries": self.max_scalar_retries,
            "validate_every": self.validate_every,
            "early_stopping_patience": self.early_stopping_patience,
        }


# The same exactly-3 predeclared configs as the full-graph trainer's
# SEARCH_CONFIGS (rgcn_train_gpu.py) -- this is a deliberate, explicit
# reuse of that same model-capacity decision, not a second independent
# choice. `fanout_hop1`/`fanout_hop2` use RGCNSampledTrainingConfig's
# approved defaults (15/10, fixed by the feasibility profiling in
# ml/experiments/rgcn_sampled_training_profile_e6ce5d9.json) and are never
# varied across these 3 configs -- only `hidden_dim`/`num_bases` do.
SEARCH_CONFIGS: tuple[tuple[str, RGCNSampledTrainingConfig], ...] = (
    ("rgcn-base", RGCNSampledTrainingConfig(hidden_dim=64, num_bases=8, margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
    ("rgcn-wide", RGCNSampledTrainingConfig(hidden_dim=128, num_bases=8, margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
    ("rgcn-moreBases", RGCNSampledTrainingConfig(hidden_dim=64, num_bases=30, margin=1.0, learning_rate=1e-3, negatives_per_positive=4)),
)


@dataclass
class SampledEpochRecord:
    epoch: int
    train_loss: float
    epoch_seconds: float
    sampler_stats: dict  # negative sampler, cumulative-since-start (BatchedSamplerStats.to_dict())
    subgraph_stats: dict  # this epoch's sampled-subgraph size summary (mean/min/max nodes & edges per batch)
    val_metrics: dict | None = None

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "epoch_seconds": self.epoch_seconds,
            "sampler_stats": self.sampler_stats,
            "subgraph_stats": self.subgraph_stats,
            "val_metrics": self.val_metrics,
        }


@dataclass
class SampledGPUTrainingResult:
    model: RGCN
    optimizer: torch.optim.Optimizer
    history: list[SampledEpochRecord]
    final_epoch: int
    stopped_early: bool
    shuffle_generator: torch.Generator
    sampler: BatchedTypeAwareNegativeSampler
    neighbor_sampler: RelationAwareNeighborSampler
    stopped_for_time_budget: bool = False


def train_sampled_gpu(
    triples: np.ndarray,
    n_entities: int,
    n_relations: int,
    type_pools: TypePools,
    positive_store: PositiveTripleStore,
    adjacency: RelationAdjacency,
    config: RGCNSampledTrainingConfig,
    *,
    resume_from: dict | None = None,
    validation_score_fn=None,
    entity_index=None,
    indication_relation_idx: int | None = None,
    val_disease_ids=None,
    val_indications=None,
    drug_ids=None,
    checkpoint_path: str | Path | None = None,
    max_wall_time_s: float | None = None,
) -> SampledGPUTrainingResult:
    """Train (or resume training) an R-GCN model with sampled-subgraph
    mini-batches. Ordinary SGD: each batch samples its own neighborhood,
    runs its own encoder forward, backward, and ``optimizer.step()`` --
    no shared encoder graph, no ``retain_graph=True``.

    ``max_wall_time_s``, if given, is measured from this call's own start
    (not cumulative across resumed calls -- each process invocation gets
    its own fresh budget). After each epoch finishes and its checkpoint is
    written, if the elapsed wall time has reached this budget, training
    stops there (``stopped_for_time_budget=True`` on the result) rather
    than continuing into another multi-hour epoch -- this is what lets a
    single config's multi-epoch training be split across several
    Kaggle sessions, each with its own hard GPU-time limit, resuming
    cleanly from the last epoch's checkpoint.

    ``triples`` is whichever leakage-safe array is "live" for the current
    phase -- ``tuning_triples`` during search, ``final_triples`` for final
    training -- same convention as the full-graph trainer. ``adjacency``
    must be built from that exact same array (see
    ``repurposemap.embeddings.rgcn_sampling.build_relation_adjacency``);
    this function never reads or requires a locked-test manifest, and has
    no parameter for one.

    Validation, if ``config.validate_every`` is set: an explicit
    ``validation_score_fn`` always wins if given. Otherwise, if
    ``entity_index``/``indication_relation_idx`` are given, this function
    falls back to full-graph no-grad inference over ``triples`` as the
    default validation strategy -- the same approach profiled and
    classified feasible on real PrimeKG in
    ``ml/experiments/rgcn_fullgraph_inference_profile_e79d3e4.json``
    (0.58s encoder forward, ~19% GPU memory), computed exactly once per
    validation event (not once per disease) and reused for every disease
    in that pass, same discipline as every other evaluation path in this
    project. This still never invents a *sampled*-inference strategy --
    it reuses the already-approved full-graph one.
    """
    device = torch.device(config.device)
    seed_everything(config.seed)

    model = RGCN(
        n_entities, n_relations, hidden_dim=config.hidden_dim, num_bases=config.num_bases,
        num_layers=config.num_layers, use_gradient_checkpointing=False,
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
    neighbor_sampler = RelationAwareNeighborSampler(adjacency, n_relations, config.sampling_config(), seed=config.seed)

    history: list[SampledEpochRecord] = []
    start_epoch = 0
    best_val_mrr = float("-inf")
    epochs_without_improvement = 0

    if resume_from is not None:
        model.load_state_dict(resume_from["state_dict"])
        optimizer.load_state_dict(resume_from["optimizer_state_dict"])
        _restore_rng_state(resume_from["rng_state"], shuffle_generator, sampler)
        neighbor_sampler.set_state(resume_from["neighbor_sampler_rng_state"])
        start_epoch = resume_from["epoch"] + 1
        history = [
            SampledEpochRecord(
                epoch=r["epoch"], train_loss=r["train_loss"], epoch_seconds=r["epoch_seconds"],
                sampler_stats=r["sampler_stats"], subgraph_stats=r["subgraph_stats"], val_metrics=r["val_metrics"],
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

    h_all = triples[:, 0]
    r_all = triples[:, 1]
    t_all = triples[:, 2]
    n = len(triples)
    batch_size = min(config.batch_size, n)

    stopped_early = False
    stopped_for_time_budget = False
    call_start = time.monotonic()

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.monotonic()
        permutation = torch.randperm(n, generator=shuffle_generator).numpy()
        epoch_loss_total, n_batches = 0.0, 0
        node_counts: list[int] = []
        edge_counts: list[int] = []

        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_h, batch_r, batch_t = h_all[idx], r_all[idx], t_all[idx]
            batch_n = len(idx)

            neg_h_cols, neg_t_cols, valid_cols = [], [], []
            for _ in range(config.negatives_per_positive):
                neg_h, _neg_r, neg_t, valid = sampler.corrupt_batch(batch_h, batch_r, batch_t)
                neg_h_cols.append(neg_h)
                neg_t_cols.append(neg_t)
                valid_cols.append(valid)
            neg_h_arr = np.stack(neg_h_cols, axis=1)
            neg_t_arr = np.stack(neg_t_cols, axis=1)
            valid_arr = np.stack(valid_cols, axis=1)

            seed_entities = np.unique(np.concatenate([batch_h, batch_t, neg_h_arr.reshape(-1), neg_t_arr.reshape(-1)]))
            subgraph = neighbor_sampler.sample(seed_entities)
            node_counts.append(subgraph.n_local_nodes)
            edge_counts.append(len(subgraph.edge_h))

            local_to_global = torch.as_tensor(subgraph.local_to_global, dtype=torch.long, device=device)
            edge_h_t = torch.as_tensor(subgraph.edge_h, dtype=torch.long, device=device)
            edge_r_t = torch.as_tensor(subgraph.edge_r, dtype=torch.long, device=device)
            edge_t_t = torch.as_tensor(subgraph.edge_t, dtype=torch.long, device=device)

            entity_embeddings = model.encode_sampled(edge_h_t, edge_r_t, edge_t_t, local_to_global)

            local_batch_h = torch.as_tensor(subgraph.to_local(batch_h), dtype=torch.long, device=device)
            local_batch_t = torch.as_tensor(subgraph.to_local(batch_t), dtype=torch.long, device=device)
            local_neg_h = torch.as_tensor(subgraph.to_local(neg_h_arr.reshape(-1)), dtype=torch.long, device=device).reshape(
                batch_n, config.negatives_per_positive
            )
            local_neg_t = torch.as_tensor(subgraph.to_local(neg_t_arr.reshape(-1)), dtype=torch.long, device=device).reshape(
                batch_n, config.negatives_per_positive
            )
            batch_r_t = torch.as_tensor(batch_r, dtype=torch.long, device=device)
            neg_r_t = batch_r_t.unsqueeze(1).repeat(1, config.negatives_per_positive)
            valid_mask = torch.as_tensor(valid_arr, dtype=torch.bool, device=device)

            pos_scores = model.score(entity_embeddings, local_batch_h, batch_r_t, local_batch_t)
            neg_scores = model.score(
                entity_embeddings, local_neg_h.reshape(-1), neg_r_t.reshape(-1), local_neg_t.reshape(-1)
            ).reshape(batch_n, config.negatives_per_positive)

            loss = margin_ranking_loss(pos_scores, neg_scores, config.margin, valid_mask=valid_mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss detected at epoch {epoch}: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_total += float(loss.item())
            n_batches += 1

        entity_weights = model.encoder.entity_embeddings.weight.detach()
        if not torch.isfinite(entity_weights).all():
            raise RuntimeError(f"non-finite entity embedding detected at the end of epoch {epoch}")
        relation_weights = model.relation_embeddings.weight.detach()
        if not torch.isfinite(relation_weights).all():
            raise RuntimeError(f"non-finite relation embedding detected at the end of epoch {epoch}")

        val_metrics = None
        if config.validate_every is not None and (epoch + 1) % config.validate_every == 0:
            if val_disease_ids is None or val_indications is None or drug_ids is None:
                raise ValueError("validate_every is set but validation inputs were not all provided")

            active_score_fn = validation_score_fn
            if active_score_fn is None:
                if entity_index is None or indication_relation_idx is None:
                    raise ValueError(
                        "validate_every is set but no validation_score_fn was given and "
                        "entity_index/indication_relation_idx (needed for the default full-graph "
                        "inference fallback) were not both provided"
                    )
                from repurposemap.embeddings.rgcn_model import build_message_passing_edges

                edge_h_np, edge_r_np, edge_t_np = build_message_passing_edges(triples, n_relations)
                edge_h_val = torch.as_tensor(edge_h_np, dtype=torch.long, device=device)
                edge_r_val = torch.as_tensor(edge_r_np, dtype=torch.long, device=device)
                edge_t_val = torch.as_tensor(edge_t_np, dtype=torch.long, device=device)
                model.eval()
                with torch.no_grad():
                    val_entity_embeddings = model.encode(edge_h_val, edge_r_val, edge_t_val, n_entities)
                model.train()

                def active_score_fn(disease_id, candidate_drug_ids):
                    """Closes over `val_entity_embeddings`, computed exactly
                    once above -- the encoder runs once per validation
                    event, never once per disease."""
                    disease_idx = entity_index.id_to_index[disease_id]
                    drug_indices = torch.as_tensor(
                        [entity_index.id_to_index[d] for d in candidate_drug_ids], dtype=torch.long, device=device
                    )
                    rel = torch.full_like(drug_indices, indication_relation_idx)
                    tail = torch.full_like(drug_indices, disease_idx)
                    with torch.no_grad():
                        return model.score(val_entity_embeddings, drug_indices, rel, tail).cpu().numpy()

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

        subgraph_stats = {
            "mean_nodes_per_batch": float(np.mean(node_counts)) if node_counts else 0.0,
            "min_nodes_per_batch": int(np.min(node_counts)) if node_counts else 0,
            "max_nodes_per_batch": int(np.max(node_counts)) if node_counts else 0,
            "mean_edges_per_batch": float(np.mean(edge_counts)) if edge_counts else 0.0,
            "min_edges_per_batch": int(np.min(edge_counts)) if edge_counts else 0,
            "max_edges_per_batch": int(np.max(edge_counts)) if edge_counts else 0,
        }

        history.append(
            SampledEpochRecord(
                epoch=epoch,
                train_loss=epoch_loss_total / max(n_batches, 1),
                epoch_seconds=time.monotonic() - epoch_start,
                sampler_stats=sampler.stats.to_dict(),
                subgraph_stats=subgraph_stats,
                val_metrics=val_metrics,
            )
        )

        if checkpoint_path is not None and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint = build_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch, history=history, config=config,
                shuffle_generator=shuffle_generator, sampler=sampler,
                extra={"neighbor_sampler_rng_state": neighbor_sampler.get_state()},
            )
            save_checkpoint(checkpoint, checkpoint_path)

        if (
            config.early_stopping_patience is not None
            and val_metrics is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break

        if max_wall_time_s is not None and (time.monotonic() - call_start) >= max_wall_time_s:
            stopped_for_time_budget = True
            break

    return SampledGPUTrainingResult(
        model=model,
        optimizer=optimizer,
        history=history,
        final_epoch=history[-1].epoch if history else start_epoch - 1,
        stopped_early=stopped_early,
        shuffle_generator=shuffle_generator,
        sampler=sampler,
        neighbor_sampler=neighbor_sampler,
        stopped_for_time_budget=stopped_for_time_budget,
    )


# --------------------------------------------------------------------------
# Validation search and final training -- orchestration on top of
# train_sampled_gpu, mirroring rgcn_train_gpu.py's run_validation_search/
# train_final exactly (same SEARCH_CONFIGS, same selection rule, same
# eligibility-gate checkpoint fields), with one addition this trainer
# needs that the full-graph one doesn't: resumability ACROSS process
# invocations at search granularity, not just epoch granularity, via
# ``search_state_path``. Each epoch here is itself expensive enough
# (~4.7h on real PrimeKG -- see
# ml/experiments/rgcn_sampled_training_profile_e6ce5d9.json) that a single
# config's whole multi-epoch run routinely exceeds one Kaggle GPU
# session's hard time limit, so this function is designed to be called
# repeatedly (once per session): it picks up exactly where the last call
# left off (same config, next epoch) using ``max_wall_time_s`` to stop
# cleanly before a session is cut off mid-epoch, and reports whether the
# whole 3-config search is done yet via the returned log's ``"status"``.
#
# Neither function here ever reads or requires a locked-test manifest:
# they operate entirely on ``LoadedExport.tuning_triples``/
# ``final_triples`` and ``LoadedExport.val_indications``.
# --------------------------------------------------------------------------


def run_sampled_validation_search(
    loaded_export,
    *,
    search_configs: tuple[tuple[str, RGCNSampledTrainingConfig], ...] = SEARCH_CONFIGS,
    epochs: int = 3,
    validate_every: int = 1,
    output_log_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    search_state_path: str | Path | None = None,
    max_wall_time_s: float | None = None,
) -> dict:
    """Train each of the (exactly 3) predefined sampled-training configs on
    ``tuning_triples``, evaluate each on validation diseases only (via the
    default full-graph no-grad inference fallback in ``train_sampled_gpu``),
    and apply the existing selection rule
    (``repurposemap.evaluation.transe_baseline``, unchanged and reused as-is
    -- it is model-agnostic).

    Resumable at search granularity: if ``search_state_path`` already
    exists (from a prior call, likely in a prior Kaggle session), configs
    it marks complete are skipped, and the in-progress config resumes from
    its own checkpoint in ``checkpoint_dir``. Returns a log whose
    ``"status"`` is ``"in_progress"`` (call again to continue) or
    ``"complete"`` (every config finished and the winner is selected).

    Never touches locked-test data: ``loaded_export`` has none to touch.
    """
    from repurposemap.evaluation.transe_baseline import SearchCandidateResult, select_best_candidate

    tuning_triples = loaded_export.tuning_triples
    n_entities = len(loaded_export.entity_index)
    n_relations = len(loaded_export.relation_index)
    positive_store = build_positive_store(tuning_triples, n_relations, n_entities)
    adjacency = build_relation_adjacency(tuning_triples)

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state_path = Path(search_state_path) if search_state_path is not None else None
    state = json.loads(state_path.read_text()) if state_path is not None and state_path.exists() else {"completed": []}
    runs: list[dict] = list(state["completed"])
    completed_names = {r["name"] for r in runs}
    candidates: list[SearchCandidateResult] = [
        SearchCandidateResult(
            name=r["name"], embedding_dim=r["config"]["hidden_dim"],
            val_mrr=r["best_val_mrr"], val_macro_auprc=r["best_val_macro_auprc"],
        )
        for r in runs
    ]

    # max_wall_time_s is a budget for this whole call, not per config:
    # train_sampled_gpu measures its own budget from its own start, so
    # passing the full value to every config would let a session that
    # finishes one config early run the next config for another full budget.
    call_start = time.monotonic()
    trained_this_call = False

    for name, base_config in search_configs:
        if name in completed_names:
            continue

        remaining_s = None if max_wall_time_s is None else max_wall_time_s - (time.monotonic() - call_start)
        # The first config trained in a call always gets to finish one epoch
        # (train_sampled_gpu does), so a zero budget still makes progress.
        if trained_this_call and remaining_s is not None and remaining_s <= 0:
            log = {
                "status": "in_progress",
                "completed_configs": [r["name"] for r in runs],
                "current_config": name,
                "current_epoch": None,
                "current_config_total_epochs": epochs,
            }
            if output_log_path is not None:
                Path(output_log_path).write_text(json.dumps(log, indent=2, default=str))
            return log

        trained_this_call = True
        config = replace(base_config, epochs=epochs, validate_every=validate_every)
        checkpoint_path = Path(checkpoint_dir) / f"{name}.pt" if checkpoint_dir is not None else None
        resume_from = None
        if checkpoint_path is not None and checkpoint_path.exists():
            # Always load onto CPU, never config.device: the checkpoint's
            # RNG-state tensors (shuffle_generator_state, torch_cpu_rng_state)
            # must stay CPU ByteTensors for torch.Generator.set_state()/
            # torch.set_rng_state() regardless of training device -- using
            # map_location=config.device moved them to CUDA too, breaking
            # resume with "RNG state must be a torch.ByteTensor". Loading
            # the model/optimizer state dicts onto CPU is safe here: both
            # load_state_dict calls below copy their tensors into the
            # already-GPU-resident model/optimizer, which handles the
            # cross-device copy correctly.
            resume_from = load_checkpoint(checkpoint_path, map_location="cpu")

        started = time.monotonic()
        result = train_sampled_gpu(
            tuning_triples, n_entities, n_relations, loaded_export.type_pools, positive_store, adjacency, config,
            resume_from=resume_from, entity_index=loaded_export.entity_index,
            indication_relation_idx=loaded_export.indication_relation_idx,
            val_disease_ids=loaded_export.val_disease_ids, val_indications=loaded_export.val_indications,
            drug_ids=loaded_export.drug_ids, checkpoint_path=checkpoint_path, max_wall_time_s=remaining_s,
        )
        runtime_s = time.monotonic() - started

        config_finished = result.stopped_early or (result.final_epoch + 1 >= config.epochs)
        if not config_finished:
            # Time budget ran out mid-config this session -- not an error,
            # just not done yet. The checkpoint already written by
            # train_sampled_gpu is exactly what the next call resumes from.
            log = {
                "status": "in_progress",
                "completed_configs": [r["name"] for r in runs],
                "current_config": name,
                "current_epoch": result.final_epoch,
                "current_config_total_epochs": config.epochs,
            }
            if output_log_path is not None:
                Path(output_log_path).write_text(json.dumps(log, indent=2, default=str))
            return log

        val_records = [r for r in result.history if r.val_metrics is not None]
        best_record = max(val_records, key=lambda r: r.val_metrics["mrr"]) if val_records else None
        best_val_mrr = best_record.val_metrics["mrr"] if best_record else float("-inf")
        best_val_macro_auprc = best_record.val_metrics["macro_auprc"] if best_record else 0.0

        candidates.append(SearchCandidateResult(name=name, embedding_dim=config.hidden_dim, val_mrr=best_val_mrr, val_macro_auprc=best_val_macro_auprc))
        run_record = {
            "name": name,
            "config": config.to_dict(),
            "history": [r.to_dict() for r in result.history],
            "runtime_s": runtime_s,
            "stopped_early": result.stopped_early,
            "best_val_epoch": best_record.epoch if best_record else None,
            "best_val_mrr": best_val_mrr,
            "best_val_macro_auprc": best_val_macro_auprc,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        }
        runs.append(run_record)
        completed_names.add(name)

        if state_path is not None:
            state_path.write_text(json.dumps({"completed": runs}, indent=2, default=str))

    selected_index = select_best_candidate(candidates)

    from repurposemap.evaluation.transe_baseline import SELECTION_RULE_DESCRIPTION

    log = {
        "status": "complete",
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
        Path(output_log_path).write_text(json.dumps(log, indent=2, default=str))
    return log


def train_sampled_final(
    loaded_export,
    *,
    selected_config: RGCNSampledTrainingConfig,
    epochs: int,
    search_log: dict,
    checkpoint_path: str | Path,
    primekg_fingerprint: dict | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    device: str | None = None,
    resume_from: dict | None = None,
    max_wall_time_s: float | None = None,
) -> dict:
    """Train exactly once on ``final_triples`` (validation diseases'
    indications restored) with the already-selected config, and save a
    fully self-describing final checkpoint -- same shape and same
    eligibility-gate fields as TransE's/the full-graph R-GCN trainer's
    ``train_final`` (``is_final_training``, ``export_fingerprint``,
    ``locked_test_manifest_fingerprint``), so
    ``assert_eligible_for_locked_test_evaluation`` accepts this
    checkpoint unchanged.

    ``epochs`` must be decided from the winning search run's observed
    plateau *before* this is called -- there is no validation signal
    during final training (validation diseases are now training data), so
    no early stopping is used here, by design. Resumable across sessions
    the same way the search is, via ``resume_from``/``max_wall_time_s`` --
    call again with the checkpoint this wrote if ``"stopped_for_time_budget"``
    comes back true in the result this function returns under
    ``"stopped_for_time_budget"``.
    """
    device_override = {"device": device} if device is not None else {}
    final_config = replace(
        selected_config, epochs=epochs, validate_every=None, early_stopping_patience=None, **device_override,
    )
    final_triples = loaded_export.final_triples
    n_entities = len(loaded_export.entity_index)
    n_relations = len(loaded_export.relation_index)
    positive_store = build_positive_store(final_triples, n_relations, n_entities)
    adjacency = build_relation_adjacency(final_triples)

    result = train_sampled_gpu(
        final_triples, n_entities, n_relations, loaded_export.type_pools, positive_store, adjacency, final_config,
        resume_from=resume_from, max_wall_time_s=max_wall_time_s,
    )

    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=final_config, shuffle_generator=result.shuffle_generator, sampler=result.sampler,
        is_final_training=True,
        extra={
            "neighbor_sampler_rng_state": result.neighbor_sampler.get_state(),
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
    return {
        "final_epoch": result.final_epoch,
        "stopped_for_time_budget": result.stopped_for_time_budget,
        "train_loss_curve": [r.train_loss for r in result.history],
    }
