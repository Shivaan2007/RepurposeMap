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

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from repurposemap.embeddings.batched_negative_sampling import BatchedTypeAwareNegativeSampler
from repurposemap.embeddings.checkpointing import (
    _restore_rng_state,
    build_checkpoint,
    save_checkpoint,
)
from repurposemap.embeddings.negative_sampling import TypePools
from repurposemap.embeddings.positive_store import PositiveTripleStore
from repurposemap.embeddings.rgcn_model import RGCN
from repurposemap.embeddings.rgcn_sampling import RelationAdjacency, RelationAwareNeighborSampler, SamplingConfig
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
    val_disease_ids=None,
    val_indications=None,
    drug_ids=None,
    checkpoint_path: str | Path | None = None,
) -> SampledGPUTrainingResult:
    """Train (or resume training) an R-GCN model with sampled-subgraph
    mini-batches. Ordinary SGD: each batch samples its own neighborhood,
    runs its own encoder forward, backward, and ``optimizer.step()`` --
    no shared encoder graph, no ``retain_graph=True``.

    ``triples`` is whichever leakage-safe array is "live" for the current
    phase -- ``tuning_triples`` during search, ``final_triples`` for final
    training -- same convention as the full-graph trainer. ``adjacency``
    must be built from that exact same array (see
    ``repurposemap.embeddings.rgcn_sampling.build_relation_adjacency``);
    this function never reads or requires a locked-test manifest, and has
    no parameter for one.

    Validation, if ``config.validate_every`` is set, uses a caller-supplied
    ``validation_score_fn`` only -- this function does not invent a
    sampled-inference evaluation strategy. Full-graph no-grad inference is
    the preferred validation strategy (cheaper than full-graph training,
    since no backward graph is retained) and, if it proves feasible on
    real data, should be what the caller's ``validation_score_fn`` wraps --
    but that feasibility check is a separate, later profiling step, not
    decided or implemented by this function.
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
            if validation_score_fn is None or val_disease_ids is None or val_indications is None or drug_ids is None:
                raise ValueError("validate_every is set but validation inputs were not all provided")
            from repurposemap.evaluation.transe_baseline import evaluate_transe_on_diseases

            val_result = evaluate_transe_on_diseases(validation_score_fn, val_disease_ids, val_indications, drug_ids)
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

    return SampledGPUTrainingResult(
        model=model,
        optimizer=optimizer,
        history=history,
        final_epoch=history[-1].epoch if history else start_epoch - 1,
        stopped_early=stopped_early,
        shuffle_generator=shuffle_generator,
        sampler=sampler,
        neighbor_sampler=neighbor_sampler,
    )
