"""The sampled-subgraph R-GCN training loop: config validation,
determinism, resume correctness, the ordinary per-batch optimizer-step
shape (no retain_graph, no one-step-per-epoch constraint), and the same
real-relation-space isolation guarantees the full-graph trainer has.

Requires torch; skipped (not failed) in an environment without it installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from repurposemap.embeddings.entity_index import EntityIndex  # noqa: E402
from repurposemap.embeddings.negative_sampling import TypePools  # noqa: E402
from repurposemap.embeddings.positive_store import build_positive_store  # noqa: E402
from repurposemap.embeddings.rgcn_sampling import build_relation_adjacency  # noqa: E402
from repurposemap.embeddings.rgcn_sampled_train_gpu import (  # noqa: E402
    RGCNSampledTrainingConfig,
    train_sampled_gpu,
)
from repurposemap.embeddings.checkpointing import load_checkpoint  # noqa: E402


def _toy_triples_and_fixtures(n_triples: int = 30):
    type_of_index = tuple(["drug"] * n_triples + ["disease"] * n_triples)
    pools = {"drug": tuple(range(n_triples)), "disease": tuple(range(n_triples, 2 * n_triples))}
    tp = TypePools(pools_by_type=pools, type_of_index=type_of_index)
    triples = np.array([[i, 0, n_triples + i] for i in range(n_triples)], dtype=np.int64)
    store = build_positive_store(triples, n_relations=1, n_entities=2 * n_triples)
    adjacency = build_relation_adjacency(triples)
    return triples, tp, store, adjacency


def _toy_entity_index(n_triples: int = 30) -> EntityIndex:
    index_to_id = tuple(f"drug{i}" for i in range(n_triples)) + tuple(f"disease{i}" for i in range(n_triples))
    return EntityIndex.from_dict({"index_to_id": list(index_to_id)})


class _FakeIndication:
    def __init__(self, disease_id, drug_id):
        self.disease_id, self.drug_id, self.disease_name, self.drug_name = disease_id, drug_id, disease_id, drug_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0}, {"num_bases": 0}, {"num_layers": 0}, {"margin": 0.0},
        {"learning_rate": 0.0}, {"batch_size": 0}, {"negatives_per_positive": 0}, {"epochs": 0},
        {"fanout_hop1": 0}, {"fanout_hop2": 0}, {"checkpoint_every": 0},
        {"max_vectorized_retries": 0}, {"max_scalar_retries": 0},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RGCNSampledTrainingConfig(**kwargs)


def test_config_to_dict_has_no_hidden_fields():
    config = RGCNSampledTrainingConfig()
    assert set(config.to_dict()) == {
        "hidden_dim", "num_bases", "num_layers", "margin", "learning_rate", "batch_size",
        "negatives_per_positive", "epochs", "seed", "device", "fanout_hop1", "fanout_hop2",
        "checkpoint_every", "max_vectorized_retries", "max_scalar_retries",
        "validate_every", "early_stopping_patience",
    }


def test_training_loss_is_finite_and_tracked_per_epoch():
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    config = RGCNSampledTrainingConfig(hidden_dim=8, num_bases=2, fanout_hop1=5, fanout_hop2=5, epochs=3, batch_size=10, negatives_per_positive=2, seed=1)
    result = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config)

    assert len(result.history) == 3
    assert all(np.isfinite(r.train_loss) for r in result.history)
    assert all(torch.isfinite(p).all() for p in result.model.parameters())


def test_optimizer_step_runs_once_per_batch_not_once_per_epoch(monkeypatch):
    # The whole point of the sampled trainer: ordinary SGD, no
    # retain_graph, no one-step-per-epoch constraint. 30 triples,
    # batch_size=10 -> 3 batches per epoch; 2 epochs -> 6 total steps.
    triples, tp, store, adjacency = _toy_triples_and_fixtures()

    step_calls = []
    original_step = torch.optim.Adam.step

    def counting_step(self, *args, **kwargs):
        step_calls.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)

    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=2, batch_size=10, negatives_per_positive=1, seed=1)
    train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config)

    assert len(step_calls) == 6


def test_sampling_never_reaches_for_the_real_locked_test_manifest():
    # Structural check: train_sampled_gpu's own signature has no
    # parameter that could carry a locked-test manifest, and its own
    # source never references one -- scoped to this one function (not the
    # whole module) because train_sampled_final legitimately stores a
    # locked_test_manifest_fingerprint (a hash, not the manifest's
    # content) in final-checkpoint metadata, exactly mirroring the
    # already-approved rgcn_train_gpu.train_final pattern, for the
    # existing eligibility-gate check to consume later.
    import inspect

    import repurposemap.embeddings.rgcn_sampled_train_gpu as module

    sig = inspect.signature(module.train_sampled_gpu)
    assert "manifest" not in sig.parameters
    assert "locked_test_manifest" not in sig.parameters
    source = inspect.getsource(module.train_sampled_gpu)
    assert "locked_test_manifest" not in source
    assert "load_locked_test_manifest" not in source


def test_resume_exactly_reproduces_uninterrupted_training(tmp_path):
    triples, tp, store, adjacency = _toy_triples_and_fixtures()

    full_config = RGCNSampledTrainingConfig(hidden_dim=8, num_bases=2, fanout_hop1=5, fanout_hop2=5, epochs=4, batch_size=10, negatives_per_positive=2, seed=42)
    full_result = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, full_config)

    checkpoint_path = tmp_path / "ckpt.pt"
    first_half_config = RGCNSampledTrainingConfig(hidden_dim=8, num_bases=2, fanout_hop1=5, fanout_hop2=5, epochs=2, batch_size=10, negatives_per_positive=2, seed=42)
    train_sampled_gpu(triples, 60, 1, tp, store, adjacency, first_half_config, checkpoint_path=checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)

    second_half_config = RGCNSampledTrainingConfig(hidden_dim=8, num_bases=2, fanout_hop1=5, fanout_hop2=5, epochs=4, batch_size=10, negatives_per_positive=2, seed=42)
    resumed_result = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, second_half_config, resume_from=checkpoint, checkpoint_path=checkpoint_path)

    full_losses = [r.train_loss for r in full_result.history]
    resumed_losses = [r.train_loss for r in resumed_result.history]
    assert full_losses == pytest.approx(resumed_losses)

    for p_a, p_b in zip(full_result.model.parameters(), resumed_result.model.parameters(), strict=True):
        assert torch.equal(p_a, p_b)

    assert full_result.sampler.stats.to_dict() == resumed_result.sampler.stats.to_dict()


def test_checkpoint_contains_neighbor_sampler_rng_state(tmp_path):
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=1, batch_size=10, negatives_per_positive=1, seed=5)
    checkpoint_path = tmp_path / "ckpt.pt"
    train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config, checkpoint_path=checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    assert "neighbor_sampler_rng_state" in checkpoint
    assert "bit_generator_state" in checkpoint["neighbor_sampler_rng_state"]
    rng = checkpoint["rng_state"]
    assert set(rng) == {
        "shuffle_generator_state", "numpy_sampler_bit_generator_state",
        "scalar_fallback_random_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
    }


def test_non_finite_loss_raises_rather_than_training_through_it(monkeypatch):
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=1, batch_size=10, negatives_per_positive=1, seed=1)

    import repurposemap.embeddings.rgcn_sampled_train_gpu as module

    def _nan_loss(*_args, **_kwargs):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(module, "margin_ranking_loss", _nan_loss)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config)


def test_validation_tracking_runs_every_configured_epoch():
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    drug_ids = tuple(f"drug{i}" for i in range(3))

    class _FakeIndication:
        def __init__(self, disease_id, drug_id):
            self.disease_id, self.drug_id, self.disease_name, self.drug_name = disease_id, drug_id, disease_id, drug_id

    val_indications = (_FakeIndication("diseaseA", "drug0"),)

    def score_fn(disease_id, candidate_drug_ids):
        del disease_id
        return np.array([1.0, 0.5, 0.1])[: len(candidate_drug_ids)]

    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=4, batch_size=10, negatives_per_positive=1, seed=1, validate_every=2)
    result = train_sampled_gpu(
        triples, 60, 1, tp, store, adjacency, config,
        validation_score_fn=score_fn, val_disease_ids=("diseaseA",), val_indications=val_indications, drug_ids=drug_ids,
    )

    val_epochs = [r.epoch for r in result.history if r.val_metrics is not None]
    assert val_epochs == [1, 3]


def test_early_stopping_driven_only_by_validation_mrr():
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    drug_ids = ("drug0", "drug1", "drug2")

    class _FakeIndication:
        def __init__(self, disease_id, drug_id):
            self.disease_id, self.drug_id, self.disease_name, self.drug_name = disease_id, drug_id, disease_id, drug_id

    val_indications = (_FakeIndication("diseaseA", "drug0"),)

    def flat_score_fn(disease_id, candidate_drug_ids):
        del disease_id
        return np.array([0.1, 0.5, 0.9])[: len(candidate_drug_ids)]

    config = RGCNSampledTrainingConfig(
        hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=10, batch_size=10,
        negatives_per_positive=1, seed=1, validate_every=1, early_stopping_patience=2,
    )
    result = train_sampled_gpu(
        triples, 60, 1, tp, store, adjacency, config,
        validation_score_fn=flat_score_fn, val_disease_ids=("diseaseA",), val_indications=val_indications, drug_ids=drug_ids,
    )

    assert result.stopped_early
    assert result.final_epoch < config.epochs - 1


def test_negative_sampling_and_positive_store_never_see_inverse_relation_ids(monkeypatch):
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    assert store.n_relations == 1

    import repurposemap.embeddings.batched_negative_sampling as sampler_module

    original_corrupt_batch = sampler_module.BatchedTypeAwareNegativeSampler.corrupt_batch

    def checked_corrupt_batch(self, h, r, t):
        assert (r < 1).all(), "sampler received a relation ID outside the real (non-inverse) relation space"
        return original_corrupt_batch(self, h, r, t)

    monkeypatch.setattr(sampler_module.BatchedTypeAwareNegativeSampler, "corrupt_batch", checked_corrupt_batch)

    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=2, batch_size=10, negatives_per_positive=1, seed=1)
    train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config)


def test_sampled_subgraphs_vary_in_size_across_batches_reported_in_stats():
    triples, tp, store, adjacency = _toy_triples_and_fixtures(n_triples=30)
    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=3, fanout_hop2=3, epochs=1, batch_size=10, negatives_per_positive=1, seed=1)
    result = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config)

    stats = result.history[0].subgraph_stats
    assert stats["mean_nodes_per_batch"] > 0
    assert stats["mean_edges_per_batch"] >= 0
    assert stats["max_nodes_per_batch"] >= stats["min_nodes_per_batch"]


def test_default_validation_falls_back_to_full_graph_inference_when_no_score_fn_given():
    # No explicit validation_score_fn -- entity_index/indication_relation_idx
    # alone must be enough to produce finite validation metrics via the
    # built-in full-graph no-grad inference fallback.
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    entity_index = _toy_entity_index()
    drug_ids = tuple(f"drug{i}" for i in range(3))
    val_indications = (_FakeIndication("disease0", "drug0"),)

    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=2, batch_size=10, negatives_per_positive=1, seed=1, validate_every=1)
    result = train_sampled_gpu(
        triples, 60, 1, tp, store, adjacency, config,
        entity_index=entity_index, indication_relation_idx=0,
        val_disease_ids=("disease0",), val_indications=val_indications, drug_ids=drug_ids,
    )

    assert len(result.history) == 2
    assert all(r.val_metrics is not None for r in result.history)
    assert all(np.isfinite(r.val_metrics["mrr"]) for r in result.history)


def test_default_validation_calls_the_encoder_exactly_once_per_validation_event_not_per_disease(monkeypatch):
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    entity_index = _toy_entity_index()
    drug_ids = tuple(f"drug{i}" for i in range(3))
    val_indications = (_FakeIndication("disease0", "drug0"), _FakeIndication("disease1", "drug1"))

    import repurposemap.embeddings.rgcn_model as rgcn_model_module

    call_count = {"n": 0}
    original_encode = rgcn_model_module.RGCN.encode

    def counting_encode(self, *args, **kwargs):
        call_count["n"] += 1
        return original_encode(self, *args, **kwargs)

    monkeypatch.setattr(rgcn_model_module.RGCN, "encode", counting_encode)

    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=1, batch_size=10, negatives_per_positive=1, seed=1, validate_every=1)
    train_sampled_gpu(
        triples, 60, 1, tp, store, adjacency, config,
        entity_index=entity_index, indication_relation_idx=0,
        val_disease_ids=("disease0", "disease1"), val_indications=val_indications, drug_ids=drug_ids,
    )

    assert call_count["n"] == 1  # one encoder forward for the whole validation pass, not one per disease


def test_default_validation_raises_without_entity_index_or_score_fn():
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=1, batch_size=10, negatives_per_positive=1, seed=1, validate_every=1)

    with pytest.raises(ValueError, match="entity_index/indication_relation_idx"):
        train_sampled_gpu(
            triples, 60, 1, tp, store, adjacency, config,
            val_disease_ids=("disease0",), val_indications=(), drug_ids=("drug0",),
        )


def test_max_wall_time_s_stops_training_and_can_be_resumed(tmp_path):
    triples, tp, store, adjacency = _toy_triples_and_fixtures()
    config = RGCNSampledTrainingConfig(hidden_dim=4, num_bases=1, fanout_hop1=5, fanout_hop2=5, epochs=4, batch_size=10, negatives_per_positive=1, seed=1)
    checkpoint_path = tmp_path / "ckpt.pt"

    first = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config, max_wall_time_s=0, checkpoint_path=checkpoint_path)
    assert first.stopped_for_time_budget is True
    assert first.final_epoch == 0  # exactly one epoch completed before the budget check tripped
    assert len(first.history) == 1

    resume_from = load_checkpoint(checkpoint_path)
    second = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config, resume_from=resume_from)
    assert second.stopped_for_time_budget is False
    assert second.final_epoch == config.epochs - 1
    assert len(second.history) == config.epochs

    straight_through = train_sampled_gpu(triples, 60, 1, tp, store, adjacency, config)
    assert [r.train_loss for r in second.history] == [r.train_loss for r in straight_through.history]
