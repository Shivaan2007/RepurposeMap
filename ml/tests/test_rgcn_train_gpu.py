"""The real R-GCN training loop: resume correctness, checkpointing,
validation, and the per-epoch (not per-batch) message-passing design this
module's docstring commits to.

Requires torch; skipped (not failed) in an environment without it installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from repurposemap.embeddings.negative_sampling import TypePools  # noqa: E402
from repurposemap.embeddings.positive_store import build_positive_store  # noqa: E402
from repurposemap.embeddings.rgcn_model import RGCN  # noqa: E402
from repurposemap.embeddings.rgcn_train_gpu import (  # noqa: E402
    RGCNGPUTrainingConfig,
    SEARCH_CONFIGS,
    train_gpu,
)
from repurposemap.embeddings.checkpointing import load_checkpoint  # noqa: E402


def _toy_triples_and_pools(n_triples: int = 6):
    # n_triples drugs, n_triples diseases, one indication edge each.
    type_of_index = tuple(["drug"] * n_triples + ["disease"] * n_triples)
    pools = {"drug": tuple(range(n_triples)), "disease": tuple(range(n_triples, 2 * n_triples))}
    tp = TypePools(pools_by_type=pools, type_of_index=type_of_index)
    triples = np.array([[i, 0, n_triples + i] for i in range(n_triples)], dtype=np.int64)
    store = build_positive_store(triples, n_relations=1, n_entities=2 * n_triples)
    return triples, tp, store


def test_training_loss_is_finite_and_tracked_per_epoch():
    triples, tp, store = _toy_triples_and_pools()
    config = RGCNGPUTrainingConfig(hidden_dim=8, num_bases=2, epochs=5, batch_size=6, negatives_per_positive=2, seed=1)
    result = train_gpu(triples, 12, 1, tp, store, config)

    assert len(result.history) == 5
    assert all(np.isfinite(r.train_loss) for r in result.history)
    assert all(torch.isfinite(p).all() for p in result.model.parameters())


def test_encoder_forward_runs_exactly_once_per_epoch_not_per_batch(monkeypatch):
    # 24 triples, batch_size=4 -> 6 mini-batches per epoch. If the encoder
    # were (incorrectly) recomputed per batch, call count would be
    # epochs * 6; this asserts it is exactly `epochs`.
    triples, tp, store = _toy_triples_and_pools(n_triples=24)

    import repurposemap.embeddings.rgcn_model as rgcn_model_module

    calls = []
    original_encode = rgcn_model_module.RGCN.encode

    def counting_encode(self, *args, **kwargs):
        calls.append(1)
        return original_encode(self, *args, **kwargs)

    monkeypatch.setattr(rgcn_model_module.RGCN, "encode", counting_encode)

    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=3, batch_size=4, negatives_per_positive=1, seed=1)
    train_gpu(triples, 48, 1, tp, store, config)

    assert len(calls) == 3


def test_optimizer_step_runs_exactly_once_per_epoch_not_per_batch(monkeypatch):
    triples, tp, store = _toy_triples_and_pools(n_triples=24)

    step_calls = []
    original_step = torch.optim.Adam.step

    def counting_step(self, *args, **kwargs):
        step_calls.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)

    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=3, batch_size=4, negatives_per_positive=1, seed=1)
    train_gpu(triples, 48, 1, tp, store, config)

    assert len(step_calls) == 3


def test_resume_exactly_reproduces_uninterrupted_training(tmp_path):
    triples, tp, store = _toy_triples_and_pools()

    full_config = RGCNGPUTrainingConfig(hidden_dim=8, num_bases=2, epochs=4, batch_size=6, negatives_per_positive=2, seed=42)
    full_result = train_gpu(triples, 12, 1, tp, store, full_config)

    checkpoint_path = tmp_path / "ckpt.pt"
    first_half_config = RGCNGPUTrainingConfig(hidden_dim=8, num_bases=2, epochs=2, batch_size=6, negatives_per_positive=2, seed=42)
    train_gpu(triples, 12, 1, tp, store, first_half_config, checkpoint_path=checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)

    second_half_config = RGCNGPUTrainingConfig(hidden_dim=8, num_bases=2, epochs=4, batch_size=6, negatives_per_positive=2, seed=42)
    resumed_result = train_gpu(triples, 12, 1, tp, store, second_half_config, resume_from=checkpoint, checkpoint_path=checkpoint_path)

    full_losses = [r.train_loss for r in full_result.history]
    resumed_losses = [r.train_loss for r in resumed_result.history]
    assert full_losses == pytest.approx(resumed_losses)

    for p_a, p_b in zip(full_result.model.parameters(), resumed_result.model.parameters(), strict=True):
        assert torch.equal(p_a, p_b)

    assert full_result.sampler.stats.to_dict() == resumed_result.sampler.stats.to_dict()


def test_checkpoint_contains_every_rng_source(tmp_path):
    triples, tp, store = _toy_triples_and_pools()
    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=1, batch_size=6, negatives_per_positive=1, seed=5)
    checkpoint_path = tmp_path / "ckpt.pt"
    train_gpu(triples, 12, 1, tp, store, config, checkpoint_path=checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    rng = checkpoint["rng_state"]
    assert set(rng) == {
        "shuffle_generator_state", "numpy_sampler_bit_generator_state",
        "scalar_fallback_random_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
    }


def test_explicit_device_cpu_is_forwarded_and_recorded():
    triples, tp, store = _toy_triples_and_pools()
    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=1, batch_size=6, negatives_per_positive=1, seed=1, device="cpu")
    result = train_gpu(triples, 12, 1, tp, store, config)

    assert result.history[-1].sampler_stats is not None
    for p in result.model.parameters():
        assert p.device.type == "cpu"


def test_non_finite_loss_raises_rather_than_training_through_it(monkeypatch):
    triples, tp, store = _toy_triples_and_pools()
    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=1, batch_size=6, negatives_per_positive=1, seed=1)

    import repurposemap.embeddings.rgcn_train_gpu as module

    def _nan_loss(*_args, **_kwargs):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(module, "margin_ranking_loss", _nan_loss)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        train_gpu(triples, 12, 1, tp, store, config)


def test_validation_tracking_runs_every_configured_epoch():
    triples, tp, store = _toy_triples_and_pools()
    drug_ids = tuple(f"drug{i}" for i in range(3))

    class _FakeIndication:
        def __init__(self, disease_id, drug_id):
            self.disease_id, self.drug_id, self.disease_name, self.drug_name = disease_id, drug_id, disease_id, drug_id

    val_indications = (_FakeIndication("diseaseA", "drug0"),)

    def score_fn(disease_id, candidate_drug_ids):
        del disease_id
        return np.array([1.0, 0.5, 0.1])[: len(candidate_drug_ids)]

    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=4, batch_size=6, negatives_per_positive=1, seed=1, validate_every=2)
    result = train_gpu(
        triples, 12, 1, tp, store, config,
        validation_score_fn=score_fn, val_disease_ids=("diseaseA",), val_indications=val_indications, drug_ids=drug_ids,
    )

    val_epochs = [r.epoch for r in result.history if r.val_metrics is not None]
    assert val_epochs == [1, 3]


def test_early_stopping_driven_only_by_validation_mrr():
    triples, tp, store = _toy_triples_and_pools()
    drug_ids = ("drug0", "drug1", "drug2")

    class _FakeIndication:
        def __init__(self, disease_id, drug_id):
            self.disease_id, self.drug_id, self.disease_name, self.drug_name = disease_id, drug_id, disease_id, drug_id

    val_indications = (_FakeIndication("diseaseA", "drug0"),)

    def flat_score_fn(disease_id, candidate_drug_ids):
        del disease_id
        return np.array([0.1, 0.5, 0.9])[: len(candidate_drug_ids)]

    config = RGCNGPUTrainingConfig(
        hidden_dim=4, num_bases=1, epochs=10, batch_size=6, negatives_per_positive=1, seed=1,
        validate_every=1, early_stopping_patience=2,
    )
    result = train_gpu(
        triples, 12, 1, tp, store, config,
        validation_score_fn=flat_score_fn, val_disease_ids=("diseaseA",), val_indications=val_indications, drug_ids=drug_ids,
    )

    assert result.stopped_early
    assert result.final_epoch < config.epochs - 1


def test_search_configs_are_exactly_three_and_match_the_approved_grid():
    assert len(SEARCH_CONFIGS) == 3
    names = [name for name, _config in SEARCH_CONFIGS]
    assert names == ["rgcn-base", "rgcn-wide", "rgcn-moreBases"]

    hidden_dims = {name: config.hidden_dim for name, config in SEARCH_CONFIGS}
    num_bases = {name: config.num_bases for name, config in SEARCH_CONFIGS}
    assert hidden_dims == {"rgcn-base": 64, "rgcn-wide": 128, "rgcn-moreBases": 64}
    assert num_bases == {"rgcn-base": 8, "rgcn-wide": 8, "rgcn-moreBases": 30}
    for _name, config in SEARCH_CONFIGS:
        assert config.margin == 1.0
        assert config.learning_rate == 1e-3
        assert config.negatives_per_positive == 4
        assert config.num_layers == 2


def test_negative_sampling_and_positive_store_never_see_inverse_relation_ids(monkeypatch):
    # The message-passing edges (relation IDs up to 2*n_relations) must
    # never reach the sampler or the positive store -- both are built with
    # n_relations=1 here, so any inverse-relation ID (>= 1) reaching them
    # would be a hard contract violation this test is designed to catch.
    triples, tp, store = _toy_triples_and_pools()
    assert store.n_relations == 1

    import repurposemap.embeddings.batched_negative_sampling as sampler_module

    original_corrupt_batch = sampler_module.BatchedTypeAwareNegativeSampler.corrupt_batch

    def checked_corrupt_batch(self, h, r, t):
        assert (r < 1).all(), "sampler received a relation ID outside the real (non-inverse) relation space"
        return original_corrupt_batch(self, h, r, t)

    monkeypatch.setattr(sampler_module.BatchedTypeAwareNegativeSampler, "corrupt_batch", checked_corrupt_batch)

    config = RGCNGPUTrainingConfig(hidden_dim=4, num_bases=1, epochs=2, batch_size=6, negatives_per_positive=1, seed=1)
    train_gpu(triples, 12, 1, tp, store, config)


def test_per_batch_gradient_accumulation_matches_single_mean_backward():
    # Isolates the exact refactor this file's train_gpu relies on, away
    # from the full R-GCN model: proves that N separate
    # (loss_i / N).backward(retain_graph=True) calls on losses sharing one
    # upstream tensor accumulate the identical gradient a single
    # torch.stack(losses).mean().backward() call would have produced --
    # the mathematical justification for switching away from "stack every
    # batch's loss, one big backward" (see rgcn_gpu_feasibility_profile).
    torch.manual_seed(0)
    shared = torch.randn(5, requires_grad=True)
    batch_weights = [torch.randn(5) for _ in range(4)]

    def make_losses(x):
        return [(x * w).sum() ** 2 for w in batch_weights]

    # Method A: stack all losses, one mean backward.
    x_a = shared.clone().detach().requires_grad_(True)
    losses_a = make_losses(x_a)
    torch.stack(losses_a).mean().backward()
    grad_a = x_a.grad.clone()

    # Method B: per-loss backward, each scaled by 1/N, retain_graph=True.
    x_b = shared.clone().detach().requires_grad_(True)
    losses_b = make_losses(x_b)
    n = len(losses_b)
    for loss in losses_b:
        (loss / n).backward(retain_graph=True)
    grad_b = x_b.grad.clone()

    assert torch.allclose(grad_a, grad_b, atol=1e-6)
