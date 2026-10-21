"""The real TransE training loop: resume correctness, checkpointing, validation (Stage D).

Requires torch; skipped (not failed) in an environment without it installed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from repurposemap.embeddings.negative_sampling import TypePools  # noqa: E402
from repurposemap.embeddings.positive_store import build_positive_store  # noqa: E402
from repurposemap.embeddings.transe_model import TransE  # noqa: E402
from repurposemap.embeddings.transe_train_gpu import (  # noqa: E402
    GPUTrainingConfig,
    SEARCH_CONFIGS,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_gpu,
)


def _toy_triples_and_pools():
    type_of_index = tuple(["drug"] * 6 + ["disease"] * 6)
    pools = {"drug": tuple(range(6)), "disease": tuple(range(6, 12))}
    tp = TypePools(pools_by_type=pools, type_of_index=type_of_index)
    triples = np.array([[i, 0, 6 + i] for i in range(6)], dtype=np.int64)
    store = build_positive_store(triples, n_relations=1, n_entities=12)
    return triples, tp, store


def test_training_loss_is_finite_and_tracked_per_epoch():
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=8, epochs=5, batch_size=6, negatives_per_positive=2, seed=1)
    result = train_gpu(triples, 12, 1, tp, store, config)

    assert len(result.history) == 5
    assert all(np.isfinite(r.train_loss) for r in result.history)
    assert all(torch.isfinite(p).all() for p in result.model.parameters())


def test_resume_exactly_reproduces_uninterrupted_training(tmp_path):
    triples, tp, store = _toy_triples_and_pools()

    full_config = GPUTrainingConfig(embedding_dim=8, epochs=4, batch_size=6, negatives_per_positive=2, seed=42)
    full_result = train_gpu(triples, 12, 1, tp, store, full_config)

    checkpoint_path = tmp_path / "ckpt.pt"
    first_half_config = GPUTrainingConfig(embedding_dim=8, epochs=2, batch_size=6, negatives_per_positive=2, seed=42)
    train_gpu(triples, 12, 1, tp, store, first_half_config, checkpoint_path=checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)

    second_half_config = GPUTrainingConfig(embedding_dim=8, epochs=4, batch_size=6, negatives_per_positive=2, seed=42)
    resumed_result = train_gpu(triples, 12, 1, tp, store, second_half_config, resume_from=checkpoint, checkpoint_path=checkpoint_path)

    # Identical full loss history.
    full_losses = [r.train_loss for r in full_result.history]
    resumed_losses = [r.train_loss for r in resumed_result.history]
    assert full_losses == pytest.approx(resumed_losses)

    # Identical final model weights.
    for p_a, p_b in zip(full_result.model.parameters(), resumed_result.model.parameters(), strict=True):
        assert torch.equal(p_a, p_b)

    # Identical sampler counters at the end.
    assert full_result.sampler.stats.to_dict() == resumed_result.sampler.stats.to_dict()


def test_checkpoint_contains_every_rng_source():
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=4, epochs=1, batch_size=6, negatives_per_positive=1, seed=5)
    result = train_gpu(triples, 12, 1, tp, store, config)

    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=config, shuffle_generator=result.shuffle_generator, sampler=result.sampler,
    )
    rng = checkpoint["rng_state"]
    assert set(rng) == {
        "shuffle_generator_state", "numpy_sampler_bit_generator_state",
        "scalar_fallback_random_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
    }


def test_checkpoint_save_and_load_round_trips(tmp_path):
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=4, epochs=1, batch_size=6, negatives_per_positive=1, seed=5)
    result = train_gpu(triples, 12, 1, tp, store, config)

    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=config, shuffle_generator=result.shuffle_generator, sampler=result.sampler,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)
    reloaded = load_checkpoint(path)

    assert reloaded["epoch"] == checkpoint["epoch"]
    assert reloaded["config"] == checkpoint["config"]
    assert reloaded["is_final_training"] is False


def test_load_checkpoint_accepts_explicit_map_location(tmp_path):
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=4, epochs=1, batch_size=6, negatives_per_positive=1, seed=5)
    result = train_gpu(triples, 12, 1, tp, store, config)
    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=config, shuffle_generator=result.shuffle_generator, sampler=result.sampler,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)

    reloaded = load_checkpoint(path, map_location="cpu")

    assert reloaded["epoch"] == checkpoint["epoch"]
    assert reloaded["config"] == checkpoint["config"]
    for tensor in reloaded["state_dict"].values():
        assert tensor.device.type == "cpu"


def test_load_checkpoint_default_map_location_unchanged(tmp_path):
    # Omitting map_location must behave exactly as before this parameter
    # existed -- no behavior change for existing callers.
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=4, epochs=1, batch_size=6, negatives_per_positive=1, seed=5)
    result = train_gpu(triples, 12, 1, tp, store, config)
    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=config, shuffle_generator=result.shuffle_generator, sampler=result.sampler,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)

    reloaded_default = load_checkpoint(path)
    reloaded_explicit_none = load_checkpoint(path, map_location=None)

    assert reloaded_default["config"] == reloaded_explicit_none["config"] == checkpoint["config"]


def test_cpu_map_location_loads_a_checkpoint_whose_config_claims_cuda(tmp_path):
    # Simulates the real failure: a checkpoint trained on CUDA (so its
    # config.device says "cuda") being loaded on a CPU-only machine.
    # map_location="cpu" must load it without raising, the config's
    # recorded training device must be left exactly as-is (never read to
    # pick an evaluation device), and the reloaded tensors must actually
    # be on CPU.
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=4, epochs=1, batch_size=6, negatives_per_positive=1, seed=5, device="cuda")
    result = train_gpu(triples, 12, 1, tp, store, replace(config, device="cpu"))  # this test machine has no CUDA build
    checkpoint = build_checkpoint(
        model=result.model, optimizer=result.optimizer, epoch=result.final_epoch, history=result.history,
        config=config,  # the RECORDED config still says "cuda" -- only the actual run used "cpu"
        shuffle_generator=result.shuffle_generator, sampler=result.sampler, is_final_training=True,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)

    reloaded = load_checkpoint(path, map_location="cpu")

    assert reloaded["config"]["device"] == "cuda"  # unmutated
    assert reloaded["is_final_training"] is True  # unmutated
    for tensor in reloaded["state_dict"].values():
        assert tensor.device.type == "cpu"

    model = TransE(12, 1, embedding_dim=reloaded["config"]["embedding_dim"], distance=reloaded["config"]["distance"])
    model.load_state_dict(reloaded["state_dict"])
    for p in model.parameters():
        assert p.device.type == "cpu"


def test_non_finite_loss_raises_rather_than_training_through_it(monkeypatch):
    triples, tp, store = _toy_triples_and_pools()
    config = GPUTrainingConfig(embedding_dim=4, epochs=1, batch_size=6, negatives_per_positive=1, seed=1)

    import repurposemap.embeddings.transe_train_gpu as module

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
            self.disease_id = disease_id
            self.drug_id = drug_id
            self.disease_name = disease_id
            self.drug_name = drug_id

    val_indications = (_FakeIndication("diseaseA", "drug0"),)

    def score_fn(disease_id, candidate_drug_ids):
        del disease_id
        return np.array([1.0, 0.5, 0.1])[: len(candidate_drug_ids)]

    config = GPUTrainingConfig(
        embedding_dim=4, epochs=4, batch_size=6, negatives_per_positive=1, seed=1, validate_every=2
    )
    result = train_gpu(
        triples, 12, 1, tp, store, config,
        validation_score_fn=score_fn, val_disease_ids=("diseaseA",), val_indications=val_indications, drug_ids=drug_ids,
    )

    val_epochs = [r.epoch for r in result.history if r.val_metrics is not None]
    assert val_epochs == [1, 3]  # 0-indexed epochs 1 and 3 are the 2nd and 4th epoch


def test_early_stopping_driven_only_by_validation_mrr():
    triples, tp, store = _toy_triples_and_pools()
    drug_ids = ("drug0", "drug1", "drug2")

    class _FakeIndication:
        def __init__(self, disease_id, drug_id):
            self.disease_id, self.drug_id, self.disease_name, self.drug_name = disease_id, drug_id, disease_id, drug_id

    val_indications = (_FakeIndication("diseaseA", "drug0"),)

    def flat_score_fn(disease_id, candidate_drug_ids):
        # Always the same (bad) ranking -- MRR never improves, so patience should trigger.
        del disease_id
        return np.array([0.1, 0.5, 0.9])[: len(candidate_drug_ids)]

    config = GPUTrainingConfig(
        embedding_dim=4, epochs=10, batch_size=6, negatives_per_positive=1, seed=1,
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
    assert names == ["search-dim64", "search-dim128", "search-l1"]

    dims = {name: config.embedding_dim for name, config in SEARCH_CONFIGS}
    distances = {name: config.distance for name, config in SEARCH_CONFIGS}
    assert dims == {"search-dim64": 64, "search-dim128": 128, "search-l1": 64}
    assert distances == {"search-dim64": "l2", "search-dim128": "l2", "search-l1": "l1"}
    for _name, config in SEARCH_CONFIGS:
        assert config.margin == 1.0
        assert config.learning_rate == 1e-3
        assert config.negatives_per_positive == 4
