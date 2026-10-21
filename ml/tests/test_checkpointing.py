"""Checkpoint/RNG-state I/O is model-agnostic: proven here against a plain
``nn.Linear`` stand-in, not ``TransE`` or ``RGCN`` -- this is the extraction
``transe_train_gpu.py``/``rgcn_train_gpu.py`` both now import from."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from repurposemap.embeddings.batched_negative_sampling import BatchedTypeAwareNegativeSampler  # noqa: E402
from repurposemap.embeddings.checkpointing import (  # noqa: E402
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from repurposemap.embeddings.negative_sampling import TypePools  # noqa: E402
from repurposemap.embeddings.positive_store import build_positive_store  # noqa: E402


def _tiny_sampler(seed: int = 0) -> BatchedTypeAwareNegativeSampler:
    type_pools = TypePools(pools_by_type={"thing": (0, 1, 2, 3)}, type_of_index=("thing",) * 4)
    store = build_positive_store([(0, 0, 1), (2, 0, 3)], n_relations=1, n_entities=4)
    return BatchedTypeAwareNegativeSampler(type_pools, n_entities=4, n_relations=1, positive_store=store, seed=seed)


@dataclass(frozen=True)
class _FakeConfig:
    """Stands in for ``GPUTrainingConfig``/``RGCNGPUTrainingConfig`` --
    ``build_checkpoint`` only ever calls ``.to_dict()`` on it."""

    knob: int = 1

    def to_dict(self) -> dict:
        return {"knob": self.knob}


@dataclass
class _FakeEpochRecord:
    """Stands in for ``EpochRecord`` -- ``build_checkpoint`` only ever calls
    ``.to_dict()`` on each history entry."""

    epoch: int
    loss: float

    def to_dict(self) -> dict:
        return {"epoch": self.epoch, "loss": self.loss}


def test_build_checkpoint_accepts_a_plain_nn_module_not_just_transe():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    sampler = _tiny_sampler()
    shuffle_generator = torch.Generator().manual_seed(0)

    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, epoch=2, history=[_FakeEpochRecord(epoch=0, loss=1.0)],
        config=_FakeConfig(knob=7), shuffle_generator=shuffle_generator, sampler=sampler,
    )

    assert checkpoint["config"] == {"knob": 7}
    assert checkpoint["history"] == [{"epoch": 0, "loss": 1.0}]
    assert checkpoint["is_final_training"] is False
    assert set(checkpoint["state_dict"]) == set(model.state_dict())
    assert "rng_state" in checkpoint


def test_build_checkpoint_extra_fields_merge_in():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    sampler = _tiny_sampler()
    shuffle_generator = torch.Generator().manual_seed(0)

    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, epoch=0, history=[], config=_FakeConfig(),
        shuffle_generator=shuffle_generator, sampler=sampler, is_final_training=True,
        extra={"export_fingerprint": "abc123"},
    )

    assert checkpoint["is_final_training"] is True
    assert checkpoint["export_fingerprint"] == "abc123"


def test_save_and_load_checkpoint_round_trips(tmp_path):
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())
    sampler = _tiny_sampler()
    shuffle_generator = torch.Generator().manual_seed(0)

    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, epoch=5, history=[], config=_FakeConfig(knob=9),
        shuffle_generator=shuffle_generator, sampler=sampler,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)
    reloaded = load_checkpoint(path)

    assert reloaded["epoch"] == 5
    assert reloaded["config"] == {"knob": 9}
    for key, tensor in reloaded["state_dict"].items():
        assert torch.equal(tensor, checkpoint["state_dict"][key])


def test_load_checkpoint_explicit_map_location_cpu(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    sampler = _tiny_sampler()
    shuffle_generator = torch.Generator().manual_seed(0)
    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, epoch=0, history=[], config=_FakeConfig(),
        shuffle_generator=shuffle_generator, sampler=sampler,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, path)

    reloaded = load_checkpoint(path, map_location="cpu")

    for tensor in reloaded["state_dict"].values():
        assert tensor.device.type == "cpu"


def test_rng_state_round_trip_restores_both_sources():
    from repurposemap.embeddings.checkpointing import _restore_rng_state, _rng_state_dict

    sampler = _tiny_sampler(seed=3)
    shuffle_generator = torch.Generator().manual_seed(3)
    # Advance both RNG sources so there's real state to capture.
    torch.randperm(10, generator=shuffle_generator)
    sampler.corrupt_batch(
        __import__("numpy").array([0, 2]), __import__("numpy").array([0, 0]), __import__("numpy").array([1, 3])
    )

    state = _rng_state_dict(shuffle_generator, sampler)

    fresh_generator = torch.Generator().manual_seed(0)
    fresh_sampler = _tiny_sampler(seed=0)
    _restore_rng_state(state, fresh_generator, fresh_sampler)

    assert torch.equal(fresh_generator.get_state(), shuffle_generator.get_state())
    assert fresh_sampler.get_state() == sampler.get_state()
