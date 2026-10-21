"""Model-agnostic checkpoint I/O and RNG-state capture/restore.

Extracted unchanged (behaviorally) from ``transe_train_gpu.py``, where this
logic originally lived next to TransE-specific training code it never
actually depended on -- ``build_checkpoint`` only calls ``model.state_dict()``
and ``optimizer.state_dict()``, and the RNG helpers only touch the shuffle
generator, the sampler, and global torch/CUDA RNG state. Shared by
``transe_train_gpu.py`` and ``rgcn_train_gpu.py`` so neither duplicates it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from repurposemap.embeddings.batched_negative_sampling import BatchedTypeAwareNegativeSampler


def _rng_state_dict(shuffle_generator: torch.Generator, sampler: BatchedTypeAwareNegativeSampler) -> dict:
    """Every RNG source a GPU trainer uses -- see the Stage D plan's §3/§7.

    Not claimed to give exact-resume reproducibility anywhere except where
    a resume test (e.g. ``tests/test_transe_train_gpu.py``) actually proves it.
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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list,
    config,
    shuffle_generator: torch.Generator,
    sampler: BatchedTypeAwareNegativeSampler,
    is_final_training: bool = False,
    extra: dict | None = None,
) -> dict:
    """The checkpoint dict saved to disk. ``extra`` adds final-training-only
    fields (export fingerprint, hyperparameter search summary, etc.) without
    this function needing to know about them.

    ``model`` is any ``torch.nn.Module`` (TransE, R-GCN, or anything future)
    -- this function only ever calls ``.state_dict()`` on it. ``history`` is
    any list of objects exposing ``.to_dict()`` (each model's own
    ``EpochRecord``). ``config`` is any object exposing ``.to_dict()``.
    """
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


def load_checkpoint(path: str | Path, *, map_location: str | torch.device | None = None) -> dict:
    """Load a checkpoint dict exactly as saved -- no mutation of its
    contents (``config["device"]`` still records whatever device it was
    trained on either way).

    ``map_location``, if given, is forwarded to ``torch.load`` unchanged
    (e.g. ``"cpu"`` to load a GPU-trained checkpoint's tensors on a
    machine with no CUDA build). Omitted, behavior is unchanged from
    before this parameter existed: ``torch.load`` deserializes tensors
    onto their original device, which requires that device to exist on
    the loading machine.
    """
    return torch.load(path, weights_only=False, map_location=map_location)
