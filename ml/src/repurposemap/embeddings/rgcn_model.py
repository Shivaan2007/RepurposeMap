"""The R-GCN encoder (Schlichtkrull et al., 2017) + DistMult decoder.

Deliberately mirrors ``transe_model.py``'s simplicity where the two models
agree (plain ``torch.nn.Module``, no external KGE library beyond PyTorch),
and diverges only where R-GCN genuinely must: it needs the *whole* graph's
structural edges to produce entity embeddings (message passing), not just
the batch of triples being scored, so ``RGCN.score`` takes already-computed
entity embeddings rather than recomputing them itself -- the caller (the
training/evaluation loop) controls exactly when the expensive full-graph
forward pass runs.

Message passing uses explicit inverse relations (standard R-GCN practice,
Schlichtkrull et al. 2017): for every real relation ``r`` in
``[0, n_relations)``, an inverse relation ``r + n_relations`` carries
information the other way along the same edges. These synthetic inverse
relation IDs exist *only* inside this module's message-passing graph
construction (``build_message_passing_edges``) and the encoder's own
per-relation weight indexing -- they are never passed to the DistMult
decoder (which has a relation table sized to the real ``n_relations`` only),
never passed to negative sampling, and never passed to the positive-triple
store. See ``build_message_passing_edges``'s docstring for the isolation
contract this module and its callers must preserve.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def build_message_passing_edges(triples: np.ndarray, n_relations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, r, t)`` triples -> doubled ``(edge_h, edge_r, edge_t)`` message-passing edges.

    For every input triple ``(h, r, t)``, produces two message-passing
    edges: the forward edge ``(h, r, t)`` and an inverse edge
    ``(t, r + n_relations, h)``. The returned relation IDs lie in
    ``[0, 2 * n_relations)``.

    **Isolation contract (safety-relevant, not just a style preference):**
    this function's output is for message-passing structure only. It must
    never be fed into ``build_positive_store``, ``TypeAwareNegativeSampler``/
    ``BatchedTypeAwareNegativeSampler``, evaluation, or anything touching
    the locked-test manifest -- all of those continue operating on the
    original, untouched ``triples`` array (relation IDs ``< n_relations``).
    Mixing the two would feed synthetic relation IDs >= ``n_relations`` into
    machinery that was built, and tested, only for the real relation space.

    Does not mutate ``triples``. Only ever called with ``tuning_triples`` or
    ``final_triples`` (already leakage-safe), so doubling them for message
    passing introduces no new leakage surface.
    """
    h = np.asarray(triples[:, 0], dtype=np.int64)
    r = np.asarray(triples[:, 1], dtype=np.int64)
    t = np.asarray(triples[:, 2], dtype=np.int64)

    edge_h = np.concatenate([h, t])
    edge_r = np.concatenate([r, r + n_relations])
    edge_t = np.concatenate([t, h])
    return edge_h, edge_r, edge_t


class _BasisRelationLayer(nn.Module):
    """One R-GCN layer: basis-decomposed per-relation weights + a separate
    self-loop transform + relation-wise mean-normalized aggregation.

    ``n_relations_mp`` is the *message-passing* relation count (already
    doubled for inverses by the caller) -- this layer has no concept of
    "real" vs. "inverse" relations; it just has that many weight slots.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations_mp: int, num_bases: int) -> None:
        super().__init__()
        if num_bases < 1:
            raise ValueError(f"num_bases must be at least 1, got {num_bases}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations_mp = n_relations_mp
        self.num_bases = num_bases

        self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.empty(n_relations_mp, num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim)

        bound = 1.0 / (in_dim**0.5)
        nn.init.uniform_(self.bases, -bound, bound)
        nn.init.uniform_(self.coefficients, -bound, bound)

    def relation_weight(self, relation_id: int) -> torch.Tensor:
        """The single ``(in_dim, out_dim)`` weight matrix for one relation,
        reconstructed from the shared bases -- exposed mainly so tests can
        check the basis-decomposition math directly."""
        return torch.einsum("b,bio->io", self.coefficients[relation_id], self.bases)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_h: torch.Tensor,
        edge_r: torch.Tensor,
        edge_t: torch.Tensor,
        n_entities: int,
    ) -> torch.Tensor:
        device = node_features.device
        self_term = self.self_loop(node_features)
        messages_sum = torch.zeros(n_entities, self.out_dim, device=device, dtype=node_features.dtype)

        # Per-relation weight matrices for every message-passing relation at
        # once -- cheap at this project's scale (n_relations_mp * in_dim *
        # out_dim is a few MB even at hidden_dim=128), not a per-edge cost.
        all_weights = torch.einsum("rb,bio->rio", self.coefficients, self.bases)

        for relation_id in range(self.n_relations_mp):
            mask = edge_r == relation_id
            if not torch.any(mask):
                continue
            h_idx = edge_h[mask]
            t_idx = edge_t[mask]

            transformed = node_features[h_idx] @ all_weights[relation_id]

            # Relation-wise mean normalization: each edge's contribution is
            # divided by how many edges of *this relation* point into its
            # target -- the standard R-GCN normalization, computed by
            # scattering a ones-vector then gathering each edge's own
            # target's count. Only ever gathered at indices that just
            # received a real edge, so this is always >= 1 -- no
            # divide-by-zero guard is reachable here, but a node that
            # receives no edges at all under any relation simply never
            # enters this loop body, and still gets its self-loop term.
            in_degree = torch.zeros(n_entities, device=device, dtype=node_features.dtype)
            in_degree.index_add_(0, t_idx, torch.ones_like(t_idx, dtype=node_features.dtype))
            normalized = transformed / in_degree[t_idx].clamp_min(1e-12).unsqueeze(-1)

            messages_sum.index_add_(0, t_idx, normalized)

        return messages_sum + self_term


class RGCNEncoder(nn.Module):
    """Learned per-entity input embeddings, through ``num_layers`` relational
    graph conv layers, producing one ``hidden_dim``-wide representation per
    entity. ReLU between layers; none after the last (the output feeds a
    bilinear decoder, not another layer)."""

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        hidden_dim: int,
        num_bases: int,
        num_layers: int = 2,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if n_relations < 1:
            raise ValueError(f"n_relations must be at least 1, got {n_relations}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_relations_mp = 2 * n_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.entity_embeddings = nn.Embedding(n_entities, hidden_dim)
        self.layers = nn.ModuleList(
            _BasisRelationLayer(hidden_dim, hidden_dim, self.n_relations_mp, num_bases) for _ in range(num_layers)
        )

    def forward(
        self,
        edge_h: torch.Tensor,
        edge_r: torch.Tensor,
        edge_t: torch.Tensor,
        n_entities: int,
    ) -> torch.Tensor:
        """One full-graph message-passing forward pass. ``edge_r`` must
        already be in the doubled ``[0, 2 * self.n_relations)`` space (see
        ``build_message_passing_edges``) -- this function does not double
        edges itself, keeping edge construction and the model's forward
        pass independently testable.

        **Gradient checkpointing, and why it's here.** Measured on real
        PrimeKG-scale data (see
        ``ml/experiments/rgcn_gpu_feasibility_profile_9b65b33.json``), a
        plain (non-checkpointed) forward through this per-relation,
        multi-layer loop retains every relation's intermediate tensors in
        memory from the moment they're created until whatever later call
        eventually backpropagates through them -- which, in the training
        loop this feeds, can be much later. That retention window, not
        anything about mini-batching, is what drove peak GPU memory to
        scale directly with total edge count and OOM well before full
        scale. Wrapping each layer's forward in
        ``torch.utils.checkpoint.checkpoint`` changes nothing about what
        is computed (the recomputed forward is bit-for-bit the same
        deterministic function; there is no dropout or other randomness
        in this layer to worry about) -- it only changes *when*: the
        initial forward pass runs without tracking gradients at all (so
        nothing from the per-relation loop is retained), and the full
        per-relation graph is rebuilt, used, and immediately discarded
        only once, right when backward actually needs it for that one
        layer. Disable via ``use_gradient_checkpointing=False`` (e.g. for
        a quick numerical cross-check against the unchanged math).
        """
        features = self.entity_embeddings.weight
        for layer_index, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and features.requires_grad and torch.is_grad_enabled():
                features = checkpoint(layer, features, edge_h, edge_r, edge_t, n_entities, use_reentrant=False)
            else:
                features = layer(features, edge_h, edge_r, edge_t, n_entities)
            if layer_index < self.num_layers - 1:
                features = torch.relu(features)
        return features

    def forward_sampled(self, edge_h: torch.Tensor, edge_r: torch.Tensor, edge_t: torch.Tensor, local_to_global: torch.Tensor) -> torch.Tensor:
        """Same per-layer message passing as ``forward``, but over a
        SAMPLED subgraph (see ``repurposemap.embeddings.rgcn_sampling``):
        layer-0 input is gathered from the global entity embedding table
        at just the sampled nodes (``local_to_global[i]`` = global entity
        id for local index ``i``), not the full table. ``edge_h``/
        ``edge_r``/``edge_t`` must already be in local-subgraph indices
        and the doubled message-passing relation space -- same contract
        as ``forward``, just over fewer nodes. The per-layer
        ``_BasisRelationLayer`` itself needed no change at all: it was
        already generic over however many nodes it's given.
        """
        n_local_nodes = local_to_global.shape[0]
        features = self.entity_embeddings(local_to_global)
        for layer_index, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and features.requires_grad and torch.is_grad_enabled():
                features = checkpoint(layer, features, edge_h, edge_r, edge_t, n_local_nodes, use_reentrant=False)
            else:
                features = layer(features, edge_h, edge_r, edge_t, n_local_nodes)
            if layer_index < self.num_layers - 1:
                features = torch.relu(features)
        return features


class RGCN(nn.Module):
    """R-GCN encoder + DistMult decoder (Schlichtkrull et al., 2017's
    standard link-prediction pairing).

    The decoder's relation table covers only the real ``n_relations`` --
    there are no DistMult embeddings for the encoder's synthetic inverse
    relations; those exist only for message passing, inside ``encoder``.
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        hidden_dim: int,
        num_bases: int,
        num_layers: int = 2,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = RGCNEncoder(
            n_entities, n_relations, hidden_dim, num_bases, num_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.relation_embeddings = nn.Embedding(n_relations, hidden_dim)

    def encode(self, edge_h: torch.Tensor, edge_r: torch.Tensor, edge_t: torch.Tensor, n_entities: int) -> torch.Tensor:
        """Convenience wrapper over ``self.encoder(...)`` -- full-graph mode."""
        return self.encoder(edge_h, edge_r, edge_t, n_entities)

    def encode_sampled(self, edge_h: torch.Tensor, edge_r: torch.Tensor, edge_t: torch.Tensor, local_to_global: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper over ``self.encoder.forward_sampled(...)`` --
        sampled-subgraph mode. Returns embeddings for the sampled nodes
        only, same global embedding table, gathered at `local_to_global`."""
        return self.encoder.forward_sampled(edge_h, edge_r, edge_t, local_to_global)

    def score(self, entity_embeddings: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """DistMult: ``sum(h_emb * relation_emb * t_emb)``. Higher is more
        plausible. Takes already-computed ``entity_embeddings`` (this
        model's ``encoder`` output for the current step) rather than
        recomputing them -- never call ``self.encode(...)`` from inside
        this method."""
        h_emb = entity_embeddings[h]
        t_emb = entity_embeddings[t]
        r_emb = self.relation_embeddings(r)
        return (h_emb * r_emb * t_emb).sum(dim=-1)
