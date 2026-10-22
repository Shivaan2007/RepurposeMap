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

    def __init__(
        self, in_dim: int, out_dim: int, n_relations_mp: int, num_bases: int,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if num_bases < 1:
            raise ValueError(f"num_bases must be at least 1, got {num_bases}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations_mp = n_relations_mp
        self.num_bases = num_bases
        self.use_gradient_checkpointing = use_gradient_checkpointing

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

    @staticmethod
    def _relation_messages(
        node_features: torch.Tensor, h_idx: torch.Tensor, weight: torch.Tensor,
        t_idx: torch.Tensor, n_entities: int,
    ) -> torch.Tensor:
        """One relation's normalized, un-scattered messages -- the part
        that's expensive to keep (``node_features[h_idx]``, a gathered
        slice sized by that relation's own edge count, is retained for
        backward by the matmul it feeds into). Checkpointed per-relation
        below so at most one relation's worth of this is ever retained at
        once, not all of them simultaneously for the whole epoch."""
        transformed = node_features[h_idx] @ weight
        # Relation-wise mean normalization: each edge's contribution is
        # divided by how many edges of *this relation* point into its
        # target -- the standard R-GCN normalization, computed by
        # scattering a ones-vector then gathering each edge's own target's
        # count. Only ever gathered at indices that just received a real
        # edge, so this is always >= 1 -- no divide-by-zero guard is
        # reachable here.
        in_degree = torch.zeros(n_entities, device=node_features.device, dtype=node_features.dtype)
        in_degree.index_add_(0, t_idx, torch.ones_like(t_idx, dtype=node_features.dtype))
        return transformed / in_degree[t_idx].clamp_min(1e-12).unsqueeze(-1)

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

        checkpoint_this = self.use_gradient_checkpointing and node_features.requires_grad and torch.is_grad_enabled()

        for relation_id in range(self.n_relations_mp):
            mask = edge_r == relation_id
            if not torch.any(mask):
                continue
            h_idx = edge_h[mask]
            t_idx = edge_t[mask]
            weight = all_weights[relation_id]

            # Checkpointed PER RELATION, not per layer: a node that
            # receives no edges at all under any relation simply never
            # enters this loop body, and still gets its self-loop term.
            # See ``_relation_messages``'s docstring for why this
            # granularity (not coarser, per-layer checkpointing) is what
            # actually bounds peak memory -- confirmed empirically (see
            # ml/experiments/rgcn_gpu_feasibility_profile_a998448.json):
            # per-layer checkpointing alone only removed cross-layer
            # duplication (~25-28% reduction), because a single
            # recompute-then-backward call for a whole layer still needed
            # every one of that layer's relations' gathered input slices
            # alive simultaneously. Checkpointing each relation
            # individually means at most one relation's slice is ever
            # retained at a time.
            if checkpoint_this:
                normalized = checkpoint(self._relation_messages, node_features, h_idx, weight, t_idx, n_entities, use_reentrant=False)
            else:
                normalized = self._relation_messages(node_features, h_idx, weight, t_idx, n_entities)

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
            _BasisRelationLayer(
                hidden_dim, hidden_dim, self.n_relations_mp, num_bases,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            for _ in range(num_layers)
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

        Gradient checkpointing (see ``_BasisRelationLayer._relation_messages``'s
        docstring) is applied per relation, inside each layer -- not here,
        at the per-layer level. An earlier version checkpointed whole
        layers instead; measured on real PrimeKG-scale data (see
        ``ml/experiments/rgcn_gpu_feasibility_profile_a998448.json``),
        that only removed cross-layer duplication (~25-28% peak memory
        reduction) because one recompute-then-backward call for an entire
        layer still needed every one of that layer's relations' gathered
        input slices alive simultaneously. Per-relation checkpointing
        bounds that to one relation's slice at a time.
        """
        features = self.entity_embeddings.weight
        for layer_index, layer in enumerate(self.layers):
            features = layer(features, edge_h, edge_r, edge_t, n_entities)
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
        """Convenience wrapper over ``self.encoder(...)``."""
        return self.encoder(edge_h, edge_r, edge_t, n_entities)

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
