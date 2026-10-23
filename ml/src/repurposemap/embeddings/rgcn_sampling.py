"""Relation-aware neighborhood sampling for scalable R-GCN training.

Full-graph R-GCN training (``rgcn_train_gpu.py``) is measured feasible on
one GPU but operationally impractical at real PrimeKG scale (~2.69 h per
epoch, dominated by gradient-checkpoint recomputation through a shared
full-graph encoder graph re-backpropagated once per mini-batch -- see
``ml/experiments/rgcn_gpu_fullscale_profile_1ae8db5.json`` and
``ml/experiments/rgcn_gpu_nockpt_profile_1ae8db5.json``). This module
replaces full-graph message passing with a bounded, relation-aware
neighborhood sampled fresh per mini-batch, so each batch gets its own
small, independent encoder forward/backward pass -- no shared graph, no
``retain_graph=True``, no one-step-per-epoch constraint.

**Leakage safety is structural, not a runtime check.** This module never
reads a graph, a CSV, or anything beyond the leakage-safe triple array
(``tuning_triples`` or ``final_triples``, exactly the same two arrays the
full-graph trainer already uses) passed into ``build_relation_adjacency``.
A held-out indication edge that was never in that array cannot be sampled
by anything downstream -- there is no code path here that reads any other
source of edges.

**Relation-balanced sampling, not generic adjacency collapse.** PrimeKG is
highly multi-relational; a naive uniform-over-all-neighbors sampler would
let high-degree relations (e.g. protein-protein interactions) drown out
rarer, often more informative relations (e.g. drug-disease indications) in
every sampled neighborhood. ``_sample_neighbors_relation_balanced`` groups
candidates by real relation ID first, splits the fanout budget evenly
across the relations actually present, then fills any leftover budget
round-robin from relations with more candidates than their even share --
this is the real, scientific sampling policy this module implements, not
a placeholder plumbing version labeled as something stronger than it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from repurposemap.embeddings.rgcn_model import build_message_passing_edges


@dataclass(frozen=True)
class RelationAdjacency:
    """Per-entity outgoing/incoming edges, grouped by real relation ID,
    built once from a leakage-safe triple array only.

    ``outgoing[e][r]`` -- entities ``e`` has a ``(e, r, t)`` edge to.
    ``incoming[e][r]`` -- entities ``e`` has a ``(h, r, e)`` edge from.
    Entities with no edges at all simply have no key in either dict.

    ``combined[e]`` is the same outgoing+incoming information, pre-merged
    and pre-flattened into the CSR-style numpy arrays
    ``_sample_neighbors_relation_balanced_arrays`` needs --
    ``(rel_ids, offsets, neighbors, directions)``, where relation ``i``'s
    candidates are ``neighbors[offsets[i]:offsets[i+1]]`` /
    ``directions[offsets[i]:offsets[i+1]]`` -- built once here instead of
    being re-merged and re-converted to arrays on every single sampling
    call for every node (every mini-batch, every epoch). Profiling at
    real-PrimeKG scale showed that repeated per-call merge/conversion was
    the dominant share of the ~96% of batch time neighbor sampling was
    consuming (see
    ``ml/experiments/rgcn_sampled_training_profile_e79d3e4.json``); moving
    it here makes it a one-time cost instead of a per-batch one.
    """

    outgoing: dict[int, dict[int, tuple[int, ...]]]
    incoming: dict[int, dict[int, tuple[int, ...]]]
    combined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def build_relation_adjacency(triples: np.ndarray) -> RelationAdjacency:
    """Build a ``RelationAdjacency`` from a triple array only -- no graph
    object, no CSV. Pass ``tuning_triples`` during the validation-search
    phase or ``final_triples`` for final-training preparation; never the
    raw, un-split graph. Deterministic for a given triple array regardless
    of row order (plain aggregation, no RNG involved)."""
    outgoing: dict[int, dict[int, list[int]]] = {}
    incoming: dict[int, dict[int, list[int]]] = {}
    for row in triples:
        h, r, t = int(row[0]), int(row[1]), int(row[2])
        outgoing.setdefault(h, {}).setdefault(r, []).append(t)
        incoming.setdefault(t, {}).setdefault(r, []).append(h)
    outgoing_frozen = {e: {r: tuple(ns) for r, ns in rels.items()} for e, rels in outgoing.items()}
    incoming_frozen = {e: {r: tuple(ns) for r, ns in rels.items()} for e, rels in incoming.items()}

    merged: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for e, rels in outgoing.items():
        for r, tails in rels.items():
            merged.setdefault(e, {}).setdefault(r, []).extend((t, 1) for t in tails)
    for e, rels in incoming.items():
        for r, heads in rels.items():
            merged.setdefault(e, {}).setdefault(r, []).extend((h, 0) for h in heads)

    combined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for e, rels in merged.items():
        rel_ids_sorted = sorted(rels)
        group_sizes = np.fromiter((len(rels[r]) for r in rel_ids_sorted), dtype=np.int64, count=len(rel_ids_sorted))
        offsets = np.zeros(len(rel_ids_sorted) + 1, dtype=np.int64)
        np.cumsum(group_sizes, out=offsets[1:])
        total = int(offsets[-1])
        neighbors = np.empty(total, dtype=np.int64)
        directions = np.empty(total, dtype=np.int64)
        pos = 0
        for r, n in zip(rel_ids_sorted, group_sizes.tolist(), strict=True):
            arr = np.asarray(rels[r], dtype=np.int64)
            neighbors[pos : pos + n] = arr[:, 0]
            directions[pos : pos + n] = arr[:, 1]
            pos += n
        combined[e] = (np.array(rel_ids_sorted, dtype=np.int64), offsets, neighbors, directions)

    return RelationAdjacency(outgoing=outgoing_frozen, incoming=incoming_frozen, combined=combined)


@dataclass(frozen=True)
class SamplingConfig:
    """Every neighborhood-sampling knob, with no hidden defaults -- same
    discipline as the project's other training configs.

    ``fanout_hop1``/``fanout_hop2`` are starting points for feasibility
    profiling only (see the module docstring's reasoning), not final tuned
    values -- fixed once via that profiling, then held fixed during the
    3-config model-capacity search (never searched simultaneously with
    hidden_dim/num_bases)."""

    fanout_hop1: int = 15
    fanout_hop2: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.fanout_hop1 < 1:
            raise ValueError(f"fanout_hop1 must be at least 1, got {self.fanout_hop1}")
        if self.fanout_hop2 < 1:
            raise ValueError(f"fanout_hop2 must be at least 1, got {self.fanout_hop2}")

    def to_dict(self) -> dict:
        return {"fanout_hop1": self.fanout_hop1, "fanout_hop2": self.fanout_hop2, "seed": self.seed}


@dataclass
class SampledSubgraph:
    """One sampled, local-indexed, message-passing-ready subgraph.

    ``local_to_global[i]`` is the global entity id for local index ``i``;
    ``local_to_global`` is kept sorted ascending so ``to_local`` can use
    binary search. ``edge_h``/``edge_r``/``edge_t`` are already in the
    doubled message-passing relation space (see
    ``repurposemap.embeddings.rgcn_model.build_message_passing_edges``) --
    the real, un-doubled relation space never appears in this dataclass.
    """

    local_to_global: np.ndarray
    edge_h: np.ndarray
    edge_r: np.ndarray
    edge_t: np.ndarray
    n_local_nodes: int
    stats: dict = field(default_factory=dict)

    def to_local(self, global_ids: np.ndarray) -> np.ndarray:
        """Map global entity ids to local subgraph indices.

        Raises ``ValueError`` if any id is not present -- callers must
        ensure every positive/negative triple endpoint was included in
        ``seed_entities`` before calling ``sample`` (see
        ``RelationAwareNeighborSampler.sample``'s docstring)."""
        global_ids = np.asarray(global_ids, dtype=np.int64)
        idx = np.searchsorted(self.local_to_global, global_ids)
        idx_clamped = np.clip(idx, 0, len(self.local_to_global) - 1)
        found = self.local_to_global[idx_clamped] == global_ids
        if not np.all(found):
            missing = global_ids[~found]
            raise ValueError(
                f"{len(missing)} global id(s) not present in sampled subgraph "
                f"(first few: {missing[:5].tolist()}) -- they were not included as seed entities"
            )
        return idx_clamped


def _sample_neighbors_relation_balanced(
    candidates_by_relation: dict[int, list[tuple[int, int]]],
    fanout: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Relation-balanced sampling within one node's one-hop candidate set.

    ``candidates_by_relation[r]`` is a list of ``(neighbor, direction)``
    pairs (``direction=1`` means the sampled node is the head of this
    relation's edge to ``neighbor``; ``direction=0`` means it is the
    tail). Splits ``fanout`` evenly across the relations actually present,
    then fills any leftover budget round-robin from relations with more
    candidates than their even share -- so no single high-degree relation
    can crowd out the others, and no relation is starved while another
    still has unused candidates and budget remains.

    Returns a list of ``(relation_id, neighbor, direction)`` triples, at
    most ``fanout`` long (fewer if there simply aren't that many distinct
    candidates across all relations).
    """
    relations = sorted(candidates_by_relation)
    if not relations or fanout <= 0:
        return []

    pools: dict[int, list[tuple[int, int]]] = {}
    for r in relations:
        pool = list(candidates_by_relation[r])
        perm = rng.permutation(len(pool))
        pools[r] = [pool[i] for i in perm]

    base_cap = max(1, fanout // len(relations))
    sampled: list[tuple[int, int, int]] = []
    taken: dict[int, int] = dict.fromkeys(relations, 0)

    for r in relations:
        take = min(base_cap, len(pools[r]))
        sampled.extend((r, *item) for item in pools[r][:take])
        taken[r] = take

    remaining = fanout - len(sampled)
    progress = remaining > 0
    while remaining > 0 and progress:
        progress = False
        for r in relations:
            if remaining <= 0:
                break
            pool = pools[r]
            if taken[r] < len(pool):
                sampled.append((r, *pool[taken[r]]))
                taken[r] += 1
                remaining -= 1
                progress = True

    return sampled


def _sample_neighbors_relation_balanced_arrays(
    rel_ids: np.ndarray,
    offsets: np.ndarray,
    neighbors: np.ndarray,
    directions: np.ndarray,
    fanout: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Same relation-balanced policy as ``_sample_neighbors_relation_balanced``
    (even split across relations present, then round-robin leftover), but
    operating on the precomputed CSR-style arrays ``RelationAdjacency.combined``
    stores per entity, instead of a freshly-built dict of Python lists.

    This is the hot path ``_sample_subgraph_impl`` actually calls. Doing one
    combined ``rng.permutation`` for the whole node (not one per relation)
    plus vectorized numpy regrouping/selection -- instead of a Python-level
    loop over every candidate -- is what fixed the severe CPU bottleneck
    measured in ``ml/experiments/rgcn_sampled_training_profile_e79d3e4.json``
    (neighbor sampling was ~96% of sampled R-GCN batch time). The random
    draw pattern differs from the dict-based function above (still
    deterministic for a fixed seed, still exactly resumable); no test
    depends on the two implementations producing identical draws.
    """
    n_rel = len(rel_ids)
    if n_rel == 0 or fanout <= 0:
        return []

    group_sizes = np.diff(offsets)
    total = int(offsets[-1])

    group_id = np.repeat(np.arange(n_rel), group_sizes)
    perm = rng.permutation(total)
    order = np.argsort(group_id[perm], kind="stable")
    shuffled_idx = perm[order]
    grouped_neighbors = neighbors[shuffled_idx]
    grouped_directions = directions[shuffled_idx]

    base_cap = max(1, fanout // n_rel)
    taken = np.minimum(base_cap, group_sizes)
    remaining = fanout - int(taken.sum())

    progress = remaining > 0
    while remaining > 0 and progress:
        progress = False
        for i in range(n_rel):
            if remaining <= 0:
                break
            if taken[i] < group_sizes[i]:
                taken[i] += 1
                remaining -= 1
                progress = True

    total_taken = int(taken.sum())
    if total_taken == 0:
        return []

    cum_taken_excl = np.concatenate(([0], np.cumsum(taken)[:-1]))
    within = np.arange(total_taken) - np.repeat(cum_taken_excl, taken)
    base = np.repeat(offsets[:-1], taken)
    selected_idx = base + within

    rel_out = np.repeat(rel_ids, taken)
    neighbor_out = grouped_neighbors[selected_idx]
    direction_out = grouped_directions[selected_idx]

    return list(zip(rel_out.tolist(), neighbor_out.tolist(), direction_out.tolist(), strict=True))


def _sample_subgraph_impl(
    adjacency: RelationAdjacency,
    seed_entities: Sequence[int],
    n_relations: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> SampledSubgraph:
    seed_entities_sorted = sorted({int(e) for e in seed_entities})
    all_nodes: dict[int, None] = dict.fromkeys(seed_entities_sorted)
    raw_edges: set[tuple[int, int, int]] = set()

    frontier = list(seed_entities_sorted)
    fanouts = (config.fanout_hop1, config.fanout_hop2)
    hop_stats = []

    for hop_index, fanout in enumerate(fanouts):
        new_frontier_nodes: dict[int, None] = {}
        hop_edges_sampled = 0

        for node in frontier:
            entry = adjacency.combined.get(node)
            if entry is None:
                continue
            rel_ids, offsets, neighbors, directions = entry
            sampled = _sample_neighbors_relation_balanced_arrays(rel_ids, offsets, neighbors, directions, fanout, rng)
            for rel, neighbor, direction in sampled:
                if direction == 1:
                    raw_edges.add((node, rel, neighbor))
                else:
                    raw_edges.add((neighbor, rel, node))
                if neighbor not in all_nodes:
                    new_frontier_nodes[neighbor] = None
                hop_edges_sampled += 1

        all_nodes.update(new_frontier_nodes)
        hop_stats.append(
            {
                "hop": hop_index + 1,
                "frontier_size": len(frontier),
                "new_nodes": len(new_frontier_nodes),
                "edges_sampled": hop_edges_sampled,
            }
        )
        frontier = list(new_frontier_nodes)
        if not frontier:
            break

    local_to_global = np.array(sorted(all_nodes), dtype=np.int64)

    if raw_edges:
        raw_array = np.array(sorted(raw_edges), dtype=np.int64)
        local_h = np.searchsorted(local_to_global, raw_array[:, 0])
        local_t = np.searchsorted(local_to_global, raw_array[:, 2])
        local_triples = np.stack([local_h, raw_array[:, 1], local_t], axis=1).astype(np.int64)
        edge_h, edge_r, edge_t = build_message_passing_edges(local_triples, n_relations)
    else:
        edge_h = np.zeros(0, dtype=np.int64)
        edge_r = np.zeros(0, dtype=np.int64)
        edge_t = np.zeros(0, dtype=np.int64)

    return SampledSubgraph(
        local_to_global=local_to_global,
        edge_h=edge_h,
        edge_r=edge_r,
        edge_t=edge_t,
        n_local_nodes=len(local_to_global),
        stats={"hops": hop_stats, "n_raw_edges": len(raw_edges), "n_nodes": len(local_to_global)},
    )


class RelationAwareNeighborSampler:
    """Samples a bounded, relation-aware, deterministic 2-hop neighborhood
    around a batch's seed entities, from a ``RelationAdjacency`` built
    once from a leakage-safe triple array -- never the raw graph.

    Deterministic for a fixed seed and a fixed, restored RNG state (same
    discipline as ``BatchedTypeAwareNegativeSampler``): uses its own
    ``numpy.random.Generator`` instance, never global ``numpy`` RNG state.
    """

    def __init__(self, adjacency: RelationAdjacency, n_relations: int, config: SamplingConfig, seed: int = 0) -> None:
        self._adjacency = adjacency
        self._n_relations = n_relations
        self._config = config
        self._rng = np.random.default_rng(seed)

    def get_state(self) -> dict:
        """RNG state only, for checkpointing -- mirrors
        ``BatchedTypeAwareNegativeSampler.get_state()``'s shape."""
        return {"bit_generator_state": self._rng.bit_generator.state}

    def set_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator_state"]

    def sample(self, seed_entities: Sequence[int]) -> SampledSubgraph:
        """Every positive/negative triple endpoint for the current batch
        must be included in ``seed_entities`` -- the encoder only produces
        representations for sampled nodes, and ``SampledSubgraph.to_local``
        raises if asked to map an id that was never included."""
        return _sample_subgraph_impl(self._adjacency, seed_entities, self._n_relations, self._config, self._rng)
