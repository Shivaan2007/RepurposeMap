"""TransE knowledge-graph embedding baseline: model and training code.

Kept separate from ``repurposemap.evaluation`` on purpose: this package knows
how to build a deterministic entity/relation index, turn a graph into integer
triples, sample type-aware negatives, and train a TransE model. It knows
nothing about diseases, splits, or held-out evaluation pairs -- that glue
lives in ``repurposemap.evaluation.transe_baseline`` (a later stage of this
milestone), which imports from here but is not imported by anything here.

Stage A scope only: entity/relation indexing, triple construction from a
graph the caller already chose (in production,
``repurposemap.evaluation.training_graph.build_training_graph``'s
leakage-safe view), type-aware negative sampling, the TransE model itself,
and a toy training loop exercised only against tiny synthetic graphs. No
real-PrimeKG training, no CLI, no evaluation artifact -- those are later
stages.

``transe_model`` and ``transe_train`` import ``torch``, an optional
dependency not guaranteed to be installed in every environment that uses
the rest of this package (entity/relation indexing, triple construction,
and negative sampling are pure Python + NumPy). They are deliberately **not
imported here** -- this package re-exports only the torch-free modules, so
``import repurposemap.embeddings`` always succeeds regardless of whether
torch is installed, and a genuine bug inside ``transe_model``/``transe_train``
(an ``ImportError`` that has nothing to do with torch being absent) is never
swallowed by a broad ``except ImportError`` here. Code that needs the model
or training pieces imports them directly:
``from repurposemap.embeddings.transe_model import TransE`` /
``from repurposemap.embeddings.transe_train import train_toy, ...``.
"""

from repurposemap.embeddings.entity_index import (
    EntityIndex,
    RelationIndex,
    build_entity_index,
    build_relation_index,
    resolve_relation_key,
)
from repurposemap.embeddings.negative_sampling import TypeAwareNegativeSampler, TypePools, build_type_pools
from repurposemap.embeddings.positive_store import (
    PositiveTripleStore,
    build_positive_store,
    encode_triple,
    encode_triples_array,
)
from repurposemap.embeddings.triples import TripleSet, build_triples

__all__ = [
    "EntityIndex",
    "RelationIndex",
    "build_entity_index",
    "build_relation_index",
    "resolve_relation_key",
    "TypeAwareNegativeSampler",
    "TypePools",
    "build_type_pools",
    "PositiveTripleStore",
    "build_positive_store",
    "encode_triple",
    "encode_triples_array",
    "TripleSet",
    "build_triples",
]
