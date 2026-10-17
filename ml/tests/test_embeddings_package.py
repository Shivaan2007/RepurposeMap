"""The embeddings package's torch-optionality contract (Stage A fix #2).

``repurposemap.embeddings`` (the package's ``__init__.py``) must import
cleanly regardless of whether torch is installed, and must never swallow a
genuine bug inside ``transe_model``/``transe_train`` behind a broad
``except ImportError``. The fix: those two modules are not imported by
``__init__.py`` at all -- callers import them directly. This is verified
without torch by `.venv` (torch is not installed in the project's own
`.venv`; the previous broad `try/except ImportError` masked this, the
narrower architecture does not need to).
"""

from __future__ import annotations

import repurposemap.embeddings as embeddings_package


def test_the_package_itself_exposes_only_the_torch_free_api():
    # transe_model/transe_train are deliberately NOT re-exported here --
    # callers import them directly from their own submodules, so a real
    # ImportError inside those modules is never hidden by this package's
    # own import statements.
    assert "TransE" not in embeddings_package.__all__
    assert "train_toy" not in embeddings_package.__all__
    assert "TransETrainingConfig" not in embeddings_package.__all__


def test_the_torch_free_api_is_fully_importable():
    from repurposemap.embeddings import (  # noqa: F401
        EntityIndex,
        RelationIndex,
        TripleSet,
        TypeAwareNegativeSampler,
        TypePools,
        build_entity_index,
        build_relation_index,
        build_triples,
        build_type_pools,
        resolve_relation_key,
    )
