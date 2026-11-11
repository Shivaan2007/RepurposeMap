"""ml/scripts/fill_final_results.py -- mechanical insertion of R-GCN's
locked-test numbers into docs/FINAL_RESULTS.md's comparison table. Pure
string/JSON logic, no torch, no real model, no real numbers below."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fill_final_results import (  # noqa: E402
    ArtifactNotLockedTestError,
    RowNotFoundError,
    _TBD_ROW,
    build_rgcn_row,
    fill_final_results,
)

_LOCKED_TEST_ARTIFACT = {
    "split": "locked-test",
    "n_pairs": 498,
    "hits_at_1": 0.012345,
    "hits_at_3": 0.023456,
    "hits_at_10": 0.045678,
    "mrr": 0.019876,
    "macro_auprc": 0.033344,
    "micro_auprc": 0.028811,
}


def test_build_rgcn_row_formats_to_four_decimals():
    row = build_rgcn_row(_LOCKED_TEST_ARTIFACT)
    assert row == "| **Final R-GCN (full universe)** | 0.0123 | 0.0235 | 0.0457 | 0.0199 | 0.0333 |"


def test_build_rgcn_row_rejects_a_validation_split_artifact():
    validation_artifact = dict(_LOCKED_TEST_ARTIFACT, split="validation")
    with pytest.raises(ArtifactNotLockedTestError):
        build_rgcn_row(validation_artifact)


def test_fill_final_results_replaces_the_exact_tbd_row(tmp_path):
    final_results = tmp_path / "FINAL_RESULTS.md"
    final_results.write_text(f"# header\n\nsome text\n\n{_TBD_ROW}\n\nmore text\n")

    updated = fill_final_results(_LOCKED_TEST_ARTIFACT, final_results)

    assert _TBD_ROW not in updated
    assert "0.0123" in updated
    assert "some text" in updated and "more text" in updated


def test_fill_final_results_raises_if_tbd_row_already_replaced(tmp_path):
    final_results = tmp_path / "FINAL_RESULTS.md"
    final_results.write_text("# header\n\nalready filled in, no TBD row here\n")

    with pytest.raises(RowNotFoundError):
        fill_final_results(_LOCKED_TEST_ARTIFACT, final_results)


def test_fill_final_results_against_the_real_docs_file():
    """The actual file this is meant to patch -- proves the frozen TBD
    row text in docs/FINAL_RESULTS.md matches what this script looks
    for, so a future real run doesn't fail on a docs/code drift."""
    real_path = Path(__file__).resolve().parents[2] / "docs" / "FINAL_RESULTS.md"
    updated = fill_final_results(_LOCKED_TEST_ARTIFACT, real_path)
    assert "0.0123" in updated
