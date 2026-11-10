import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "sample" / "synthetic_demo_graph.csv"

# These test files import torch unconditionally at module level (CLI/embeddings
# coverage). torch is in the optional "ml" extra, not "dev" -- see pyproject.toml --
# so a plain `pip install -e ".[dev]"` + `python -m pytest`, exactly as README.md's
# "Setup"/"Run the tests" sections document, would otherwise fail at collection
# with ModuleNotFoundError instead of running the rest of the suite. Skip only
# when torch really is unavailable; once it's installed (`.[ml]` extra) these
# collect and run normally.
_TORCH_ONLY_TEST_FILES = {
    "test_cli.py",
    "test_primekg_cli.py",
    "test_rgcn_cli.py",
    "test_rgcn_sampling.py",
    "test_scoring_cli.py",
    "test_transe_cli.py",
}

if importlib.util.find_spec("torch") is None:
    collect_ignore = list(_TORCH_ONLY_TEST_FILES)


@pytest.fixture
def sample_csv() -> Path:
    return SAMPLE_CSV


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[[str], Path]:
    """Write CSV text to a temporary file and return its path."""

    def _write(text: str) -> Path:
        path = tmp_path / "edges.csv"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def primekg_csv() -> Path:
    """Tiny synthetic fixture in the PrimeKG schema. The full kg.csv is not needed for tests."""
    return Path(__file__).resolve().parent / "fixtures" / "primekg_mini.csv"
