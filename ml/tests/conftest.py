from collections.abc import Callable
from pathlib import Path

import pytest

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "sample" / "synthetic_demo_graph.csv"


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
