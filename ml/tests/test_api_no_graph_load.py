"""Neither /health nor /api/v1/system/info may load the PrimeKG graph or
hash the raw CSV -- both are explicitly meant to answer instantly (Task D)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from repurposemap.api.app import app  # noqa: E402


def test_health_and_system_info_never_call_load_primekg_or_fingerprint(monkeypatch):
    import repurposemap.adapters.primekg as primekg_module

    def _poisoned(*_args, **_kwargs):
        raise AssertionError("load_primekg/compute_file_fingerprint must never be called for health/info routes")

    monkeypatch.setattr(primekg_module, "load_primekg", _poisoned)
    monkeypatch.setattr(primekg_module, "compute_file_fingerprint", _poisoned)

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/system/info").status_code == 200
    # If either poisoned function had been called, the AssertionError above
    # would have propagated as a 500 (or raised directly) -- reaching here
    # with both 200s is the proof.
