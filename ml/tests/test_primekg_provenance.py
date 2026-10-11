"""PrimeKG dataset provenance: read-only file fingerprinting.

Uses a tiny synthetic file written by the test, never the real ~980 MB
kg.csv — hashing is cheap to test correctly without it.
"""

from __future__ import annotations

import hashlib

import pytest

from repurposemap.adapters.primekg import PrimeKGFingerprint, compute_file_fingerprint


def test_fingerprint_matches_a_manually_computed_sha256(tmp_path):
    path = tmp_path / "tiny.csv"
    content = b"relation,display_relation\nindication,indication\n"
    path.write_bytes(content)

    fingerprint = compute_file_fingerprint(path)

    assert fingerprint.sha256 == hashlib.sha256(content).hexdigest()
    assert fingerprint.size_bytes == len(content)


def test_fingerprint_is_stable_across_repeated_calls(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_bytes(b"some bytes, hashed twice, must match")

    first = compute_file_fingerprint(path)
    second = compute_file_fingerprint(path)

    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes


def test_fingerprinting_does_not_modify_the_file(tmp_path):
    path = tmp_path / "tiny.csv"
    original_bytes = b"untouched content"
    path.write_bytes(original_bytes)
    mtime_before = path.stat().st_mtime

    compute_file_fingerprint(path)

    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime == mtime_before


def test_fingerprint_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_file_fingerprint(tmp_path / "does-not-exist.csv")


def test_fingerprint_handles_a_file_larger_than_one_hash_chunk(tmp_path):
    # Exercises the chunked-read loop, not just a single read() call.
    from repurposemap.adapters.primekg import HASH_CHUNK_BYTES

    path = tmp_path / "bigger.csv"
    content = b"x" * (HASH_CHUNK_BYTES + 1337)
    path.write_bytes(content)

    fingerprint = compute_file_fingerprint(path)
    assert fingerprint.sha256 == hashlib.sha256(content).hexdigest()
    assert fingerprint.size_bytes == len(content)


def test_provenance_fields_are_honest_about_what_is_not_known():
    fingerprint = PrimeKGFingerprint(sha256="abc", size_bytes=1)
    # Known by hand, recorded in ml/README.md's data section.
    assert fingerprint.retrieved_date == "2026-10-05"
    assert fingerprint.dataverse_doi == "doi:10.7910/DVN/IXA7BM"
    # Not invented: neither is available anywhere local to this repository.
    assert fingerprint.release_version is None
    assert fingerprint.snapshot_build_date is None


def test_fingerprint_has_no_filesystem_mtime_field():
    # mtime is not dataset provenance; it must not be able to stand in for one.
    fingerprint = PrimeKGFingerprint(sha256="abc", size_bytes=1)
    assert not hasattr(fingerprint, "mtime")
