"""Adapters that turn a source dataset's file format into the generic graph."""

from repurposemap.adapters.primekg import DEFAULT_PRIMEKG_CSV, load_primekg

__all__ = ["DEFAULT_PRIMEKG_CSV", "load_primekg"]
