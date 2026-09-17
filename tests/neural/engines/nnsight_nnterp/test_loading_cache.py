"""``load_model`` caches its bundles (``normalized_cache``, four entries);
pinned so an edit that unwraps the cache fails a test."""

from __future__ import annotations

import pytest

from causalab.neural.engines.nnsight_nnterp import loading

pytestmark = pytest.mark.unit


def test_load_model_is_a_four_entry_cache() -> None:
    assert hasattr(loading.load_model, "cache_info")
    assert loading.load_model.cache_info().maxsize == 4
